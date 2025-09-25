import uvicorn
from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from transformers import pipeline
from fastapi import FastAPI, HTTPException, Depends, Header, Query, Body, Path
from typing import List, Optional
import datetime

from database import get_db, Review, AIReply
from schemas import (
    AnalyticsOut,
    SentimentResult,
    SearchResultOut,
    ReviewIn,
    ReviewOut,
    AIReplyOut,
)
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from init_db import init_db

app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_db()


@app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nava1904-frontend2-streamlit-pageshome-ho0oei.streamlit.app"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------
# Security
# ----------------------
API_KEY = "secret123"

def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# ----------------------
# AI Pipeline
# ----------------------
nlp = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
)

# ----------------------
# FastAPI App
# ----------------------

# ----------------------
# Schemas
# ----------------------

class SuggestReplyOut(AIReplyOut):
    review_id: int

# ----------------------
# Endpoints
# ----------------------
@app.get("/")
def get_root():
    return {"message": "Welcome to the Sentiment Analysis & Reviews API"}

@app.get("/health")
def get_health():
    return {"status": "OK"}

@app.post("/predict", response_model=SentimentResult, dependencies=[Depends(get_api_key)])
def predict_sentiment(payload: TextIn):
    """Predict sentiment for a single piece of text"""
    try:
        result = nlp(payload.text)
        return {"label": result[0]["label"], "score": result[0]["score"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

@app.post("/ingest", dependencies=[Depends(get_api_key)])
def ingest_reviews(reviews: List[ReviewIn], db: Session = Depends(get_db)):
    """Ingest multiple reviews into the database"""
    inserted_count = 0
    try:
        for r in reviews:
            sent_result = nlp(r.text)[0]
            db_review = Review(
                ext_id=None,
                location=r.location or "",
                rating=r.rating or 0,
                text=r.text,
                date=r.date or datetime.date.today(),
                sentiment=sent_result["label"],
                topic=r.topic,
            )
            db.add(db_review)
            inserted_count += 1
        db.commit()
        return {"inserted": inserted_count}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Ingest error: {e}")

@app.get("/reviews", response_model=List[ReviewOut], dependencies=[Depends(get_api_key)])
def get_reviews(
    location: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Fetch reviews with optional filters and pagination"""
    query = db.query(Review)
    if location:
        query = query.filter(Review.location.ilike(f"%{location}%"))
    if sentiment:
        query = query.filter(Review.sentiment.ilike(f"%{sentiment}%"))
    if q:
        query = query.filter(
            or_(
                Review.text.ilike(f"%{q}%"),
                Review.topic.ilike(f"%{q}%"),
            )
        )

    offset = (page - 1) * page_size
    reviews = query.offset(offset).limit(page_size).all()
    if not reviews:
        raise HTTPException(status_code=404, detail="No reviews found matching criteria")
    return reviews

@app.get("/analytics", response_model=AnalyticsOut, dependencies=[Depends(get_api_key)])
def get_analytics(db: Session = Depends(get_db)):
    """Return sentiment and topic counts across reviews"""
    sentiment_counts_query = (
        db.query(Review.sentiment, func.count(Review.id))
        .group_by(Review.sentiment)
        .all()
    )
    topic_counts_query = (
        db.query(Review.topic, func.count(Review.id))
        .group_by(Review.topic)
        .all()
    )

    sentiment_counts = {
        key if key is not None else "UNKNOWN": count
        for key, count in sentiment_counts_query
    }
    topic_counts = {
        key if key is not None else "UNKNOWN": count
        for key, count in topic_counts_query
    }

    return AnalyticsOut(sentiment_counts=sentiment_counts, topic_counts=topic_counts)

@app.get("/search", response_model=List[SearchResultOut], dependencies=[Depends(get_api_key)])
def search(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """Search similar reviews using TF-IDF + cosine similarity"""
    reviews = db.query(Review).all()
    if not reviews:
        return []

    review_texts = [r.text for r in reviews]
    review_ids = [r.id for r in reviews]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(review_texts)
    query_vec = vectorizer.transform([q])
    cosine_similarities = cosine_similarity(query_vec, vectors).flatten()
    top_k_indices = cosine_similarities.argsort()[-k:][::-1]

    results = []
    for idx in top_k_indices:
        results.append(
            SearchResultOut(
                id=review_ids[idx],
                score=float(cosine_similarities[idx]),
                text=review_texts[idx],
            )
        )
    return results



@app.get("/reviews/{review_id}", response_model=ReviewOut, dependencies=[Depends(get_api_key)])
def get_review(review_id: int = Path(..., ge=1), db: Session = Depends(get_db)):
    review = db.query(Review).filter(Review.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return review

@app.get("/reviews/{review_id}/reply", response_model=Optional[AIReplyOut], dependencies=[Depends(get_api_key)])
def get_review_reply(review_id: int = Path(..., ge=1), db: Session = Depends(get_db)):
    ai_reply = db.query(AIReply).filter(AIReply.review_id == review_id).first()
    if not ai_reply:
        return None
    return ai_reply

@app.post("/reviews/{review_id}/suggest-reply", dependencies=[Depends(get_api_key)])
def suggest_ai_reply(
    review_id: int = Path(..., ge=1),
    db: Session = Depends(get_db)
):
    review = db.query(Review).filter(Review.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    sentiment_result = nlp(review.text)[0]
    sentiment_label = sentiment_result['label']

    if sentiment_label == "POSITIVE":
        reply_text = "Thank you for your positive feedback! We are glad you had a great experience."
    elif sentiment_label == "NEGATIVE":
        reply_text = "We're sorry to hear about your experience. We will strive to improve and hope to serve you better next time."
    else:
        reply_text = "Thank you for your review. We appreciate your feedback and will use it to improve."

    tags = sentiment_label.lower()
    reasoning_log = f"Reply generated based on detected sentiment '{sentiment_label}'."

    ai_reply = db.query(AIReply).filter(AIReply.review_id == review_id).first()
    if ai_reply:
        ai_reply.reply = reply_text
        ai_reply.tags = tags
        ai_reply.reasoning_log = reasoning_log
    else:
        ai_reply = AIReply(
            review_id=review_id,
            reply=reply_text,
            tags=tags,
            reasoning_log=reasoning_log
        )
        db.add(ai_reply)

    db.commit()

    return {
        "review_id": review_id,
        "reply": reply_text,
        "tags": tags,
        "reasoning_log": reasoning_log
    }


# ----------------------
# Entry point
# ----------------------
if __name__ == "__main__":
    uvicorn.run("models:app", host="0.0.0.0", port=8000, reload=True)  
