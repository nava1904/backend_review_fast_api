from pydantic import BaseModel, Field
from typing import Optional, Dict
import datetime
import torch
class ReviewIn(BaseModel):
    text: str                                # matches DB column "text"
    location: Optional[str] = None           # optional
    rating: Optional[int] = Field(None, ge=1, le=5)
    date: Optional[datetime.date] = None
    topic: Optional[str] = None

class AIReplyOut(BaseModel):
    reply: str
    tags: Optional[str] = None
    reasoning_log: Optional[str] = None

    class Config:
        orm_mode = True

class ReviewOut(ReviewIn):
    id: int
    sentiment: Optional[str] = None
    probability: Optional[float] = None
    ai_reply: Optional[AIReplyOut] = None

    class Config:
        orm_mode = True

class AnalyticsOut(BaseModel):
    sentiment_counts: Dict[str, int]
    topic_counts: Dict[str, int]

class SearchResultOut(BaseModel):
    id: int
    score: float
    text: str

class SentimentResult(BaseModel):
    label: str
    score: float
