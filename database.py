from sqlalchemy import Column, Integer, String, Date, Text, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./review.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ext_id = Column(String, index=True, nullable=True)
    location = Column(String, index=True, nullable=True)   
    rating = Column(Integer, nullable=True)                
    text = Column(Text, nullable=False)                    
    date = Column(Date, default=datetime.date.today)
    sentiment = Column(String, index=True, nullable=True)
    topic = Column(String, index=True, nullable=True)

    ai_reply = relationship("AIReply", uselist=False, back_populates="review")

class AIReply(Base):
    __tablename__ = "ai_replies"

    id = Column(Integer, primary_key=True)
    review_id = Column(Integer, ForeignKey("reviews.id"), unique=True)
    reply = Column(Text, nullable=False)
    tags = Column(String, nullable=True)
    reasoning_log = Column(Text, nullable=True)
    created_at = Column(Date, default=datetime.date.today)

    review = relationship("Review", back_populates="ai_reply")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
