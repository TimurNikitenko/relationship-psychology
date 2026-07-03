import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from pgvector.sqlalchemy import Vector

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "postgresql://postgres:postgres@localhost:5436/relationships_psychology"

# Create engine and session maker
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Video(Base):
    __tablename__ = "videos"

    video_id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    url = Column(String, nullable=False)
    upload_date = Column(String, nullable=True)
    view_count = Column(Integer, nullable=True)
    like_count = Column(Integer, nullable=True)
    tags = Column(ARRAY(String), nullable=True)
    description = Column(Text, nullable=True)
    full_text = Column(Text, nullable=True)

    chunks = relationship("Chunk", back_populates="video", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    video_id = Column(String, ForeignKey("videos.video_id", ondelete="CASCADE"), nullable=False)
    source = Column(String, nullable=True)
    start_time = Column(Float, nullable=True)
    end_time = Column(Float, nullable=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(312), nullable=True)
    summary = Column(Text, nullable=True)
    key_points = Column(ARRAY(String), nullable=True)

    video = relationship("Video", back_populates="chunks")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
