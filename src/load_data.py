import json
import os
from sqlalchemy.orm import Session
from src.database import SessionLocal, Video, Chunk

def load_data():
    db: Session = SessionLocal()
    try:
        # Clear existing data to prevent duplicates if script is run multiple times
        print("Clearing existing database tables...")
        db.query(Chunk).delete()
        db.query(Video).delete()
        db.commit()

        # Load videos
        videos_file = "videos_data.json"
        print(f"Loading videos from {videos_file}...")
        with open(videos_file, "r", encoding="utf-8") as f:
            videos_data = json.load(f)
        
        print(f"Found {len(videos_data)} videos. Ingesting...")
        for item in videos_data:
            # Create video object
            video = Video(
                video_id=item["video_id"],
                title=item["title"],
                url=item["url"],
                upload_date=item.get("upload_date"),
                view_count=item.get("view_count"),
                like_count=item.get("like_count"),
                tags=item.get("tags", []),
                description=item.get("description"),
                full_text=item.get("full_text")
            )
            db.add(video)
        db.commit()
        print("Videos ingestion completed.")

        # Load chunks
        chunks_file = "chunks_with_summaries.json"
        if not os.path.exists(chunks_file):
            chunks_file = "chunks_with_embeddings.json"
        print(f"Loading chunks from {chunks_file}...")
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        print(f"Found {len(chunks_data)} chunks. Ingesting...")
        # To make it fast, we can insert in batches
        batch_size = 500
        for i in range(0, len(chunks_data), batch_size):
            batch = chunks_data[i:i+batch_size]
            for item in batch:
                chunk = Chunk(
                    video_id=item["video_id"],
                    source=item.get("source"),
                    start_time=item.get("start_time"),
                    end_time=item.get("end_time"),
                    text=item["text"],
                    embedding=item.get("embedding"),
                    summary=item.get("summary"),
                    key_points=item.get("key_points")
                )
                db.add(chunk)
            db.commit()
            print(f"  Ingested {min(i + batch_size, len(chunks_data))} / {len(chunks_data)} chunks...")
        
        print("Chunks ingestion completed.")
    except Exception as e:
        db.rollback()
        print(f"Error during ingestion: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    load_data()
