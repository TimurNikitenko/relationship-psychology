#!/bin/sh

# Exit immediately if any command fails
set -e

echo "Starting Relationships Psychology Bot service..."

# 1. Wait for PostgreSQL to be ready
python -c "
import os, sys, time, socket
from urllib.parse import urlparse

db_url = os.getenv('DATABASE_URL')
if not db_url:
    print('Error: DATABASE_URL environment variable is not set!')
    sys.exit(1)

parsed = urlparse(db_url)
host = parsed.hostname or 'localhost'
port = parsed.port or 5432

print(f'Checking database connectivity to {host}:{port}...')
start_time = time.time()
while time.time() - start_time < 90:
    try:
        s = socket.create_connection((host, port), timeout=2)
        s.close()
        print('Database is ready to accept connections.')
        sys.exit(0)
    except (socket.error, socket.timeout):
        time.sleep(2)
print('Timeout: Could not connect to the database within 90 seconds.')
sys.exit(1)
"

# 2. Run Alembic migrations
echo "Running database migrations..."
alembic upgrade head

# 3. Check database and ingest data if empty and JSON files exist
echo "Checking database initialization state..."
python -c "
import os
from src.database import SessionLocal, Chunk
db = SessionLocal()
try:
    count = db.query(Chunk).count()
    print(f'Current chunk count in database: {count}')
    if count == 0:
        # Check if source data files exist in the container
        videos_exist = os.path.exists('videos_data.json')
        chunks_exist = os.path.exists('chunks_with_summaries.json') or os.path.exists('chunks_with_embeddings.json')
        
        if videos_exist and chunks_exist:
            print('Database is empty. Initiating automatic data ingestion...')
            from src.load_data import load_data
            load_data()
            print('Automatic data ingestion completed successfully.')
        else:
            print('Database is empty, but required JSON files (videos_data.json and chunks_with_summaries.json) are not mounted. Skipping ingestion.')
    else:
        print('Database already populated. Skipping ingestion.')
except Exception as e:
    print(f'Error during database initialization check: {e}')
finally:
    db.close()
"

# 4. Start the Telegram Bot
echo "Starting Telegram Bot..."
exec python -m src.bot
