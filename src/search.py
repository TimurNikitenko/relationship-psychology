import re
import os
import torch
import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import joinedload
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

from src.database import SessionLocal, Chunk, Video
from src.logger import logger

# Initialize device and model for semantic search
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "cointegrated/rubert-tiny2"

logger.info(f"Loading embedding model '{MODEL_NAME}' on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
logger.info("Embedding model loaded successfully.")

def get_query_embedding(text: str) -> list:
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_emb = outputs.last_hidden_state[0, 0].cpu().numpy()
    norm = np.linalg.norm(cls_emb)
    if norm > 1e-9:
        cls_emb = cls_emb / norm
    return cls_emb.tolist()

# Tokenizer for BM25
def tokenize_text(text: str) -> list[str]:
    # Extract Russian and English alphanumeric words
    return re.findall(r'[а-яёa-z0-9]+', text.lower())

class SearchEngine:
    def __init__(self):
        self.db = SessionLocal()
        self.chunks_cache = []
        self.bm25 = None
        self.chunk_id_map = {}  # index in corpus -> chunk object
        self.load_bm25_corpus()
        
        # Load optional Re-ranker
        self.use_reranker = os.getenv("USE_RERANKER", "False").lower() in ("true", "1", "yes")
        self.reranker = None
        if self.use_reranker:
            reranker_model = os.getenv("RERANKER_MODEL_NAME", "DiS-Lab/ruReranker-base")
            logger.info(f"Loading re-ranker model '{reranker_model}' on {device}...")
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(reranker_model, device=device)
                logger.info("Re-ranker loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load re-ranker model: {e}. Re-ranking will be disabled.", exc_info=True)
                self.use_reranker = False

    def load_bm25_corpus(self):
        logger.info("Caching chunks and fitting BM25 index...")
        try:
            # Load all chunks with their associated video metadata
            self.chunks_cache = self.db.query(Chunk).options(joinedload(Chunk.video)).all()
            tokenized_corpus = []
            for idx, chunk in enumerate(self.chunks_cache):
                tokenized_corpus.append(tokenize_text(chunk.text))
                self.chunk_id_map[idx] = chunk
            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)
                logger.info(f"BM25 index ready with {len(self.chunks_cache)} chunks.")
            else:
                logger.warning("BM25 corpus is empty! Database might not be initialized yet.")
        except Exception as e:
            logger.error(f"Error building BM25 corpus: {e}", exc_info=True)
            self.db.rollback()

    def fallback_search(self, query: str, limit: int = 5) -> list:
        """Fallback search over summary and key_points columns using OR matching of query terms."""
        words = tokenize_text(query)
        if not words:
            logger.warning(f"Fallback search skipped: query '{query}' tokenizes to empty list.")
            return []
        
        logger.info(f"Executing database fallback search for terms: {words}")
        try:
            filters = []
            for word in words:
                filters.append(Chunk.summary.ilike(f"%{word}%"))
                filters.append(Chunk.key_points.cast(sa.Text).ilike(f"%{word}%"))
            
            results = (
                self.db.query(Chunk)
                .options(joinedload(Chunk.video))
                .filter(sa.or_(*filters))
                .limit(limit)
                .all()
            )
            logger.info(f"Fallback search query returned {len(results)} rows.")
            return [(chunk, 1.0) for chunk in results]
        except Exception as e:
            logger.error(f"Fallback search failed: {e}", exc_info=True)
            self.db.rollback()
            return []

    def strict_search(self, pattern: str, limit: int = 5) -> list:
        logger.info(f"Executing strict search for pattern: '{pattern}'")
        
        # 1. PostgreSQL case-insensitive regex search (~*) on full text
        try:
            results = (
                self.db.query(Chunk)
                .options(joinedload(Chunk.video))
                .filter(Chunk.text.op("~*")(pattern))
                .limit(limit)
                .all()
            )
            if results:
                logger.info(f"Strict search [Stage 1: Regex Full-Text] found {len(results)} matches.")
                return [(chunk, 1.0) for chunk in results]
        except Exception as e:
            logger.error(f"Regex full-text search failed with pattern '{pattern}': {e}. Falling back...")
            self.db.rollback()

        # 2. Fallback to simple substring search on full text
        try:
            results = (
                self.db.query(Chunk)
                .options(joinedload(Chunk.video))
                .filter(Chunk.text.ilike(f"%{pattern}%"))
                .limit(limit)
                .all()
            )
            if results:
                logger.info(f"Strict search [Stage 2: Substring Full-Text] found {len(results)} matches.")
                return [(chunk, 1.0) for chunk in results]
        except Exception as e:
            logger.error(f"Substring full-text search failed with pattern '{pattern}': {e}. Falling back...")
            self.db.rollback()

        # 3. Fallback to regex search on summary/key_points
        try:
            results = (
                self.db.query(Chunk)
                .options(joinedload(Chunk.video))
                .filter(
                    (Chunk.summary.op("~*")(pattern)) |
                    (Chunk.key_points.cast(sa.Text).op("~*")(pattern))
                )
                .limit(limit)
                .all()
            )
            if results:
                logger.info(f"Strict search [Stage 3: Regex Summary/KeyPoints] found {len(results)} matches.")
                return [(chunk, 1.0) for chunk in results]
        except Exception as e:
            logger.error(f"Regex summary/keypoints search failed: {e}. Falling back...")
            self.db.rollback()

        # 4. Fallback to substring search on summary/key_points
        try:
            results = (
                self.db.query(Chunk)
                .options(joinedload(Chunk.video))
                .filter(
                    (Chunk.summary.ilike(f"%{pattern}%")) |
                    (Chunk.key_points.cast(sa.Text).ilike(f"%{pattern}%"))
                )
                .limit(limit)
                .all()
            )
            if results:
                logger.info(f"Strict search [Stage 4: Substring Summary/KeyPoints] found {len(results)} matches.")
                return [(chunk, 1.0) for chunk in results]
        except Exception as e:
            logger.error(f"Substring summary/keypoints search failed: {e}")
            self.db.rollback()

        logger.info("Strict search returned 0 matches.")
        return []

    def semantic_search(self, query: str, limit: int = 5) -> list:
        logger.info(f"Executing semantic search for query: '{query}'")
        try:
            query_emb = get_query_embedding(query)
            distance_col = Chunk.embedding.cosine_distance(query_emb)
            results = (
                self.db.query(Chunk, distance_col)
                .options(joinedload(Chunk.video))
                .order_by(distance_col.asc())
                .limit(limit)
                .all()
            )
            # Convert cosine distance to cosine similarity
            parsed_results = [(chunk, 1.0 - float(dist)) for chunk, dist in results]
            logger.info(f"Semantic database query returned {len(parsed_results)} rows.")
            
            # If no results or very low similarity, try fallback
            if not parsed_results:
                logger.info("Semantic search yielded 0 results. Triggering database fallback...")
                return self.fallback_search(query, limit)
                
            return parsed_results
        except Exception as e:
            logger.error(f"Semantic search failed: {e}", exc_info=True)
            self.db.rollback()
            return self.fallback_search(query, limit)

    def combined_search(self, query: str, limit: int = 5, k_rrf: int = 60) -> list:
        logger.info(f"Executing combined search for query: '{query}'")
        
        # 1. Get Semantic results (top 50)
        semantic_results = []
        try:
            query_emb = get_query_embedding(query)
            distance_col = Chunk.embedding.cosine_distance(query_emb)
            semantic_raw = (
                self.db.query(Chunk, distance_col)
                .options(joinedload(Chunk.video))
                .order_by(distance_col.asc())
                .limit(50)
                .all()
            )
            semantic_results = [chunk for chunk, _ in semantic_raw]
            logger.debug(f"Combined Search - Semantic step found {len(semantic_results)} raw candidates.")
        except Exception as e:
            logger.error(f"Combined Search - Semantic step failed: {e}", exc_info=True)
            self.db.rollback()

        # 2. Get BM25 results (top 50)
        bm25_results = []
        if self.bm25:
            try:
                query_tokens = tokenize_text(query)
                bm25_scores = self.bm25.get_scores(query_tokens)
                top_indices = np.argsort(bm25_scores)[::-1][:50]
                for idx in top_indices:
                    if bm25_scores[idx] > 0:  # only keep matching documents
                        bm25_results.append(self.chunk_id_map[idx])
                logger.debug(f"Combined Search - BM25 step found {len(bm25_results)} raw candidates.")
            except Exception as e:
                logger.error(f"Combined Search - BM25 step failed: {e}", exc_info=True)
        else:
            logger.warning("BM25 index not fit. Skipping BM25 step in combined search.")

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}      # chunk_id -> rrf_score
        chunk_lookup = {}    # chunk_id -> chunk object

        # Score BM25 results
        for rank, chunk in enumerate(bm25_results):
            chunk_lookup[chunk.id] = chunk
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (k_rrf + (rank + 1)))

        # Score Semantic results
        for rank, chunk in enumerate(semantic_results):
            chunk_lookup[chunk.id] = chunk
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (k_rrf + (rank + 1)))

        # If no RRF matches, fallback
        if not rrf_scores:
            logger.info("Combined Search - No candidates found in RRF fusion. Triggering database fallback...")
            return self.fallback_search(query, limit)

        # Sort by RRF score descending
        # If re-ranker is enabled, retrieve top 25 candidates to rerank
        rerank_limit = 25 if self.use_reranker else limit
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:rerank_limit]
        
        candidates = [chunk_lookup[cid] for cid in sorted_chunk_ids]
        logger.info(f"Combined Search - RRF fused {len(rrf_scores)} unique candidates. Selected top {len(candidates)} for scoring.")

        # Apply Cross-Encoder re-ranking if enabled
        if self.use_reranker and self.reranker and candidates:
            try:
                logger.info(f"Re-ranking {len(candidates)} candidates using Cross-Encoder model...")
                pairs = [[query, chunk.text] for chunk in candidates]
                scores = self.reranker.predict(pairs)
                scored_candidates = list(zip(candidates, [float(s) for s in scores]))
                # Sort by Cross-Encoder score descending
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                logger.info(f"Re-ranking completed successfully. Top score: {scored_candidates[0][1]:.4f}")
                return scored_candidates[:limit]
            except Exception as e:
                logger.error(f"Re-ranking execution failed: {e}. Falling back to RRF ordering.", exc_info=True)
                return [(chunk_lookup[cid], rrf_scores[cid]) for cid in sorted_chunk_ids[:limit]]

        # Return list of tuples: (chunk, rrf_score)
        logger.info(f"Combined search returned {len(sorted_chunk_ids[:limit])} items based on RRF ordering.")
        return [(chunk_lookup[cid], rrf_scores[cid]) for cid in sorted_chunk_ids[:limit]]

    def get_random_insight(self) -> tuple[Chunk, str] | None:
        """Fetches a random insight (key point or summary) from the database."""
        logger.info("Fetching a random insight from the database...")
        import random
        try:
            # Query chunks that have key_points and are not empty
            results = (
                self.db.query(Chunk)
                .options(joinedload(Chunk.video))
                .filter(Chunk.key_points.isnot(None))
                .filter(sa.func.array_length(Chunk.key_points, 1) > 0)
                .all()
            )
            if not results:
                logger.warning("No chunks with non-empty key_points found. Trying fallback to summaries...")
                # Fallback to chunks with summaries
                results = (
                    self.db.query(Chunk)
                    .options(joinedload(Chunk.video))
                    .filter(Chunk.summary.isnot(None))
                    .all()
                )
                if not results:
                    logger.warning("No summaries found in the database.")
                    return None
                chunk = random.choice(results)
                logger.info(f"Random insight (summary fallback) selected from chunk ID {chunk.id}.")
                return chunk, chunk.summary
                
            chunk = random.choice(results)
            thesis = random.choice(chunk.key_points)
            logger.info(f"Random insight (key point) selected from chunk ID {chunk.id}.")
            return chunk, thesis
        except Exception as e:
            logger.error(f"Error fetching random insight: {e}", exc_info=True)
            self.db.rollback()
            return None

    def close(self):
        logger.info("Closing database connection in SearchEngine.")
        self.db.close()
