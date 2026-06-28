#!/usr/bin/env python3
"""
Offline precomputation: BM25 index + candidate embeddings.
Run once before rank.py. Output goes to ./cache/.

Runtime on M2/M3 CPU: ~3-5 minutes for 100K candidates.
This does NOT count toward the 5-minute ranking budget.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from ranker.data.jd_profile import BM25_QUERY_TOKENS, JD_TEXT_KEY_SECTIONS
from ranker.utils.text import candidate_bm25_text, candidate_embed_text, tokenize
from ranker.utils.cache import save_pickle, save_numpy, save_json, cache_exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def load_candidates(path: str):
    candidates = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def build_bm25_index(candidates, cache_path: str):
    log.info("Building BM25 index over %d candidates...", len(candidates))
    t0 = time.time()
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        log.error("rank-bm25 not installed. Run: pip install rank-bm25")
        sys.exit(1)

    corpus = [tokenize(candidate_bm25_text(c)) for c in candidates]
    index = BM25Okapi(corpus)
    save_pickle(index, cache_path)
    log.info("BM25 index built in %.1fs", time.time() - t0)
    return index


def build_embeddings(candidates, cache_path: str, ids_path: str, model_name: str, batch_size: int):
    log.info("Computing embeddings for %d candidates (model: %s)...", len(candidates), model_name)
    t0 = time.time()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer(model_name)

    texts = [candidate_embed_text(c) for c in candidates]
    candidate_ids = [c["candidate_id"] for c in candidates]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True,
    )

    save_numpy(embeddings, cache_path)
    save_json(candidate_ids, ids_path)
    log.info("Embeddings computed in %.1fs → shape %s", time.time() - t0, embeddings.shape)
    return embeddings, candidate_ids


def build_jd_embedding(model_name: str, cache_path: str) -> np.ndarray:
    log.info("Computing JD embedding...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit(1)

    model = SentenceTransformer(model_name)
    jd_vec = model.encode([JD_TEXT_KEY_SECTIONS], normalize_embeddings=False)[0]
    save_numpy(jd_vec, cache_path)
    log.info("JD embedding saved: shape %s", jd_vec.shape)
    return jd_vec


def main():
    parser = argparse.ArgumentParser(description="Precompute BM25 index and embeddings.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    bm25_path = str(cache / "bm25_index.pkl")
    embed_path = str(cache / "candidate_embeddings.npy")
    ids_path = str(cache / "candidate_ids.json")
    jd_embed_path = str(cache / "jd_embedding.npy")

    log.info("Loading candidates from %s...", args.candidates)
    candidates = load_candidates(args.candidates)
    log.info("Loaded %d candidates", len(candidates))

    if args.force or not cache_exists(bm25_path):
        build_bm25_index(candidates, bm25_path)
    else:
        log.info("BM25 index already cached, skipping (use --force to rebuild)")

    if args.force or not cache_exists(embed_path, ids_path):
        build_embeddings(candidates, embed_path, ids_path, args.model, args.batch_size)
    else:
        log.info("Embeddings already cached, skipping (use --force to rebuild)")

    if args.force or not cache_exists(jd_embed_path):
        build_jd_embedding(args.model, jd_embed_path)
    else:
        log.info("JD embedding already cached, skipping")

    log.info("Precomputation complete. Cache at: %s", args.cache_dir)


if __name__ == "__main__":
    main()
