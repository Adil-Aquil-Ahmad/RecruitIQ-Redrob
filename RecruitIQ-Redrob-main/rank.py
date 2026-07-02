#!/usr/bin/env python3
"""
Main ranking entry point. Produces submission CSV from candidates.jsonl.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Requires precompute.py to have been run first:
    python precompute.py --candidates ./candidates.jsonl

Ranking step runs in < 60 seconds on CPU. No network access required.
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from ranker.pipeline import stage1_title_filter, stage2_retrieval, stage3_rerank
from ranker.utils.cache import load_pickle, load_numpy, load_json, cache_exists
from ranker.utils.turbovec import load_turbovec_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

CACHE_DIR = "./cache"
STAGE1_OUTPUT = 12000
STAGE2_OUTPUT = 1000
TOP_N = 100


def load_candidates(path: str):
    candidates = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def ensure_precomputed(cache_dir: str, candidates_path: str):
    """Run precomputation inline if cache is missing."""
    bm25_path = f"{cache_dir}/bm25_index.pkl"
    embed_path = f"{cache_dir}/candidate_embeddings.turbovec.npz"
    ids_path = f"{cache_dir}/candidate_ids.json"
    jd_path = f"{cache_dir}/jd_embedding.npy"

    if not cache_exists(bm25_path, embed_path, ids_path, jd_path):
        log.info("Cache missing — running precomputation inline...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "precompute.py",
             "--candidates", candidates_path,
             "--cache-dir", cache_dir],
            check=True,
        )
        log.info("Precomputation complete.")


def write_csv(ranked: list, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Round scores, then re-sort: score desc, candidate_id asc on ties (spec requirement)
    rows = []
    for entry in ranked:
        rows.append({
            "cid": entry["candidate"]["candidate_id"],
            "score": round(entry["score"], 6),
            "reasoning": entry.get("reasoning", ""),
        })
    rows.sort(key=lambda r: (-r["score"], r["cid"]))

    # Enforce non-increasing scores after tie-sort
    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i - 1]["score"]:
            rows[i]["score"] = rows[i - 1]["score"]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, row in enumerate(rows, start=1):
            writer.writerow([row["cid"], rank, row["score"], row["reasoning"]])

    log.info("Written %d rows to %s", len(ranked), out_path)


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob AI challenge.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--stage1-size", type=int, default=STAGE1_OUTPUT)
    parser.add_argument("--stage2-size", type=int, default=STAGE2_OUTPUT)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()

    t_start = time.time()

    # Ensure precomputed artifacts exist
    ensure_precomputed(args.cache_dir, args.candidates)

    # Load precomputed artifacts
    log.info("Loading precomputed cache from %s...", args.cache_dir)
    bm25_index = load_pickle(f"{args.cache_dir}/bm25_index.pkl")
    all_embeddings = load_turbovec_embeddings(f"{args.cache_dir}/candidate_embeddings.turbovec.npz")
    all_candidate_ids = load_json(f"{args.cache_dir}/candidate_ids.json")
    jd_embedding = load_numpy(f"{args.cache_dir}/jd_embedding.npy")
    log.info("Cache loaded: embeddings %s", all_embeddings.shape)

    # Load candidates
    log.info("Loading candidates from %s...", args.candidates)
    candidates = load_candidates(args.candidates)
    log.info("Loaded %d candidates", len(candidates))

    # Stage 1: title + industry + YoE filter (100K → 12K)
    t1 = time.time()
    stage1_candidates = stage1_title_filter.run(candidates, output_size=args.stage1_size)
    log.info("Stage 1 done in %.1fs: %d candidates", time.time() - t1, len(stage1_candidates))

    # Stage 2: BM25 + embedding RRF (12K → 1K)
    t2 = time.time()
    stage2_candidates = stage2_retrieval.run(
        stage1_candidates,
        bm25_index=bm25_index,
        all_embeddings=all_embeddings,
        all_candidate_ids=all_candidate_ids,
        jd_embedding=jd_embedding,
        output_size=args.stage2_size,
    )
    log.info("Stage 2 done in %.1fs: %d candidates", time.time() - t2, len(stage2_candidates))

    # Stage 3: full feature scoring (1K → 100)
    t3 = time.time()
    ranked = stage3_rerank.run(stage2_candidates, top_n=args.top_n)
    log.info("Stage 3 done in %.1fs: %d candidates", time.time() - t3, len(ranked))

    # Write output
    write_csv(ranked, args.out)

    elapsed = time.time() - t_start
    log.info("Total ranking time: %.1fs", elapsed)
    if elapsed > 300:
        log.warning("Ranking exceeded 5-minute budget (%.1fs). Review performance.", elapsed)

    # Print top-10 summary
    print("\n=== TOP 10 ===")
    for entry in ranked[:10]:
        c = entry["candidate"]
        print(f"  #{entry['rank']:2d} {c['candidate_id']}  score={entry['score']:.4f}  "
              f"{c['profile']['current_title']} | {c['profile']['years_of_experience']}yrs")


if __name__ == "__main__":
    main()
