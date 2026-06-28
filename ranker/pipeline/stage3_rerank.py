"""
Stage 3: Full feature extraction and weighted scoring.
1K → 100 candidates using all feature groups.
Includes top-10 sanity pass to maximize NDCG@10.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from ranker.features import technical, career, behavioral, credibility, education, logistics
from ranker.scoring.ensemble import compute_scores, merge_feature_dicts
from ranker.scoring.reasoning import generate as generate_reasoning

log = logging.getLogger(__name__)


def run(
    candidates: List[Dict[str, Any]],
    top_n: int = 100,
) -> List[Dict[str, Any]]:
    """Returns top_n candidates with scores and reasoning, sorted by score descending."""
    log.info("Stage 3: full feature scoring on %d candidates", len(candidates))

    feature_dicts = []
    for c in candidates:
        tech = technical.extract(c)
        car = career.extract(c)
        beh = behavioral.extract(c)
        cred = credibility.extract(c)
        edu = education.extract(c)
        logi = logistics.extract(c)
        merged = merge_feature_dicts(tech, car, beh, cred, edu, logi)
        feature_dicts.append(merged)

    scores = compute_scores(candidates, feature_dicts)

    # Sort by score descending; tie-break by candidate_id ascending (spec requirement)
    ids = [c["candidate_id"] for c in candidates]
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], ids[i]))
    ranked = []
    for i, idx in enumerate(order):
        ranked.append({
            "candidate": candidates[idx],
            "features": feature_dicts[idx],
            "score": float(scores[idx]),
            "rank": i + 1,
        })

    # Top-10 sanity pass: ensure no clearly wrong candidates in top 10
    ranked = _top10_sanity_pass(ranked)

    # Add reasoning strings to top 100
    for entry in ranked[:top_n]:
        entry["reasoning"] = generate_reasoning(
            entry["candidate"], entry["features"], entry["score"]
        )

    log.info("Stage 3 complete. Top score: %.4f, rank-100 score: %.4f",
             ranked[0]["score"], ranked[min(top_n - 1, len(ranked) - 1)]["score"])

    return ranked[:top_n]


def _top10_sanity_pass(ranked: List[Dict]) -> List[Dict]:
    """
    Swap out any top-10 candidate that is clearly wrong:
    - wrong title + no ML career months + honeypot signals
    Replace with highest-scoring candidate from positions 11-50 that passes.
    """
    def _is_clearly_wrong(entry: Dict) -> bool:
        f = entry["features"]
        is_wrong_title = f.get("is_wrong_title", 0.0) > 0.5
        ml_months = f.get("ml_months", 0.0)
        career_text = f.get("career_text_score", 0.0)
        honeypot = f.get("is_honeypot", 0.0) > 0.5
        # Wrong if: wrong title AND no ML career evidence AND no JD text hits
        return (is_wrong_title and ml_months < 6 and career_text < 2.0) or honeypot

    def _is_acceptable(entry: Dict) -> bool:
        f = entry["features"]
        has_ml_title = f.get("is_ml_title", 0.0) > 0.5 or f.get("is_tech_adjacent", 0.0) > 0.5
        has_ml_career = f.get("ml_months", 0.0) >= 12 or f.get("career_text_score", 0.0) >= 2.0
        not_honeypot = f.get("is_honeypot", 0.0) < 0.5
        return (has_ml_title or has_ml_career) and not_honeypot

    # Find swap candidates from positions 10-49
    swap_pool = [e for e in ranked[10:50] if _is_acceptable(e)]
    swap_idx = 0

    for i in range(min(10, len(ranked))):
        if _is_clearly_wrong(ranked[i]) and swap_idx < len(swap_pool):
            log.warning(
                "Top-10 sanity swap: rank %d (%s) → replaced with rank %d (%s)",
                i + 1,
                ranked[i]["candidate"]["candidate_id"],
                ranked[swap_pool[swap_idx]["rank"]]["rank"] if "rank" in ranked[swap_pool[swap_idx]["rank"]] else "?",
                swap_pool[swap_idx]["candidate"]["candidate_id"],
            )
            # Swap entries
            swap_entry = swap_pool[swap_idx]
            original_pos_of_swap = ranked.index(swap_entry)
            ranked[i], ranked[original_pos_of_swap] = swap_entry, ranked[i]
            swap_idx += 1

    return ranked
