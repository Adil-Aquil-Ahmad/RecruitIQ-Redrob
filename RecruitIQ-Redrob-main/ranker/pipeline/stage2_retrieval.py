"""
Stage 2: BM25 + Embedding hybrid retrieval via Reciprocal Rank Fusion.
12K → 1K candidates.

BM25 over career descriptions captures lexical JD fit.
MiniLM embeddings over headline+summary capture semantic fit (hidden gems).
RRF k=60 is standard; robust to different score scales across retrievers.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from ranker.data.jd_profile import BM25_QUERY_TOKENS, JD_TEXT_KEY_SECTIONS
from ranker.utils.turbovec import TurboVecMatrix

log = logging.getLogger(__name__)

RRF_K = 60


def run(
    candidates: List[Dict[str, Any]],
    bm25_index,
    all_embeddings: np.ndarray | TurboVecMatrix,
    all_candidate_ids: List[str],
    jd_embedding: np.ndarray,
    output_size: int = 1000,
) -> List[Dict[str, Any]]:
    """
    candidates: Stage 1 output (12K)
    bm25_index: precomputed BM25 index over all 100K candidates
    all_embeddings: (100K, 384) precomputed embeddings
    all_candidate_ids: ordered list of all 100K candidate IDs
    jd_embedding: (384,) query vector
    """
    log.info("Stage 2: retrieving from %d candidates", len(candidates))

    # Build lookup from candidate_id → position in all_embeddings
    id_to_global_idx = {cid: i for i, cid in enumerate(all_candidate_ids)}

    # --- BM25 retrieval ---
    query_tokens = BM25_QUERY_TOKENS
    bm25_scores = _bm25_score(candidates, bm25_index, all_candidate_ids, query_tokens)

    # --- Embedding retrieval ---
    embed_scores = _embedding_score(candidates, all_embeddings, id_to_global_idx, jd_embedding)

    # --- RRF fusion ---
    rrf_scores = _reciprocal_rank_fusion(bm25_scores, embed_scores, k=RRF_K)

    # Sort and return top output_size
    sorted_indices = np.argsort(rrf_scores)[::-1]
    result = [candidates[i] for i in sorted_indices[:output_size]]

    log.info("Stage 2 output: %d candidates", len(result))
    return result


def _bm25_score(
    candidates: List[Dict],
    bm25_index,
    all_candidate_ids: List[str],
    query_tokens: List[str],
) -> np.ndarray:
    """Get BM25 scores for each Stage 1 candidate from the precomputed index."""
    # BM25 index was built over all 100K; we need scores for Stage 1 subset
    id_to_bm25_rank = {}

    # Get scores for all candidates in the index
    try:
        all_scores = bm25_index.get_scores(query_tokens)
        id_to_score = dict(zip(all_candidate_ids, all_scores))
    except Exception:
        # Fallback: zero scores
        return np.zeros(len(candidates))

    scores = np.array([id_to_score.get(c["candidate_id"], 0.0) for c in candidates])
    return scores


def _embedding_score(
    candidates: List[Dict],
    all_embeddings: np.ndarray | TurboVecMatrix,
    id_to_global_idx: Dict[str, int],
    jd_embedding: np.ndarray,
) -> np.ndarray:
    """Cosine similarity of each candidate's embedding with the JD embedding."""
    jd_norm = jd_embedding / (np.linalg.norm(jd_embedding) + 1e-8)

    indices = np.array([id_to_global_idx.get(c["candidate_id"], -1) for c in candidates], dtype=np.int64)
    valid_positions = np.flatnonzero(indices >= 0)
    valid_global_indices = indices[valid_positions]

    scores = np.zeros(len(candidates))
    if valid_positions.size == 0:
        return scores

    if isinstance(all_embeddings, TurboVecMatrix):
        vecs = all_embeddings.dequantize(valid_global_indices)
        sims = vecs @ jd_norm
    else:
        vecs = all_embeddings[valid_global_indices]  # (n_valid, 384)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        vecs_norm = vecs / norms
        sims = vecs_norm @ jd_norm  # (n_valid,)

    scores[valid_positions] = sims

    return scores


def _reciprocal_rank_fusion(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    k: int = 60,
) -> np.ndarray:
    """Standard RRF: score = 1/(k + rank_a) + 1/(k + rank_b)."""
    n = len(scores_a)
    ranks_a = _scores_to_ranks(scores_a)
    ranks_b = _scores_to_ranks(scores_b)
    return 1.0 / (k + ranks_a) + 1.0 / (k + ranks_b)


def _scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """Convert scores to 1-based ranks (highest score = rank 1)."""
    order = np.argsort(scores)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.astype(float)
