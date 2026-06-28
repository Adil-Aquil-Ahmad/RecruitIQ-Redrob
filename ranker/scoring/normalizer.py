"""
Percentile normalization for feature scores.

Normalize within the Stage 3 candidate pool (not global 100K).
This maximizes resolution in the ranking range and is robust to outliers.
"""

import numpy as np
from typing import Dict, List, Any


def percentile_normalize(
    records: List[Dict[str, Any]],
    feature_key: str,
) -> np.ndarray:
    """Map raw feature values to [0, 1] via percentile rank."""
    values = np.array([r[feature_key] for r in records], dtype=float)
    n = len(values)
    if n == 0:
        return values
    ranks = np.argsort(np.argsort(values))
    return ranks / max(n - 1, 1)


def normalize_feature_matrix(
    records: List[Dict[str, Any]],
    feature_keys: List[str],
) -> Dict[str, np.ndarray]:
    """Percentile-normalize a set of features across the candidate pool."""
    return {key: percentile_normalize(records, key) for key in feature_keys}


def clip_score(score: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, score))
