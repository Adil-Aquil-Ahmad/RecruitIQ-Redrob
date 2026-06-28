"""
Final score computation from normalized feature components.

Data-driven weights (validated against platform proxy label correlations):
  platform_composite: 0.28  (r=0.71/0.69, strongest features)
  career_text:        0.20  (r=0.45, JD keyword hits)
  title_career:       0.18  (r=0.50, ML title + ML months)
  assessment:         0.10  (r=0.27, verified signal)
  industry:           0.08  (r=0.32/0.14)
  github:             0.06  (r=0.17)
  engagement:         0.05  (r=0.09/0.08)
  education:          0.03
  location:           0.02

Dropped: consulting penalty (r=0.02), notice period (r=-0.02)
"""

import numpy as np
from typing import Dict, List, Any, Tuple

from ranker.scoring.normalizer import normalize_feature_matrix, clip_score

WEIGHTS = {
    "platform_composite": 0.28,
    "career_text": 0.20,
    "title_career": 0.18,
    "assessment": 0.10,
    "industry": 0.08,
    "github": 0.06,
    "engagement": 0.05,
    "education": 0.03,
    "location": 0.02,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

# Features to normalize (keys in the combined feature dict)
_FEATURES_TO_NORMALIZE = [
    "platform_composite",
    "career_text_score",
    "title_career_composite",
    "assessment_score",
    "industry_score",
    "github_score",
    "engagement_composite",
    "education_score",
    "logistics_score",
]

# Wrong-title platform cap: prevent high platform signals from rescuing wrong-title candidates
_WRONG_TITLE_PLATFORM_CAP = 0.50


def compute_scores(
    candidates: List[Dict[str, Any]],
    feature_dicts: List[Dict[str, float]],
) -> np.ndarray:
    """
    Given 1K candidates and their feature dicts, return final scores [0, 1].
    Applies percentile normalization within the pool.
    """
    n = len(candidates)
    assert len(feature_dicts) == n

    # Normalize each feature within the pool
    norm = normalize_feature_matrix(feature_dicts, _FEATURES_TO_NORMALIZE)

    scores = np.zeros(n)
    for i in range(n):
        f = feature_dicts[i]
        is_wrong = f.get("is_wrong_title", 0.0)

        # Platform composite (with wrong-title cap)
        platform_norm = norm["platform_composite"][i]
        if is_wrong > 0.5:
            platform_norm = min(platform_norm, _WRONG_TITLE_PLATFORM_CAP)

        s = (
            WEIGHTS["platform_composite"] * platform_norm
            + WEIGHTS["career_text"] * norm["career_text_score"][i]
            + WEIGHTS["title_career"] * norm["title_career_composite"][i]
            + WEIGHTS["assessment"] * norm["assessment_score"][i]
            + WEIGHTS["industry"] * norm["industry_score"][i]
            + WEIGHTS["github"] * norm["github_score"][i]
            + WEIGHTS["engagement"] * norm["engagement_composite"][i]
            + WEIGHTS["education"] * norm["education_score"][i]
            + WEIGHTS["location"] * norm["logistics_score"][i]
        )

        # Apply honeypot hard penalty (subtract from final score)
        honeypot_penalty = f.get("honeypot_penalty", 0.0)
        s = clip_score(s - honeypot_penalty)

        scores[i] = s

    return scores


def build_title_career_composite(feat: Dict[str, float]) -> float:
    """Combine ML title and ML career months into a single composite."""
    is_ml = feat.get("is_ml_title", 0.0)
    is_adj = feat.get("is_tech_adjacent", 0.0)
    ml_months_norm = feat.get("ml_months_normalized", 0.0)
    recency = feat.get("recency_career_score", 0.0)
    yoe_score = feat.get("yoe_score", 0.5)

    # ML title is the strongest signal, ML career months adds depth
    title_component = 1.0 * is_ml + 0.4 * is_adj
    career_component = 0.6 * ml_months_norm + 0.4 * recency

    composite = (
        0.45 * min(title_component, 1.0)
        + 0.35 * career_component
        + 0.20 * yoe_score
    )
    return clip_score(composite)


def merge_feature_dicts(
    tech: Dict, career: Dict, behavioral: Dict,
    credibility: Dict, education: Dict, logistics: Dict,
) -> Dict[str, float]:
    """Merge all feature group dicts into a single flat dict."""
    merged = {}
    merged.update(tech)
    merged.update(career)
    merged.update(behavioral)
    merged.update(credibility)
    merged.update(education)
    merged.update(logistics)

    # Build derived composites needed by compute_scores
    merged["title_career_composite"] = build_title_career_composite(merged)

    return merged
