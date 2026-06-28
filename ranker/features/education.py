"""Education features: institution tier and field relevance (3% weight)."""

from typing import Any, Dict

_TIER_SCORES = {
    "tier_1": 1.00,
    "tier_2": 0.75,
    "tier_3": 0.50,
    "tier_4": 0.30,
    "unknown": 0.25,
}

_STEM_FIELDS = {
    "computer science", "computer engineering", "software engineering",
    "information technology", "artificial intelligence", "machine learning",
    "data science", "statistics", "mathematics", "electrical engineering",
    "electronics", "electronics and communication", "information systems",
    "computational science", "applied mathematics", "operations research",
    "physics", "engineering",
}


def extract(candidate: Dict[str, Any]) -> Dict[str, float]:
    education = candidate.get("education", [])

    if not education:
        return {"education_score": 0.25, "best_tier_score": 0.25, "is_stem": 0.5}

    # Use best (highest-tier) institution
    best_tier_score = max(
        _TIER_SCORES.get(edu.get("tier", "unknown"), 0.25)
        for edu in education
    )

    # Field relevance
    is_stem = 0.0
    for edu in education:
        field = edu.get("field_of_study", "").lower()
        if any(s in field for s in _STEM_FIELDS):
            is_stem = 1.0
            break
        if "engineer" in field or "science" in field or "tech" in field:
            is_stem = 0.7

    education_score = 0.70 * best_tier_score + 0.30 * is_stem

    return {
        "education_score": education_score,
        "best_tier_score": best_tier_score,
        "is_stem": is_stem,
    }
