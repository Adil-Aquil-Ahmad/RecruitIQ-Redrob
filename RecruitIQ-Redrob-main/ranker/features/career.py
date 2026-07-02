"""
Career quality features: ML-native months, trajectory, industry, company quality.

Key data insights:
- is_ml_title: r=+0.50 with proxy label (strongest non-platform signal)
- is_good_industry: r=+0.32
- is_bad_industry: r=-0.14
- Consulting penalty DROPPED (r=+0.02, noise — captured by industry/text signals)
- YoE r=-0.12 (slight neg) — use bell curve, not linear
"""

import math
from datetime import datetime
from typing import Any, Dict, List

from ranker.data.title_taxonomy import classify_title, ML_CORE_TITLES, TECH_ADJACENT_TITLES
from ranker.data.company_signals import get_industry_score, ML_ADJACENT_INDUSTRIES
from ranker.utils.text import normalize

_TODAY = datetime(2026, 6, 26)

_ML_TITLE_KEYWORDS = {
    "ml", "machine learning", "ai ", " ai", "nlp", "deep learning",
    "data scien", "search engin", "recomm", "applied sci", "research engineer",
    "llm", "retrieval",
}

_SCALE_KEYWORDS = {
    "million", "billion", "production", "deployed", "serving", "100k", "1m ",
    "10m ", "real-time", "real time", "latency", "throughput", "scale",
    "100,000", "1,000,000", "users", "queries", "requests per second",
}


def extract(candidate: Dict[str, Any]) -> Dict[str, float]:
    profile = candidate["profile"]
    career = candidate.get("career_history", [])

    title = profile.get("current_title", "")
    title_category = classify_title(title)
    yoe = float(profile.get("years_of_experience", 0))

    # --- ML title score ---
    is_ml_title = 1.0 if title_category == "ml_core" else 0.0
    is_tech_adjacent = 1.0 if title_category == "tech_adjacent" else 0.0
    is_wrong_title = 1.0 if title_category == "wrong" else 0.0

    # --- ML-native career months (across entire history) ---
    ml_months = _count_ml_career_months(career)

    # --- Industry scoring ---
    current_industry = profile.get("current_industry", "")
    industry_score = get_industry_score(current_industry)

    # Industry trajectory: is the candidate moving toward better industries?
    industry_trajectory = _compute_industry_trajectory(career)

    # --- Production scale evidence ---
    scale_evidence = _count_scale_mentions(career)

    # --- Career stability / avg tenure ---
    avg_tenure = _compute_avg_tenure(career)
    # JD warns against title-chasers (<18 months avg)
    tenure_score = _tenure_score(avg_tenure)

    # --- Company size startup affinity (JD is Series A) ---
    startup_affinity = _startup_affinity(career)

    # --- Recency-weighted career relevance ---
    recency_career_score = _recency_weighted_ml_months(career)

    # --- YoE bell curve ---
    yoe_score = _yoe_bell_curve(yoe)

    # --- Career description specificity (length proxy) ---
    total_words = sum(len(j.get("description", "").split()) for j in career)
    desc_specificity = min(math.log1p(total_words) / math.log1p(500), 1.0)

    return {
        "is_ml_title": is_ml_title,
        "is_tech_adjacent": is_tech_adjacent,
        "is_wrong_title": is_wrong_title,
        "ml_months": ml_months,
        "ml_months_normalized": min(ml_months / 60.0, 1.0),
        "industry_score": industry_score,
        "industry_trajectory": industry_trajectory,
        "scale_evidence": min(scale_evidence / 3.0, 1.0),
        "avg_tenure_months": avg_tenure,
        "tenure_score": tenure_score,
        "startup_affinity": startup_affinity,
        "recency_career_score": recency_career_score,
        "yoe_score": yoe_score,
        "yoe": yoe,
        "desc_specificity": desc_specificity,
    }


def _is_ml_title(title: str) -> bool:
    t = title.lower()
    if t in ML_CORE_TITLES:
        return True
    return any(kw in t for kw in _ML_TITLE_KEYWORDS)


def _count_ml_career_months(career: List[Dict]) -> float:
    total = 0
    for job in career:
        if _is_ml_title(job.get("title", "")):
            total += job.get("duration_months", 0)
    return float(total)


def _count_scale_mentions(career: List[Dict]) -> int:
    count = 0
    for job in career:
        desc = normalize(job.get("description", ""))
        count += sum(1 for kw in _SCALE_KEYWORDS if kw in desc)
    return min(count, 10)


def _compute_avg_tenure(career: List[Dict]) -> float:
    durations = [j.get("duration_months", 0) for j in career if j.get("duration_months", 0) > 0]
    if not durations:
        return 24.0
    return float(sum(durations) / len(durations))


def _tenure_score(avg_months: float) -> float:
    """Title-chaser penalty below 18 months; ideal around 24-36 months."""
    if avg_months < 6:
        return 0.1
    if avg_months < 12:
        return 0.4
    if avg_months < 18:
        return 0.7
    if avg_months <= 48:
        return 1.0
    return 0.9  # Very long tenure = slight penalty (may be stale)


def _startup_affinity(career: List[Dict]) -> float:
    """Score based on experience at small/medium companies (startup-like)."""
    size_scores = {
        "1-10": 1.0, "11-50": 1.0, "51-200": 0.9,
        "201-500": 0.8, "501-1000": 0.6,
        "1001-5000": 0.4, "5001-10000": 0.2, "10001+": 0.1,
    }
    if not career:
        return 0.5
    scores = [size_scores.get(j.get("company_size", ""), 0.3) for j in career]
    # Weight recent roles more
    weighted = sum(s * (1.2 ** i) for i, s in enumerate(reversed(scores)))
    normalizer = sum(1.2 ** i for i in range(len(scores)))
    return min(weighted / normalizer, 1.0)


def _compute_industry_trajectory(career: List[Dict]) -> float:
    """Is the candidate's industry improving over time? Moving toward AI/ML = +."""
    if len(career) < 2:
        return 0.5
    # Sort by start date (most recent first already in schema)
    recent_industries = [j.get("industry", "") for j in career[:3]]
    old_industries = [j.get("industry", "") for j in career[-3:]]
    recent_score = sum(1 for ind in recent_industries if ind in ML_ADJACENT_INDUSTRIES) / max(len(recent_industries), 1)
    old_score = sum(1 for ind in old_industries if ind in ML_ADJACENT_INDUSTRIES) / max(len(old_industries), 1)
    trajectory = 0.5 + 0.5 * (recent_score - old_score)
    return max(0.0, min(1.0, trajectory))


def _recency_weighted_ml_months(career: List[Dict]) -> float:
    """ML career months weighted by recency — recent ML experience > old."""
    total = 0.0
    for i, job in enumerate(career):
        if _is_ml_title(job.get("title", "")):
            recency_weight = 1.0 / (1.0 + i * 0.3)
            total += job.get("duration_months", 0) * recency_weight
    return min(total / 60.0, 1.0)


def _yoe_bell_curve(yoe: float) -> float:
    """Bell curve centered at 6.5 years. Soft penalty above 12, hard below 1."""
    if yoe < 1.0:
        return 0.05
    if yoe < 3.0:
        return 0.3 + 0.2 * (yoe - 1.0)
    if yoe <= 9.0:
        # Peak at 6.5, gentle curve
        return 1.0 - 0.02 * (yoe - 6.5) ** 2
    if yoe <= 12.0:
        return 0.85 - 0.05 * (yoe - 9.0)
    return max(0.5, 0.70 - 0.03 * (yoe - 12.0))
