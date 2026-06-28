"""
Technical features: JD keyword hits, assessment scores, GitHub activity.

Key data insights:
- jd_career_keyword_hits: r=+0.45 with platform proxy label
- assessment_score: r=+0.27 (verified, cannot be faked)
- github_activity_score: r=+0.17
- Skills are uniformly distributed — raw skill count is noise
- Skill duration + endorsements for JD-specific skills are genuine signals
"""

import math
from typing import Any, Dict

from ranker.data.jd_profile import REQUIRED_SKILL_VOCAB, JD_SKILL_NAMES
from ranker.utils.text import candidate_career_text, normalize


# Weighted JD vocabulary — higher weight for discriminating terms
_CAREER_KW_WEIGHTS: Dict[str, float] = {
    # Core retrieval/ranking (highest weight — most discriminating)
    "embedding": 2.0, "embeddings": 2.0, "vector": 1.5, "dense retrieval": 2.5,
    "semantic search": 2.5, "neural search": 2.0,
    "faiss": 2.5, "pinecone": 2.5, "qdrant": 2.5, "weaviate": 2.5,
    "elasticsearch": 2.0, "opensearch": 2.0, "milvus": 2.5, "chroma": 2.0,
    "approximate nearest neighbor": 2.5, "ann ": 1.5, "hnsw": 2.0,
    "hybrid search": 2.5, "bm25": 2.0,
    # Ranking/recommendation
    "ranking": 2.0, "ranker": 2.0, "recommendation": 2.0,
    "learning to rank": 2.5, "ltr": 2.0, "collaborative filtering": 2.0,
    # NLP/ML
    "nlp": 1.5, "natural language": 1.5, "bert": 1.5, "transformer": 1.5,
    "huggingface": 1.5, "sentence transformer": 2.0,
    "pytorch": 1.0, "tensorflow": 1.0,
    # Fine-tuning / LLMs
    "fine-tuning": 2.0, "fine tuning": 2.0, "finetuning": 2.0,
    "lora": 2.0, "qlora": 2.0, "peft": 2.0, "rag": 2.0,
    "retrieval augmented": 2.0, "llm": 1.5, "large language model": 1.5,
    # Evaluation
    "ndcg": 2.5, "mrr": 2.0, "map": 1.5, "a/b test": 2.0, "ab test": 2.0,
    "online experiment": 2.0, "offline evaluation": 2.0,
    # Production signals
    "production": 1.5, "deployed": 1.5, "serving": 1.5, "inference": 1.5,
    "pipeline": 1.0, "scale": 1.0, "latency": 1.5,
    # General ML (lower weight — more common)
    "machine learning": 1.0, "deep learning": 1.0, "neural network": 1.0,
    "python": 0.8, "scikit": 0.8,
}

_HEADLINE_KW_WEIGHTS: Dict[str, float] = {
    k: v * 1.5 for k, v in _CAREER_KW_WEIGHTS.items()
}

_JD_SKILL_NAMES_LOWER = {s.lower() for s in JD_SKILL_NAMES}

_PROFICIENCY_WEIGHTS = {
    "beginner": 0.25,
    "intermediate": 0.50,
    "advanced": 0.75,
    "expert": 1.00,
}


def extract(candidate: Dict[str, Any]) -> Dict[str, float]:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    skills = candidate.get("skills", [])

    career_text = normalize(candidate_career_text(candidate))
    headline_text = normalize(profile.get("headline", ""))
    summary_text = normalize(profile.get("summary", ""))

    # --- JD keyword hit scores ---
    career_kw_score = _weighted_hits(career_text, _CAREER_KW_WEIGHTS)
    headline_kw_score = _weighted_hits(headline_text, _HEADLINE_KW_WEIGHTS)
    summary_kw_score = _weighted_hits(summary_text, _CAREER_KW_WEIGHTS)

    # Combined career text score (career is richest)
    career_text_score = career_kw_score + 0.4 * headline_kw_score + 0.2 * summary_kw_score

    # --- Assessment scores (bonus only — 76% have none) ---
    assessment_scores = signals.get("skill_assessment_scores", {})
    jd_assess_scores = []
    for skill_name, score in assessment_scores.items():
        if skill_name.lower() in _JD_SKILL_NAMES_LOWER:
            jd_assess_scores.append(score)
    assessment_composite = (sum(jd_assess_scores) / len(jd_assess_scores) / 100.0
                            if jd_assess_scores else 0.0)
    # Coverage bonus: more JD-relevant assessments = higher credibility
    assessment_coverage = min(len(jd_assess_scores) / 3.0, 1.0)
    assessment_score = assessment_composite * (0.7 + 0.3 * assessment_coverage)

    # --- GitHub activity ---
    github_raw = signals.get("github_activity_score", -1)
    github_score = max(github_raw, 0) / 100.0

    # --- Skill duration + endorsements for JD skills ---
    jd_skill_duration = 0.0
    jd_skill_endorsements = 0.0
    jd_skill_count = 0
    for sk in skills:
        if sk["name"].lower() in _JD_SKILL_NAMES_LOWER:
            prof_weight = _PROFICIENCY_WEIGHTS.get(sk.get("proficiency", "intermediate"), 0.5)
            duration = sk.get("duration_months", 0)
            endorsements = sk.get("endorsements", 0)
            jd_skill_duration += math.log1p(duration) * prof_weight
            jd_skill_endorsements += math.log1p(endorsements)
            jd_skill_count += 1

    return {
        "career_text_score": career_text_score,
        "headline_kw_score": headline_kw_score,
        "assessment_score": assessment_score,
        "github_score": github_score,
        "jd_skill_duration": jd_skill_duration,
        "jd_skill_endorsements": jd_skill_endorsements,
        "jd_skill_count": jd_skill_count,
        "has_assessment": float(len(jd_assess_scores) > 0),
    }


def _weighted_hits(text: str, weights: Dict[str, float]) -> float:
    return sum(w for kw, w in weights.items() if kw in text)
