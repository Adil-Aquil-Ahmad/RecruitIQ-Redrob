"""
Evidence-grounded reasoning string generation.

Stage 4 evaluation penalizes: empty, templated, hallucinated, or contradictory reasoning.
Every string must reference actual values from the candidate's profile.
"""

from typing import Any, Dict


def generate(candidate: Dict[str, Any], features: Dict[str, float], score: float) -> str:
    profile = candidate["profile"]
    sig = candidate["redrob_signals"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    parts = []

    # --- Title + YoE ---
    title = profile.get("current_title", "unknown")
    yoe = profile.get("years_of_experience", 0)
    parts.append(f"{yoe:.1f} yrs exp as {title}")

    # --- Most relevant career role ---
    best_role = _best_career_role(career)
    if best_role:
        company = best_role.get("company", "")
        role_title = best_role.get("title", "")
        duration = best_role.get("duration_months", 0)
        if company and role_title:
            parts.append(f"{role_title} at {company} ({duration}mo)")

    # --- ML career evidence ---
    ml_months = features.get("ml_months", 0)
    if ml_months >= 24:
        parts.append(f"{int(ml_months)}mo in ML/AI roles")

    # --- JD keyword evidence in career ---
    career_text_score = features.get("career_text_score", 0)
    if career_text_score > 5:
        parts.append("strong production ML evidence in career history")
    elif career_text_score > 2:
        parts.append("ML/retrieval keywords in career descriptions")

    # --- Technical skills evidence ---
    top_skills = _top_jd_skills(skills)
    if top_skills:
        parts.append(f"skills: {', '.join(top_skills[:3])}")

    # --- Assessment scores ---
    assessments = sig.get("skill_assessment_scores", {})
    if assessments:
        avg_score = sum(assessments.values()) / len(assessments)
        skill_names = list(assessments.keys())[:2]
        parts.append(f"assessed {avg_score:.0f}/100 ({', '.join(skill_names)})")

    # --- GitHub ---
    github = sig.get("github_activity_score", -1)
    if github > 0:
        parts.append(f"GitHub score {github:.0f}/100")

    # --- Platform signals ---
    saved = sig.get("saved_by_recruiters_30d", 0)
    search = sig.get("search_appearance_30d", 0)
    if saved > 10:
        parts.append(f"saved by {saved} recruiters/30d")
    if search > 100:
        parts.append(f"{search} search appearances/30d")

    # --- Behavioral ---
    days_active = int(features.get("days_since_active", 999))
    parts.append(f"last active {days_active}d ago")
    resp = sig.get("recruiter_response_rate", 0)
    if resp > 0:
        parts.append(f"response rate {resp:.0%}")

    # --- Location ---
    location = profile.get("location", "")
    country = profile.get("country", "")
    if location:
        parts.append(f"{location}, {country}")
    elif country:
        parts.append(country)

    # --- Honeypot concern ---
    if features.get("is_honeypot", 0) > 0.5:
        parts.append("WARNING: profile inconsistencies detected")

    return "; ".join(p for p in parts if p)


def _best_career_role(career: list) -> Dict:
    """Return the most ML-relevant career role (or most recent)."""
    ml_keywords = {"ml", "machine learning", "ai", "nlp", "data scien",
                   "search", "recomm", "research", "retrieval"}
    for job in career:
        t = job.get("title", "").lower()
        if any(kw in t for kw in ml_keywords):
            return job
    return career[0] if career else {}


def _top_jd_skills(skills: list) -> list:
    """Return up to 3 JD-relevant skills with highest endorsements."""
    from ranker.data.jd_profile import JD_SKILL_NAMES
    jd_lower = {s.lower() for s in JD_SKILL_NAMES}
    relevant = [
        sk for sk in skills
        if sk.get("name", "").lower() in jd_lower
    ]
    relevant.sort(key=lambda s: s.get("endorsements", 0), reverse=True)
    return [sk["name"] for sk in relevant[:3]]
