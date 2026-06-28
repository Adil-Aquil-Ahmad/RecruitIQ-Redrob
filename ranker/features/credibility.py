"""
Credibility and honeypot detection.

Data findings:
- Only 84 expert-skills with duration=0 across 100K candidates (0.08%)
- Timeline inconsistencies: only 22 candidates (0.02%) — weak honeypot signal
- Require 2+ signals to apply penalty (high precision, avoid false positives)
- Profile completeness and verification are legitimate quality signals
"""

from typing import Any, Dict, List


def extract(candidate: Dict[str, Any]) -> Dict[str, float]:
    sig = candidate["redrob_signals"]
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate["profile"]
    education = candidate.get("education", [])

    # --- Honeypot detection ---
    honeypot_signals = 0

    # Signal 1: Expert skill with 0 duration (6.4% of all expert skills are suspicious)
    expert_zero = sum(
        1 for sk in skills
        if sk.get("proficiency") == "expert" and sk.get("duration_months", 1) == 0
    )
    if expert_zero >= 1:
        honeypot_signals += expert_zero

    # Signal 2: Career duration exceeds YoE significantly
    yoe = profile.get("years_of_experience", 0)
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    if total_career_months > yoe * 12 + 36:  # Allow 3yr overlap buffer
        honeypot_signals += 1

    # Signal 3: Education timeline impossible
    for edu in education:
        start = edu.get("start_year", 2000)
        end = edu.get("end_year", 2004)
        if end < start or (end - start) > 10:
            honeypot_signals += 1

    # Signal 4: Skills with extreme proficiency but very low endorsements
    high_prof_low_endorse = sum(
        1 for sk in skills
        if sk.get("proficiency") == "expert"
        and sk.get("endorsements", 0) == 0
        and sk.get("duration_months", 0) < 6
    )
    if high_prof_low_endorse >= 3:
        honeypot_signals += 1

    # Require 2+ signals for penalty (high precision)
    is_honeypot = honeypot_signals >= 2
    honeypot_penalty = 0.30 if is_honeypot else 0.0
    honeypot_probability = min(honeypot_signals / 4.0, 1.0)

    # --- Profile verification ---
    verified_email = float(sig.get("verified_email", False))
    verified_phone = float(sig.get("verified_phone", False))
    linkedin = float(sig.get("linkedin_connected", False))
    verification_score = (verified_email + verified_phone + linkedin) / 3.0

    # --- Profile completeness ---
    completeness = sig.get("profile_completeness_score", 0) / 100.0

    # --- Profile credibility composite ---
    credibility_score = (
        0.40 * verification_score +
        0.35 * completeness +
        0.25 * (1.0 - honeypot_probability)
    )

    return {
        "honeypot_signals": float(honeypot_signals),
        "is_honeypot": float(is_honeypot),
        "honeypot_penalty": honeypot_penalty,
        "honeypot_probability": honeypot_probability,
        "verification_score": verification_score,
        "completeness": completeness,
        "credibility_score": credibility_score,
        "expert_zero_duration": float(expert_zero),
    }
