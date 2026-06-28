"""
Behavioral / platform signal features.

CRITICAL DATA FINDING: Platform signals are the strongest features in the dataset:
  search_appearance_30d:   r = +0.7135 with proxy label
  saved_by_recruiters_30d: r = +0.6867 with proxy label

These two features alone drive most of the ranking signal.
All other behavioral signals are secondary (r < 0.10).

Notice period DROPPED (r = -0.017, essentially noise, only 22 candidates < 30d).
"""

import math
from datetime import datetime
from typing import Any, Dict

_TODAY = datetime(2026, 6, 26)


def extract(candidate: Dict[str, Any]) -> Dict[str, float]:
    sig = candidate["redrob_signals"]

    # --- Platform composite (strongest feature pair) ---
    search = float(sig.get("search_appearance_30d", 0))
    saved = float(sig.get("saved_by_recruiters_30d", 0))
    # log1p compresses outliers (max search=1490, mean=118)
    platform_composite = math.log1p(search) * math.log1p(saved)

    # Individual platform signals (for reasoning)
    platform_search_norm = math.log1p(search)
    platform_saved_norm = math.log1p(saved)

    # --- Open to work ---
    open_to_work = float(sig.get("open_to_work_flag", False))

    # --- Active recency: exponential decay ---
    try:
        last_active = datetime.fromisoformat(sig["last_active_date"])
        days_since = max(((_TODAY - last_active).days), 0)
    except Exception:
        days_since = 180
    # Half-life 90 days: e^(-days/90). 30d→0.72, 90d→0.37, 180d→0.14
    recency_score = math.exp(-days_since / 90.0)

    # --- Recruiter response rate (r=+0.09, weak but real) ---
    response_rate = float(sig.get("recruiter_response_rate", 0.0))

    # --- Interview completion (r small but signals reliability) ---
    interview_rate = float(sig.get("interview_completion_rate", 0.0))

    # --- Applications submitted (active job seeker signal) ---
    applications = float(sig.get("applications_submitted_30d", 0))
    application_score = math.log1p(applications) / math.log1p(20)  # normalize

    # --- Composite engagement ---
    # Combines recency + open_to_work + response rate
    engagement_composite = (
        0.40 * recency_score +
        0.25 * open_to_work +
        0.20 * response_rate +
        0.10 * interview_rate +
        0.05 * min(application_score, 1.0)
    )

    return {
        "platform_composite": platform_composite,
        "platform_search_norm": platform_search_norm,
        "platform_saved_norm": platform_saved_norm,
        "search_30d": search,
        "saved_30d": saved,
        "open_to_work": open_to_work,
        "recency_score": recency_score,
        "days_since_active": float(days_since),
        "response_rate": response_rate,
        "interview_completion": interview_rate,
        "engagement_composite": engagement_composite,
    }
