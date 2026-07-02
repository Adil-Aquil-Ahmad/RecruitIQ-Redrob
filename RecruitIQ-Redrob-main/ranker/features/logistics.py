"""
Logistics features: location and work preference (2% weight).
willing_to_relocate r=+0.03 — very weak, used only for tie-breaking.
"""

from typing import Any, Dict

from ranker.data.jd_profile import IDEAL_PROFILE

_PREFERRED_LOCATIONS = {loc.lower() for loc in IDEAL_PROFILE["preferred_locations"]}


def extract(candidate: Dict[str, Any]) -> Dict[str, float]:
    profile = candidate["profile"]
    sig = candidate["redrob_signals"]

    country = (profile.get("country") or "").lower().strip()
    location = (profile.get("location") or "").lower().strip()
    willing_to_relocate = bool(sig.get("willing_to_relocate", False))
    work_mode = (sig.get("preferred_work_mode") or "").lower()

    # Location scoring
    is_india = country == "india"
    in_preferred_city = any(city in location for city in _PREFERRED_LOCATIONS)

    if is_india and in_preferred_city:
        location_score = 1.0
    elif is_india:
        location_score = 0.8
    elif willing_to_relocate:
        location_score = 0.5
    else:
        location_score = 0.15

    # Work mode compatibility (JD: Pune/Noida hybrid)
    if work_mode in {"hybrid", "onsite", "flexible"}:
        work_mode_score = 1.0
    elif work_mode == "remote":
        work_mode_score = 0.5
    else:
        work_mode_score = 0.7

    logistics_score = 0.75 * location_score + 0.25 * work_mode_score

    return {
        "location_score": location_score,
        "is_india": float(is_india),
        "in_preferred_city": float(in_preferred_city),
        "willing_to_relocate": float(willing_to_relocate),
        "work_mode_score": work_mode_score,
        "logistics_score": logistics_score,
    }
