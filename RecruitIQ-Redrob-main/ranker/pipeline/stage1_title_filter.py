"""
Stage 1: Title + Industry + YoE fast filter.
100K → 12K candidates using O(N) scoring.

Title signal validated: ML core has 4-5x higher platform scores than wrong titles.
Output is generous (12K not 5K) to preserve recall for Stage 2.
"""

import logging
from typing import Any, Dict, List, Tuple

from ranker.data.title_taxonomy import classify_title, TITLE_STAGE1_SCORES
from ranker.data.company_signals import get_industry_score

log = logging.getLogger(__name__)


def score_candidate(candidate: Dict[str, Any]) -> float:
    profile = candidate["profile"]
    title = profile.get("current_title", "")
    industry = profile.get("current_industry", "")
    yoe = float(profile.get("years_of_experience", 0))

    title_cat = classify_title(title)
    title_score = TITLE_STAGE1_SCORES.get(title_cat, -0.5)

    # Industry contribution
    ind_score = get_industry_score(industry)
    # Map 0-1 industry score to -0.5 to +1.0 range
    industry_component = (ind_score - 0.3) * 1.43

    # YoE: hard disqualify under 1 year, soft penalty under 3
    if yoe < 1.0:
        yoe_component = -3.0
    elif yoe < 3.0:
        yoe_component = -0.5
    elif yoe <= 9.0:
        yoe_component = 0.5
    else:
        yoe_component = 0.2

    return title_score + industry_component + yoe_component


def run(candidates: List[Dict[str, Any]], output_size: int = 12000) -> List[Dict[str, Any]]:
    log.info("Stage 1: scoring %d candidates", len(candidates))

    scored: List[Tuple[float, Dict]] = []
    for c in candidates:
        s = score_candidate(c)
        scored.append((s, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [c for _, c in scored[:output_size]]

    # Log category breakdown for diagnostics
    from collections import Counter
    cats = Counter(classify_title(c["profile"]["current_title"]) for c in result)
    log.info("Stage 1 output: %d candidates | categories: %s", len(result), dict(cats))

    return result
