"""
Industry and company quality signals.
Data-driven: good_industry r=+0.32, bad_industry r=-0.14 with proxy label.
Consulting company penalty DROPPED (r=+0.02, noise).
"""

GOOD_INDUSTRIES = {
    "AI/ML", "SaaS", "Fintech", "Food Delivery", "E-commerce",
    "EdTech", "HealthTech", "Conversational AI", "Gaming",
    "HealthTech AI", "AdTech", "Transportation", "Insurance Tech",
    "AI Services", "Software",
}

BAD_INDUSTRIES = {
    "Manufacturing", "Paper Products", "Conglomerate",
}

# IT Services treated as neutral (slightly negative but not strongly)
NEUTRAL_INDUSTRIES = {"IT Services", "Consulting"}

INDUSTRY_SCORES = {
    "AI/ML": 1.0,
    "Conversational AI": 1.0,
    "AI Services": 0.95,
    "SaaS": 0.85,
    "HealthTech AI": 0.85,
    "Fintech": 0.80,
    "Food Delivery": 0.75,
    "E-commerce": 0.75,
    "EdTech": 0.70,
    "Gaming": 0.65,
    "HealthTech": 0.65,
    "AdTech": 0.60,
    "Transportation": 0.55,
    "Insurance Tech": 0.55,
    "Software": 0.50,
    "IT Services": 0.25,
    "Consulting": 0.20,
    "Manufacturing": 0.10,
    "Conglomerate": 0.10,
    "Paper Products": 0.05,
}

# ML-relevant career industries for trajectory scoring
ML_ADJACENT_INDUSTRIES = {
    "AI/ML", "Conversational AI", "AI Services", "HealthTech AI",
    "SaaS", "Fintech", "Food Delivery", "E-commerce", "EdTech",
    "Gaming", "AdTech", "Software",
}


def get_industry_score(industry: str) -> float:
    return INDUSTRY_SCORES.get(industry, 0.30)


def is_good_industry(industry: str) -> bool:
    return industry in GOOD_INDUSTRIES


def is_bad_industry(industry: str) -> bool:
    return industry in BAD_INDUSTRIES
