"""
Title → category mapping. Categories drive Stage 1 scoring.
Derived from analysis of the 100K candidate distribution.
"""

ML_CORE_TITLES = {
    "ml engineer", "machine learning engineer", "senior ml engineer",
    "junior ml engineer", "staff machine learning engineer", "lead ml engineer",
    "principal ml engineer", "ai engineer", "senior ai engineer", "lead ai engineer",
    "ai research engineer", "applied ml engineer", "applied scientist",
    "data scientist", "senior data scientist", "lead data scientist",
    "principal data scientist", "staff data scientist",
    "nlp engineer", "senior nlp engineer",
    "search engineer", "senior search engineer",
    "recommendation systems engineer", "recommender systems engineer",
    "ai specialist", "machine learning scientist",
    "deep learning engineer", "research engineer",
    "computer vision engineer",
}

TECH_ADJACENT_TITLES = {
    "software engineer", "senior software engineer", "staff software engineer",
    "backend engineer", "senior backend engineer",
    "full stack developer", "senior full stack developer",
    "data engineer", "senior data engineer", "lead data engineer",
    "cloud engineer", "senior cloud engineer",
    "devops engineer", "senior devops engineer", "platform engineer",
    "java developer", "senior java developer",
    ".net developer", "senior .net developer",
    "mobile developer", "ios developer", "android developer",
    "frontend engineer", "senior frontend engineer",
    "python developer", "senior python developer",
    "analytics engineer", "bi engineer", "data analyst",
    "site reliability engineer", "sre",
    "infrastructure engineer", "systems engineer",
    "solutions architect", "technical architect",
    "senior software engineer (ml)",
}

WRONG_TITLES = {
    "hr manager", "human resources manager", "hr executive", "talent acquisition",
    "accountant", "chartered accountant", "financial analyst", "finance manager",
    "business analyst", "senior business analyst", "business intelligence analyst",
    "project manager", "senior project manager", "program manager",
    "customer support", "customer service", "customer success manager",
    "operations manager", "senior operations manager",
    "content writer", "technical writer", "copywriter",
    "sales executive", "sales manager", "account executive", "business development",
    "civil engineer", "structural engineer",
    "mechanical engineer", "manufacturing engineer",
    "graphic designer", "ui designer", "ux designer",
    "marketing manager", "digital marketing manager", "seo specialist",
    "supply chain manager", "logistics manager",
    "legal counsel", "lawyer",
}


def classify_title(title: str) -> str:
    """Returns 'ml_core', 'tech_adjacent', 'wrong', or 'other'."""
    t = title.lower().strip()
    if t in ML_CORE_TITLES:
        return "ml_core"
    if t in TECH_ADJACENT_TITLES:
        return "tech_adjacent"
    if t in WRONG_TITLES:
        return "wrong"
    # Fuzzy fallback
    ml_keywords = {"ml", "machine learning", "ai ", " ai", "nlp", "deep learning",
                   "data scien", "search engin", "recomm", "applied sci"}
    if any(kw in t for kw in ml_keywords):
        return "ml_core"
    tech_keywords = {"engineer", "developer", "architect", "devops", "backend",
                     "frontend", "fullstack", "full stack", "platform", "sre", "data"}
    if any(kw in t for kw in tech_keywords):
        return "tech_adjacent"
    return "other"


TITLE_STAGE1_SCORES = {
    "ml_core": 3.0,
    "tech_adjacent": 1.0,
    "other": -0.5,
    "wrong": -2.5,
}
