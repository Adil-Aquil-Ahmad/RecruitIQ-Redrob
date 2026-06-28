"""Text preprocessing utilities."""

import re
from typing import List


_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    if not text:
        return ""
    return _WHITESPACE.sub(" ", text.lower()).strip()


def tokenize(text: str) -> List[str]:
    text = normalize(text)
    return [t for t in re.split(r"[^a-z0-9]", text) if len(t) > 1]


def candidate_bm25_text(candidate: dict) -> str:
    """Full text corpus for BM25 indexing: career + headline + summary."""
    parts = [
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    for sk in candidate.get("skills", []):
        parts.append(sk.get("name", ""))
    return " ".join(p for p in parts if p)


def candidate_embed_text(candidate: dict) -> str:
    """Short text for embedding: headline + summary only."""
    h = candidate["profile"].get("headline", "")
    s = candidate["profile"].get("summary", "")
    return f"{h} {s}".strip()


def candidate_career_text(candidate: dict) -> str:
    """Career descriptions only — for JD keyword hit counting."""
    parts = []
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", ""))
    return " ".join(parts)


def count_keyword_hits(text: str, keywords: List[str]) -> int:
    """Count how many distinct keywords appear in text (case-insensitive)."""
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def count_keyword_hits_weighted(text: str, keyword_weights: dict) -> float:
    """Sum weights of keywords present in text."""
    t = text.lower()
    return sum(w for kw, w in keyword_weights.items() if kw in t)
