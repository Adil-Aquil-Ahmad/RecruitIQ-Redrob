# Redrob Hackathon — Intelligent Candidate Discovery

Three-stage CPU ranking pipeline for the Senior AI Engineer JD. Ranks 100K candidates into a top-100 shortlist in under 60 seconds.

## Quick Start

```bash
pip install -r requirements.txt

# Step 1 — offline precomputation (~5 min, run once)
python precompute.py --candidates ./candidates.jsonl

# Step 2 — generate submission (~30 sec)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Validate
python validate_submission.py submission.csv
```

## Architecture

```
100K candidates
      │
      ▼ Stage 1 — Title + Industry + YoE scoring (O(N), ~0.1s)
   12K candidates
      │
      ▼ Stage 2 — BM25 career text + MiniLM embeddings via RRF (~5s)
   1K candidates
      │
      ▼ Stage 3 — Full feature matrix scoring (~0.1s)
   Top 100
```

### Stage 1: Title Classification (100K → 12K)

Scores every candidate on title category (ml_core/tech_adjacent/wrong), industry goodness, and YoE. Eliminates ~88K clearly irrelevant candidates (HR Managers, Accountants, Mechanical Engineers) in 100ms.

**Data evidence**: ML-core candidates have 4–5× higher recruiter saves/searches than wrong-titled candidates.

### Stage 2: Hybrid Retrieval — BM25 + Embeddings (12K → 1K)

- **BM25** over concatenated career descriptions + headline + summary, queried with an expanded JD vocabulary including synonyms (e.g. "dense retrieval" alongside "semantic search")
- **MiniLM** (`all-MiniLM-L6-v2`) embeddings of headline+summary, compared to JD query embedding
- Combined via **Reciprocal Rank Fusion** (k=60), which is robust to different score scales

### Stage 3: Full Feature Scoring (1K → 100)

Eight feature groups, weights validated against platform recruiter-behavior proxy signals:

| Group | Weight | Key signals |
|---|---|---|
| Platform signals | 28% | `search_appearance_30d × saved_by_recruiters_30d` (r=0.71/0.69 with proxy label) |
| Career text | 20% | Weighted JD keyword hits in career descriptions + headline |
| Title + ML career | 18% | ML title category + months in ML/AI roles across career history |
| Assessment scores | 10% | Verified JD-relevant skill assessments (bonus only, not penalizing absence) |
| Industry | 8% | Good industries (AI/ML, SaaS, Fintech) vs bad (Manufacturing, Paper Products) |
| GitHub activity | 6% | Activity score (ML-core: avg 46 vs wrong-title: avg 7) |
| Engagement | 5% | Recruiter response rate + recency + open-to-work |
| Education + Location | 5% | Institution tier + India/preferred city |

**Dropped signals** (data-validated as noise):
- Consulting firm penalty: r=+0.02 — captured implicitly by industry and career text
- Notice period: r=-0.02 — only 22 candidates below 30 days across 100K

### Honeypot Detection

Requires 2+ impossible signals to apply penalty (high precision, avoids false positives):
- Expert-proficiency skill with 0 months duration
- Career duration exceeds years-of-experience + 3-year buffer
- Impossible education timelines

### Top-10 Sanity Pass

After scoring, verifies that no clearly wrong candidate (wrong title + no ML career evidence + no JD text hits) appears in positions 1–10. Swaps with the highest-scoring acceptable candidate from positions 11–50.

## Key Design Decisions

**Why not pure keyword matching?** Skills are uniformly distributed across 100K candidates (~12K occurrences per skill). Raw skill-count is noise. Career descriptions are the real signal.

**Why platform signals at 28%?** Validated correlation: `search_appearance_30d` r=+0.71, `saved_by_recruiters_30d` r=+0.69 against recruiter-behavior proxy label. These reflect actual recruiter intent.

**Why no consulting penalty?** Data shows r=+0.02 (near zero). The JD fit signals (industry + career text) already capture this implicitly.

**Why YoE bell curve?** Raw YoE has r=-0.12 with quality proxy — more experience slightly correlates with *lower* platform activity. The JD wants 5–9 years; candidates above 12 years get a soft penalty.

## Compute Constraints

- CPU only, no GPU
- No network calls during ranking
- Precomputation: ~5 minutes (embeddings + BM25 index)
- Ranking step: ~30 seconds
- RAM: < 4GB (100K × 384 float32 embeddings = ~153MB)

## Files

```
rank.py                  # Entry point
precompute.py            # Offline BM25 + embedding computation
config.yaml              # Weights and thresholds
requirements.txt
submission_metadata.yaml
ranker/
  data/                  # JD profile, title taxonomy, industry signals
  features/              # 6 feature group modules
  scoring/               # Normalizer, ensemble, reasoning
  pipeline/              # Stage 1, 2, 3
  utils/                 # Text, cache
cache/                   # Precomputed artifacts (gitignored)
```
