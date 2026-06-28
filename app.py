import streamlit as st
import streamlit.components.v1 as components
import csv
import json
import os
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Runs · Candidate Rankings",
    layout="wide",
    initial_sidebar_state="expanded",
)

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL        = "llama-3.3-70b-versatile"
LLM_CACHE_PATH    = "cache/llm_analysis.json"
SUBMISSIONS_PATH  = "submission.csv"
CANDIDATES_PATH   = "India_runs_data_and_ai_challenge/candidates.jsonl"

JD_CONTEXT = """Role: Senior AI Engineer
Company: Redrob AI (Series A, Pune / Noida, India)
Mission: Own the AI/ML layer of a hiring platform.

Requirements:
- 5-9 years in ML / AI engineering
- Production-grade IR, ranking, search systems
- RAG pipelines, vector DBs (FAISS, Pinecone, Weaviate)
- LLM fine-tuning, RLHF, evaluation frameworks
- Hybrid retrieval (BM25 + dense vectors)
- Python, PyTorch / TensorFlow
- Preferred: Pune or Noida, India"""

# ── Styles + Font Awesome ─────────────────────────────────────────────────────
def inject_styles():
    # In Streamlit 1.3x+, <style> inside st.markdown renders as visible text.
    # Use a zero-height iframe that scripts its styles into the parent <head>.
    css = """
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #080813 !important; }
section[data-testid="stSidebar"] { background: #0d0d1a !important; border-right: 1px solid #1e1e36; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
details { margin-bottom: 8px !important; }
details summary {
    background: #111128 !important; border: 1px solid #1e1e3a !important;
    border-radius: 10px !important; padding: 14px 18px !important;
    cursor: pointer !important; color: #c4c4e0 !important;
    font-weight: 600 !important; font-size: 14px !important;
    transition: border-color .2s, background .2s !important;
}
details summary:hover { border-color: #6366f1 !important; background: #151530 !important; }
details[open] summary {
    border-bottom-left-radius: 0 !important; border-bottom-right-radius: 0 !important;
    border-color: #6366f1 !important; background: #151530 !important;
}
details[open] > div:last-child {
    background: #0f0f22 !important; border: 1px solid #6366f1 !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important; padding: 20px !important;
}
.stSlider > div > div { background: #6366f1 !important; }
.stSlider label { color: #94a3b8 !important; font-size: 13px !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080813; }
::-webkit-scrollbar-thumb { background: #2d2d4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }
hr { border: none !important; border-top: 1px solid #1e1e36 !important; margin: 6px 0 !important; }
"""
    components.html(f"""
<!DOCTYPE html>
<html><head></head><body>
<script>
(function() {{
    var p = window.parent.document;

    // Font Awesome
    if (!p.getElementById('fa6-css')) {{
        var fa = p.createElement('link');
        fa.id = 'fa6-css';
        fa.rel = 'stylesheet';
        fa.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css';
        fa.crossOrigin = 'anonymous';
        p.head.appendChild(fa);
    }}

    // Google Fonts — Inter
    if (!p.getElementById('inter-font')) {{
        var gf = p.createElement('link');
        gf.id = 'inter-font';
        gf.rel = 'stylesheet';
        gf.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap';
        p.head.appendChild(gf);
    }}

    // Custom CSS
    if (!p.getElementById('custom-app-css')) {{
        var s = p.createElement('style');
        s.id = 'custom-app-css';
        s.textContent = `{css}`;
        p.head.appendChild(s);
    }}
}})();
</script>
</body></html>
""", height=0, scrolling=False)


# ── Persistent LLM cache (disk + memory) ─────────────────────────────────────
@st.cache_resource
def get_analysis_cache() -> dict:
    """Loads llm_analysis.json once per server session; returns mutable dict."""
    if os.path.exists(LLM_CACHE_PATH):
        try:
            with open(LLM_CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def persist_analysis(cache: dict, cid: str, text: str):
    cache[cid] = text
    try:
        with open(LLM_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_submissions():
    rows = []
    with open(SUBMISSIONS_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "candidate_id": row["candidate_id"],
                "rank":         int(row["rank"]),
                "score":        float(row["score"]),
                "reasoning":    row["reasoning"],
            })
    return rows


@st.cache_data(show_spinner=False)
def load_profiles(candidate_ids: frozenset):
    profiles = {}
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["candidate_id"] in candidate_ids:
                profiles[c["candidate_id"]] = c
    return profiles


# ── Groq analysis ─────────────────────────────────────────────────────────────
def get_or_generate_analysis(cid: str, candidate: dict, row: dict) -> str:
    cache = get_analysis_cache()
    if cid in cache:
        return cache[cid]

    p   = candidate["profile"]
    sig = candidate.get("redrob_signals", {})

    career_lines = []
    for job in candidate.get("career_history", []):
        tag = " (current)" if job.get("is_current") else ""
        career_lines.append(
            f"- {job['title']} @ {job['company']} [{job['industry']}]"
            f" {job['duration_months']}mo{tag}\n"
            f"  {job.get('description','')[:250]}"
        )

    top_skills = sorted(candidate.get("skills", []),
                        key=lambda s: s.get("endorsements", 0), reverse=True)[:8]
    skills_txt = ", ".join(
        f"{s['name']} ({s['proficiency']}, {s['endorsements']} endorsements)"
        for s in top_skills
    )
    assessments = sig.get("skill_assessment_scores", {})
    assess_txt  = ", ".join(f"{k}: {v}" for k, v in assessments.items()) or "None"

    prompt = f"""You are a senior technical recruiter at Redrob AI.

=== JD ===
{JD_CONTEXT}

=== CANDIDATE #{row['rank']} — Score {row['score']:.4f}/1.0 ===
Title: {p['current_title']} | {p['years_of_experience']} yrs | {p['location']}, {p['country']}
Company: {p['current_company']} ({p.get('current_industry','')})
Headline: {p['headline']}
Summary: {p.get('summary','')[:400]}

Career:
{chr(10).join(career_lines[:4])}

Top Skills: {skills_txt}
Assessments: {assess_txt}
Recruiter saves (30d): {sig.get('saved_by_recruiters_30d','N/A')} | Search views: {sig.get('search_appearance_30d','N/A')}
GitHub score: {sig.get('github_activity_score','N/A')} | Response rate: {sig.get('recruiter_response_rate','N/A')}
Notice period: {sig.get('notice_period_days','N/A')} days | Open to work: {sig.get('open_to_work_flag',False)}

Pipeline reasoning: {row['reasoning']}

Write exactly 4 bullet points explaining WHY this candidate ranks #{row['rank']}. Be specific — reference actual companies, roles, numbers. Note gaps where relevant. No intro or conclusion. Bullets only."""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
            max_tokens=380,
        )
        result = resp.choices[0].message.content.strip()
    except Exception as e:
        result = f"Analysis unavailable: {e}"

    persist_analysis(cache, cid, result)
    return result


# ── HTML helpers ──────────────────────────────────────────────────────────────
def fa(icon, cls=""):
    return f'<i class="fa-solid fa-{icon} {cls}"></i>'


def pill(icon, text, color):
    colors = {
        "indigo": ("rgba(99,102,241,.15)", "#818cf8"),
        "green":  ("rgba(16,185,129,.15)", "#34d399"),
        "blue":   ("rgba(59,130,246,.15)", "#60a5fa"),
        "amber":  ("rgba(245,158,11,.15)", "#fbbf24"),
        "rose":   ("rgba(244,63,94,.15)",  "#fb7185"),
        "slate":  ("rgba(100,116,139,.12)","#94a3b8"),
    }
    bg, fg = colors.get(color, colors["slate"])
    return (
        f"<span style='display:inline-flex;align-items:center;gap:5px;"
        f"background:{bg};color:{fg};border:1px solid {fg}33;"
        f"padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;"
        f"margin:3px 4px 3px 0;'>"
        f"{fa(icon)} {text}</span>"
    )


def metric_card(icon, value, label, color="#6366f1"):
    return (
        f"<div style='background:#111128;border:1px solid #1e1e3a;border-radius:10px;"
        f"padding:14px 12px;text-align:center;'>"
        f"<div style='font-size:11px;color:#64748b;text-transform:uppercase;"
        f"letter-spacing:.06em;margin-bottom:6px;'>{fa(icon)} {label}</div>"
        f"<div style='font-size:24px;font-weight:800;color:{color};'>{value}</div>"
        f"</div>"
    )


def rank_badge(rank):
    if rank == 1:
        bg, fg = "linear-gradient(135deg,#f59e0b,#d97706)", "#fff"
        label  = "#1"
    elif rank == 2:
        bg, fg = "linear-gradient(135deg,#94a3b8,#64748b)", "#fff"
        label  = "#2"
    elif rank == 3:
        bg, fg = "linear-gradient(135deg,#c27803,#92400e)", "#fff"
        label  = "#3"
    elif rank <= 10:
        bg, fg = "linear-gradient(135deg,#10b981,#059669)", "#fff"
        label  = f"#{rank}"
    elif rank <= 50:
        bg, fg = "linear-gradient(135deg,#3b82f6,#2563eb)", "#fff"
        label  = f"#{rank}"
    else:
        bg, fg = "linear-gradient(135deg,#374151,#1f2937)", "#94a3b8"
        label  = f"#{rank}"
    return (
        f"<span style='display:inline-flex;align-items:center;justify-content:center;"
        f"width:38px;height:38px;border-radius:50%;background:{bg};color:{fg};"
        f"font-weight:800;font-size:13px;flex-shrink:0;'>{label}</span>"
    )


def score_range_bar(score, min_score, max_score):
    pct = int((score - min_score) / max(max_score - min_score, 0.001) * 100)
    return (
        f"<div style='margin:10px 0 6px;'>"
        f"<div style='display:flex;justify-content:space-between;font-size:11px;"
        f"color:#64748b;margin-bottom:4px;'>"
        f"<span>{fa('chart-simple')} Match strength</span>"
        f"<span style='color:#e2e8f0;font-weight:700;'>{score:.4f}</span></div>"
        f"<div style='background:#1e1e3a;border-radius:6px;height:8px;position:relative;'>"
        f"<div style='width:{pct}%;height:8px;border-radius:6px;"
        f"background:linear-gradient(90deg,#6366f1,#a78bfa);'></div>"
        f"<div style='position:absolute;right:0;top:-3px;height:14px;width:2px;"
        f"background:#374151;border-radius:2px;'></div>"
        f"</div>"
        f"<div style='display:flex;justify-content:space-between;font-size:10px;"
        f"color:#374151;margin-top:3px;'><span>{min_score:.3f}</span><span>{max_score:.3f}</span></div>"
        f"</div>"
    )


def career_timeline(career_history):
    items = career_history[:5]
    html  = f"<div class='section-label'>{fa('briefcase')} Career History</div>"
    for i, job in enumerate(items):
        is_current = job.get("is_current", False)
        dot_color  = "#10b981" if is_current else "#374151"
        dot_border = "#10b981" if is_current else "#2d2d4a"
        line_color = "#1e1e3a"
        title_color = "#e2e8f0" if is_current else "#c4c4e0"
        co_color   = "#6366f1" if is_current else "#94a3b8"
        curr_badge = (
            f"<span style='background:rgba(16,185,129,.15);color:#34d399;"
            f"font-size:10px;padding:1px 7px;border-radius:20px;margin-left:8px;"
            f"border:1px solid #34d39944;'>current</span>"
            if is_current else ""
        )
        line_html = (
            f"<div style='position:absolute;left:7px;top:18px;bottom:-4px;"
            f"width:2px;background:{line_color};'></div>"
            if i < len(items) - 1 else ""
        )
        desc = job.get("description", "")[:210]
        html += (
            f"<div style='display:flex;gap:14px;margin-bottom:16px;position:relative;'>"
            f"<div style='flex-shrink:0;width:16px;position:relative;'>"
            f"<div style='width:16px;height:16px;border-radius:50%;"
            f"background:{dot_color};border:2px solid {dot_border};margin-top:2px;'></div>"
            f"{line_html}</div>"
            f"<div style='flex:1;padding-bottom:4px;'>"
            f"<div style='font-size:13px;font-weight:700;color:{title_color};'>"
            f"{job['title']}{curr_badge}</div>"
            f"<div style='font-size:12px;color:{co_color};margin:2px 0;'>"
            f"{job['company']} &middot; {job['industry']} &middot; {job['duration_months']}mo</div>"
            f"<div style='font-size:12px;color:#64748b;line-height:1.5;'>{desc}{'...' if len(job.get('description',''))>210 else ''}</div>"
            f"</div></div>"
        )
    return html


def skill_chips(skills):
    prof_style = {
        "expert":       ("rgba(139,92,246,.18)", "#a78bfa", "#6d28d9"),
        "advanced":     ("rgba(59,130,246,.18)", "#60a5fa", "#1d4ed8"),
        "intermediate": ("rgba(100,116,139,.15)","#94a3b8", "#334155"),
        "beginner":     ("rgba(71,85,105,.12)",  "#64748b", "#1e293b"),
    }
    top = sorted(skills, key=lambda s: s.get("endorsements", 0), reverse=True)[:12]
    html = f"<div class='section-label'>{fa('code')} Skills</div><div style='display:flex;flex-wrap:wrap;gap:6px;'>"
    for sk in top:
        bg, fg, border = prof_style.get(sk["proficiency"], prof_style["beginner"])
        sk_name   = sk["name"]
        sk_dur    = sk["duration_months"]
        sk_end    = sk["endorsements"]
        html += (
            f"<span style='background:{bg};color:{fg};border:1px solid {border}66;"
            f"padding:4px 10px;border-radius:20px;font-size:12px;font-weight:500;"
            f"white-space:nowrap;' title='{sk_dur}mo &middot; {sk_end} endorsements'>"
            f"{sk_name}"
            f"<span style='opacity:.6;margin-left:5px;font-size:10px;'>"
            f"{sk_end}{fa('star', 'fa-xs')}</span>"
            f"</span>"
        )
    return html + "</div>"


def assessment_bars(assessments):
    if not assessments:
        return ""
    html = f"<div class='section-label'>{fa('circle-check')} Verified Assessments</div>"
    for skill, val in sorted(assessments.items(), key=lambda x: -x[1]):
        color = "#10b981" if val >= 80 else "#f59e0b" if val >= 60 else "#ef4444"
        html += (
            f"<div style='margin-bottom:10px;'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:12px;color:#c4c4e0;margin-bottom:3px;'>"
            f"<span>{skill}</span>"
            f"<span style='color:{color};font-weight:700;'>{val:.1f}</span></div>"
            f"<div style='background:#1e1e3a;border-radius:4px;height:5px;'>"
            f"<div style='width:{int(val)}%;height:5px;border-radius:4px;"
            f"background:{color};'></div></div>"
            f"</div>"
        )
    return html


def platform_signals(sig):
    rr     = sig.get("recruiter_response_rate", 0)
    github = sig.get("github_activity_score", 0)
    last   = sig.get("last_active_date", "—")
    mode   = sig.get("preferred_work_mode", "—").title()
    html   = f"<div class='section-label'>{fa('chart-bar')} Platform & Engagement</div>"
    rows   = [
        (fa("reply"), "Response rate", f"{int(rr*100)}%"),
        (fa("code-branch"), "GitHub score", f"{github:.1f} / 100"),
        (fa("clock"), "Last active", last),
        (fa("laptop"), "Work mode", mode),
    ]
    for icon, lbl, val in rows:
        html += (
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center;padding:6px 0;"
            f"border-bottom:1px solid #1a1a30;font-size:12px;'>"
            f"<span style='color:#64748b;'>{icon} {lbl}</span>"
            f"<span style='color:#e2e8f0;font-weight:600;'>{val}</span>"
            f"</div>"
        )
    return html


# ── Page sections ─────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
<div style="text-align:center;padding:10px 0 6px;">
  <div style="font-size:13px;color:#6366f1;font-weight:700;letter-spacing:.15em;
              text-transform:uppercase;margin-bottom:6px;">
    <i class="fa-solid fa-trophy"></i>&nbsp; India Runs Data & AI Challenge &nbsp;<i class="fa-solid fa-trophy"></i>
  </div>
  <div style="font-size:32px;font-weight:900;color:#e2e8f0;line-height:1.2;">
    Track 01 &mdash; Intelligent Candidate Discovery
  </div>
  <div style="font-size:14px;color:#64748b;margin-top:6px;">
    <i class="fa-solid fa-building-columns"></i> Senior AI Engineer &nbsp;&bull;&nbsp;
    <i class="fa-solid fa-location-dot"></i> Redrob AI, Pune / Noida &nbsp;&bull;&nbsp;
    <i class="fa-solid fa-users"></i> 100,000 candidates ranked
  </div>
</div>
<hr style="margin:16px 0 20px;"/>
""", unsafe_allow_html=True)


def render_dashboard(submissions):
    scores    = [r["score"] for r in submissions]
    top_score = scores[0]
    avg_score = sum(scores) / len(scores)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("users", "100,000", "Candidates Screened", "#6366f1"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("filter", "100", "Final Shortlist", "#10b981"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("star", f"{top_score:.4f}", "Top Score", "#f59e0b"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("bolt", "6.5s", "Pipeline Runtime", "#a78bfa"),
                    unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # pipeline flow
    st.markdown("""
<div style="display:flex;align-items:center;justify-content:center;gap:4px;
            background:#111128;border:1px solid #1e1e3a;border-radius:10px;
            padding:14px 20px;flex-wrap:wrap;font-size:13px;">
  <div style="text-align:center;padding:0 14px;">
    <div style="font-size:18px;font-weight:800;color:#6366f1;">100K</div>
    <div style="font-size:11px;color:#64748b;"><i class="fa-solid fa-database"></i> All candidates</div>
  </div>
  <div style="color:#2d2d4a;font-size:20px;">&#8594;</div>
  <div style="text-align:center;padding:0 14px;">
    <div style="font-size:18px;font-weight:800;color:#8b5cf6;">12K</div>
    <div style="font-size:11px;color:#64748b;"><i class="fa-solid fa-tag"></i> Title + Industry filter</div>
  </div>
  <div style="color:#2d2d4a;font-size:20px;">&#8594;</div>
  <div style="text-align:center;padding:0 14px;">
    <div style="font-size:18px;font-weight:800;color:#3b82f6;">1K</div>
    <div style="font-size:11px;color:#64748b;"><i class="fa-solid fa-magnifying-glass"></i> BM25 + MiniLM</div>
  </div>
  <div style="color:#2d2d4a;font-size:20px;">&#8594;</div>
  <div style="text-align:center;padding:0 14px;">
    <div style="font-size:18px;font-weight:800;color:#10b981;">100</div>
    <div style="font-size:11px;color:#64748b;"><i class="fa-solid fa-ranking-star"></i> Final shortlist</div>
  </div>
</div>
<div style="height:24px;"></div>
""", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
<div style="padding:16px 0 8px;">
  <div style="font-size:18px;font-weight:800;color:#e2e8f0;">
    {fa('sliders')} Filters
  </div>
</div>
""", unsafe_allow_html=True)

        top_n     = st.slider("Show top N", 5, 100, 25, 5)
        min_score = st.slider("Min score", 0.60, 0.90, 0.60, 0.01, format="%.2f")

        st.markdown(f"""
<hr/>
<div style="font-size:13px;font-weight:700;color:#94a3b8;margin:12px 0 8px;">
  {fa('clipboard-list')} JD at a glance
</div>
""", unsafe_allow_html=True)

        for line in JD_CONTEXT.strip().splitlines():
            color = "#818cf8" if line.startswith("Role") or line.startswith("Company") or line.startswith("Mission") else "#94a3b8"
            dash  = line.startswith("-")
            icon  = f"{fa('circle-dot', 'fa-xs')} " if dash else ""
            display = line.lstrip("- ") if dash else line
            st.markdown(
                f"<p style='font-size:12px;color:{color};margin:4px 0;"
                f"line-height:1.5;'>{icon}{display}</p>",
                unsafe_allow_html=True,
            )

        st.markdown(f"""
<hr/>
<div style="font-size:11px;color:#374151;margin-top:10px;line-height:1.6;">
  {fa('robot')} AI analysis by <b style="color:#6366f1;">Groq</b> (llama-3.3-70b)<br/>
  {fa('floppy-disk')} Cached to disk &mdash; instant on reload<br/>
  {fa('microchip')} Pipeline: BM25 + MiniLM + RRF
</div>
""", unsafe_allow_html=True)

    return top_n, min_score


def section_label(icon, text):
    return (
        f"<div style='font-size:11px;font-weight:700;letter-spacing:.08em;"
        f"text-transform:uppercase;color:#4a4a6a;margin:18px 0 8px;'>"
        f"{fa(icon)} {text}</div>"
    )


def render_candidate_card(row, candidate, min_score, max_score):
    cid   = row["candidate_id"]
    rank  = row["rank"]
    score = row["score"]
    p     = candidate.get("profile", {})
    sig   = candidate.get("redrob_signals", {})

    title   = p.get("current_title", "—")
    yoe     = p.get("years_of_experience", "—")
    loc     = f"{p.get('location','?')}, {p.get('country','')}"
    company = p.get("current_company", "—")
    industry = p.get("current_industry", "")
    otw     = sig.get("open_to_work_flag", False)
    notice  = sig.get("notice_period_days", "")
    saves   = sig.get("saved_by_recruiters_30d", "—")
    views   = sig.get("search_appearance_30d", "—")
    assessments = sig.get("skill_assessment_scores", {})

    label = f"{'[#1]' if rank==1 else '[#2]' if rank==2 else '[#3]' if rank==3 else f'[#{rank}]'}  {cid}  |  {title}  |  {yoe} yrs  |  score {score:.4f}"

    with st.expander(label, expanded=(rank <= 3)):

        # ── top metric row ──────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(metric_card("ranking-star", f"#{rank}", "Rank",
                                    "#f59e0b" if rank <= 3 else "#6366f1"),
                        unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card("percent", f"{score:.3f}", "Score", "#8b5cf6"),
                        unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card("clock", f"{yoe} yrs", "Experience", "#10b981"),
                        unsafe_allow_html=True)
        with m4:
            st.markdown(metric_card("bookmark", str(saves), "Recruiter Saves", "#3b82f6"),
                        unsafe_allow_html=True)
        with m5:
            st.markdown(metric_card("eye", str(views), "Search Views", "#f59e0b"),
                        unsafe_allow_html=True)

        # ── score bar ───────────────────────────────────────────────────────
        st.markdown(score_range_bar(score, min_score, max_score), unsafe_allow_html=True)

        # ── pills ───────────────────────────────────────────────────────────
        pills_html = (
            pill("location-dot", loc, "indigo")
            + pill("building", company, "blue")
            + pill("industry", industry, "slate")
        )
        if otw:
            pills_html += pill("circle-check", "Open to Work", "green")
        if notice == 0:
            pills_html += pill("bolt", "Available Now", "amber")
        elif notice:
            pills_html += pill("calendar", f"{notice}d notice", "amber")

        st.markdown(f"<div style='margin:12px 0;'>{pills_html}</div>",
                    unsafe_allow_html=True)

        # ── headline ────────────────────────────────────────────────────────
        headline = p.get("headline", "")
        if headline:
            st.markdown(
                f"<div style='font-size:13px;color:#64748b;font-style:italic;"
                f"margin-bottom:14px;padding:8px 14px;background:#111128;"
                f"border-left:3px solid #6366f1;border-radius:0 6px 6px 0;'>"
                f"{fa('quote-left')} {headline}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── two-column body ─────────────────────────────────────────────────
        left, right = st.columns([55, 45])

        with left:
            # AI Analysis
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;letter-spacing:.08em;"
                f"text-transform:uppercase;color:#4a4a6a;margin-bottom:8px;'>"
                f"{fa('robot')} AI Analysis &nbsp;"
                f"<span style='color:#6366f1;font-size:10px;font-weight:500;"
                f"text-transform:none;letter-spacing:0;'>"
                f"Groq &middot; llama-3.3-70b</span></div>",
                unsafe_allow_html=True,
            )
            with st.spinner("Loading analysis..."):
                analysis = get_or_generate_analysis(cid, candidate, row)

            # render bullets with custom styling
            bullet_html = "<div style='line-height:1.8;'>"
            for line in analysis.splitlines():
                line = line.strip()
                if not line:
                    continue
                stripped = line.lstrip("-•*").strip()
                bullet_html += (
                    f"<div style='display:flex;gap:8px;margin-bottom:8px;'>"
                    f"<span style='color:#6366f1;flex-shrink:0;margin-top:2px;'>"
                    f"{fa('chevron-right', 'fa-xs')}</span>"
                    f"<span style='font-size:13px;color:#c4c4e0;'>{stripped}</span>"
                    f"</div>"
                )
            bullet_html += "</div>"
            st.markdown(bullet_html, unsafe_allow_html=True)

            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
            st.markdown(career_timeline(candidate.get("career_history", [])),
                        unsafe_allow_html=True)

        with right:
            if assessments:
                st.markdown(assessment_bars(assessments), unsafe_allow_html=True)
                st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

            st.markdown(skill_chips(candidate.get("skills", [])), unsafe_allow_html=True)
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            st.markdown(platform_signals(sig), unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    inject_styles()
    render_header()

    with st.spinner("Loading data..."):
        submissions = load_submissions()
        cids        = frozenset(r["candidate_id"] for r in submissions)
        profiles    = load_profiles(cids)

    render_dashboard(submissions)
    top_n, min_score = render_sidebar()

    visible = [
        r for r in submissions
        if r["rank"] <= top_n and r["score"] >= min_score
    ]

    scores    = [r["score"] for r in submissions]
    max_score = scores[0]
    pool_min  = scores[-1]

    cache    = get_analysis_cache()
    cached_n = sum(1 for r in visible if r["candidate_id"] in cache)
    total_n  = len(visible)

    st.markdown(
        f"<div style='font-size:12px;color:#4a4a6a;margin-bottom:14px;'>"
        f"{fa('list-ol')} Showing <b style='color:#818cf8;'>{total_n}</b> candidates &nbsp;&bull;&nbsp;"
        f"Score range <b style='color:#e2e8f0;'>{visible[-1]['score']:.4f}</b> &ndash; "
        f"<b style='color:#e2e8f0;'>{visible[0]['score']:.4f}</b> &nbsp;&bull;&nbsp;"
        f"{fa('floppy-disk')} <b style='color:#34d399;'>{cached_n}</b> / {total_n} analyses cached"
        f"</div>",
        unsafe_allow_html=True,
    )

    for row in visible:
        cid = row["candidate_id"]
        c   = profiles.get(cid, {})
        render_candidate_card(row, c, pool_min, max_score)

    st.markdown(
        "<p style='text-align:center;color:#1e1e3a;font-size:12px;margin-top:40px;'>"
        "India Runs Data & AI Challenge &middot; Track 01 &middot; "
        "BM25 + MiniLM + Groq &middot; Team s24cseu1655</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
