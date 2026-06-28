"use client";
import { useState, useMemo } from "react";
import type { Candidate } from "../types";
import CandidateCard from "./CandidateCard";

const PIPELINE_STEPS = [
  { n: "100K", label: "All Candidates", icon: "🗄" },
  { n: "12K",  label: "Title + Industry Filter", icon: "🏷" },
  { n: "1K",   label: "BM25 + MiniLM",           icon: "🔍" },
  { n: "100",  label: "Final Shortlist",          icon: "🏆" },
];

export default function RankingDashboard({
  candidates, maxScore, minScore,
}: {
  candidates: Candidate[];
  maxScore: number;
  minScore: number;
}) {
  const [topN, setTopN]         = useState(25);
  const [minScoreF, setMinScoreF] = useState(minScore);
  const [search, setSearch]     = useState("");

  const visible = useMemo(() =>
    candidates.filter(c =>
      c.rank <= topN &&
      c.score >= minScoreF &&
      (search === "" ||
        c.candidate_id.toLowerCase().includes(search.toLowerCase()) ||
        c.profile.current_title.toLowerCase().includes(search.toLowerCase()) ||
        c.profile.current_company.toLowerCase().includes(search.toLowerCase()) ||
        c.profile.location.toLowerCase().includes(search.toLowerCase()))
    ),
  [candidates, topN, minScoreF, search]);

  const avgScore = (candidates.reduce((s, c) => s + c.score, 0) / candidates.length).toFixed(4);

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: "#080813" }}>

      {/* ── Sidebar ── */}
      <aside style={{
        width: 260, flexShrink: 0, background: "#0d0d1a",
        borderRight: "1px solid #1e1e3a",
        padding: "24px 16px", display: "flex", flexDirection: "column", gap: 24,
        position: "sticky", top: 0, height: "100vh", overflowY: "auto",
      }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", color: "#6366f1", marginBottom: 4 }}>
            RecruitIQ
          </div>
          <div style={{ fontSize: 15, fontWeight: 800, color: "#e2e8f0", lineHeight: 1.3 }}>
            India Runs<br />Data &amp; AI Challenge
          </div>
          <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>Track 01 · Candidate Discovery</div>
        </div>

        <hr style={{ border: "none", borderTop: "1px solid #1e1e3a" }} />

        {/* Filters */}
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "#64748b", marginBottom: 12 }}>
            ⚙ Filters
          </div>

          <label style={{ fontSize: 12, color: "#94a3b8", display: "block", marginBottom: 6 }}>
            Show top&nbsp;<strong style={{ color: "#818cf8" }}>{topN}</strong>
          </label>
          <input type="range" min={5} max={100} step={5} value={topN}
            onChange={e => setTopN(+e.target.value)}
            style={{ width: "100%", accentColor: "#6366f1", marginBottom: 14 }} />

          <label style={{ fontSize: 12, color: "#94a3b8", display: "block", marginBottom: 6 }}>
            Min score&nbsp;<strong style={{ color: "#818cf8" }}>{minScoreF.toFixed(3)}</strong>
          </label>
          <input type="range"
            min={minScore} max={maxScore} step={0.001}
            value={minScoreF}
            onChange={e => setMinScoreF(+e.target.value)}
            style={{ width: "100%", accentColor: "#6366f1", marginBottom: 14 }} />

          <input
            type="text" placeholder="Search by ID, title, company…"
            value={search} onChange={e => setSearch(e.target.value)}
            style={{
              width: "100%", background: "#111128", border: "1px solid #1e1e3a",
              borderRadius: 8, padding: "8px 10px", fontSize: 12, color: "#e2e8f0",
              outline: "none",
            }}
          />
        </div>

        <hr style={{ border: "none", borderTop: "1px solid #1e1e3a" }} />

        {/* JD */}
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "#64748b", marginBottom: 10 }}>
            📋 Job Description
          </div>
          {[
            ["Role",    "Senior AI Engineer"],
            ["Company", "Redrob AI (Series A)"],
            ["Location","Pune / Noida, India"],
            ["YoE",     "5–9 years"],
            ["Stack",   "RAG · BM25 · Vector DBs · LLMs · PyTorch"],
          ].map(([k, v]) => (
            <div key={k} style={{ marginBottom: 7 }}>
              <div style={{ fontSize: 10, color: "#4a4a6a", textTransform: "uppercase", letterSpacing: "0.06em" }}>{k}</div>
              <div style={{ fontSize: 12, color: "#94a3b8" }}>{v}</div>
            </div>
          ))}
        </div>

        <hr style={{ border: "none", borderTop: "1px solid #1e1e3a" }} />

        <div style={{ fontSize: 11, color: "#374151", lineHeight: 1.8 }}>
          🤖 AI by <span style={{ color: "#6366f1" }}>Groq</span> llama-3.3-70b<br />
          🔍 BM25 + MiniLM + RRF<br />
          ⏱ 6.5s pipeline · CPU only<br />
          👥 Team s24cseu1655
        </div>
      </aside>

      {/* ── Main ── */}
      <main style={{ flex: 1, padding: "28px 32px", maxWidth: 1100, minWidth: 0 }}>

        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 11, color: "#6366f1", fontWeight: 700, letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: 6 }}>
            🏆 India Runs Data &amp; AI Challenge
          </div>
          <h1 style={{ fontSize: 28, fontWeight: 900, color: "#e2e8f0", lineHeight: 1.2, marginBottom: 4 }}>
            Intelligent Candidate Discovery
          </h1>
          <p style={{ fontSize: 14, color: "#64748b" }}>
            Senior AI Engineer · Redrob AI · 100,000 candidates ranked in 6.5 seconds
          </p>
        </div>

        {/* Stats row */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 20 }}>
          {[
            { icon: "👥", label: "Screened",    value: "100,000", color: "#6366f1" },
            { icon: "🎯", label: "Shortlisted", value: "100",     color: "#10b981" },
            { icon: "⭐", label: "Top Score",   value: maxScore.toFixed(4), color: "#f59e0b" },
            { icon: "⚡", label: "Runtime",     value: "6.5s",    color: "#a78bfa" },
          ].map(s => (
            <div key={s.label} style={{
              background: "#111128", border: "1px solid #1e1e3a", borderRadius: 10,
              padding: "14px 12px", textAlign: "center",
            }}>
              <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 6 }}>
                {s.icon} {s.label}
              </div>
              <div style={{ fontSize: 22, fontWeight: 800, color: s.color }}>{s.value}</div>
            </div>
          ))}
        </div>

        {/* Pipeline banner */}
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          gap: 6, background: "#111128", border: "1px solid #1e1e3a",
          borderRadius: 10, padding: "14px 20px", marginBottom: 24, flexWrap: "wrap",
        }}>
          {PIPELINE_STEPS.map((step, i) => (
            <div key={step.n} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{ textAlign: "center", padding: "0 10px" }}>
                <div style={{ fontSize: 17, fontWeight: 800, color: i === 3 ? "#10b981" : "#6366f1" }}>{step.n}</div>
                <div style={{ fontSize: 10, color: "#64748b" }}>{step.icon} {step.label}</div>
              </div>
              {i < PIPELINE_STEPS.length - 1 && (
                <span style={{ color: "#2d2d4a", fontSize: 16 }}>→</span>
              )}
            </div>
          ))}
        </div>

        {/* Results count */}
        <div style={{ fontSize: 12, color: "#4a4a6a", marginBottom: 14 }}>
          Showing <strong style={{ color: "#818cf8" }}>{visible.length}</strong> candidates
          &nbsp;·&nbsp; score range&nbsp;
          <strong style={{ color: "#e2e8f0" }}>
            {visible.length ? visible[visible.length - 1].score.toFixed(4) : "—"}
          </strong>
          &nbsp;–&nbsp;
          <strong style={{ color: "#e2e8f0" }}>
            {visible.length ? visible[0].score.toFixed(4) : "—"}
          </strong>
          &nbsp;·&nbsp; avg pool score&nbsp;
          <strong style={{ color: "#e2e8f0" }}>{avgScore}</strong>
        </div>

        {/* Cards */}
        {visible.map(c => (
          <CandidateCard key={c.candidate_id} c={c} minScore={minScore} maxScore={maxScore} />
        ))}

        {visible.length === 0 && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "#374151" }}>
            No candidates match your filters.
          </div>
        )}

        <div style={{ textAlign: "center", color: "#1e1e3a", fontSize: 12, marginTop: 40 }}>
          RecruitIQ · India Runs Data &amp; AI Challenge · Track 01 · Team s24cseu1655
        </div>
      </main>
    </div>
  );
}
