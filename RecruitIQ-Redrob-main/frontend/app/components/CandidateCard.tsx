"use client";
import { useState } from "react";
import type { Candidate } from "../types";
import MetricBox from "./MetricBox";

const PROF_COLORS: Record<string, [string, string]> = {
  expert:       ["rgba(139,92,246,.18)", "#a78bfa"],
  advanced:     ["rgba(59,130,246,.18)",  "#60a5fa"],
  intermediate: ["rgba(100,116,139,.15)","#94a3b8"],
  beginner:     ["rgba(71,85,105,.12)",   "#64748b"],
};

function rankStyle(rank: number): { bg: string; color: string } {
  if (rank === 1) return { bg: "linear-gradient(135deg,#f59e0b,#d97706)", color: "#fff" };
  if (rank === 2) return { bg: "linear-gradient(135deg,#94a3b8,#64748b)", color: "#fff" };
  if (rank === 3) return { bg: "linear-gradient(135deg,#c27803,#92400e)", color: "#fff" };
  if (rank <= 10) return { bg: "linear-gradient(135deg,#10b981,#059669)", color: "#fff" };
  if (rank <= 50) return { bg: "linear-gradient(135deg,#3b82f6,#2563eb)", color: "#fff" };
  return { bg: "linear-gradient(135deg,#374151,#1f2937)", color: "#94a3b8" };
}

function AssessmentBar({ skill, score }: { skill: string; score: number }) {
  const color = score >= 80 ? "#10b981" : score >= 60 ? "#f59e0b" : "#ef4444";
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#c4c4e0", marginBottom: 3 }}>
        <span>{skill}</span>
        <span style={{ color, fontWeight: 700 }}>{score.toFixed(1)}</span>
      </div>
      <div style={{ background: "#1e1e3a", borderRadius: 4, height: 5 }}>
        <div style={{ width: `${score}%`, height: 5, borderRadius: 4, background: color, transition: "width .4s" }} />
      </div>
    </div>
  );
}

export default function CandidateCard({ c, minScore, maxScore }: {
  c: Candidate;
  minScore: number;
  maxScore: number;
}) {
  const [open, setOpen] = useState(c.rank <= 3);
  const { bg, color } = rankStyle(c.rank);
  const p   = c.profile;
  const sig = c.redrob_signals;
  const pct = Math.round(((c.score - minScore) / Math.max(maxScore - minScore, 0.001)) * 100);

  const topSkills = [...c.skills].sort((a, b) => b.endorsements - a.endorsements).slice(0, 10);
  const assessments = Object.entries(sig.skill_assessment_scores ?? {}).sort((a, b) => b[1] - a[1]);

  const pills = [
    { icon: "📍", text: `${p.location}, ${p.country}`, bg: "rgba(99,102,241,.13)", fg: "#818cf8" },
    { icon: "🏢", text: p.current_company, bg: "rgba(59,130,246,.13)", fg: "#60a5fa" },
    { icon: "🏭", text: p.current_industry, bg: "rgba(100,116,139,.12)", fg: "#94a3b8" },
    ...(sig.open_to_work_flag ? [{ icon: "✅", text: "Open to Work", bg: "rgba(16,185,129,.13)", fg: "#34d399" }] : []),
    ...(sig.notice_period_days === 0
      ? [{ icon: "⚡", text: "Available Now", bg: "rgba(245,158,11,.13)", fg: "#fbbf24" }]
      : sig.notice_period_days
        ? [{ icon: "📅", text: `${sig.notice_period_days}d notice`, bg: "rgba(245,158,11,.13)", fg: "#fbbf24" }]
        : []),
  ];

  return (
    <div style={{ marginBottom: 8 }}>
      {/* Header row — always visible */}
      <div
        onClick={() => setOpen(!open)}
        style={{
          display: "flex", alignItems: "center", gap: 14,
          background: open ? "#151530" : "#111128",
          border: `1px solid ${open ? "#6366f1" : "#1e1e3a"}`,
          borderRadius: open ? "12px 12px 0 0" : 12,
          padding: "14px 18px",
          cursor: "pointer",
          transition: "all .2s",
        }}
      >
        {/* Rank badge */}
        <div style={{
          flexShrink: 0, width: 40, height: 40, borderRadius: "50%",
          background: bg, color, fontWeight: 800, fontSize: 13,
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>#{c.rank}</div>

        {/* Title + meta */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            {c.candidate_id} &nbsp;·&nbsp; {p.current_title}
          </div>
          <div style={{ fontSize: 12, color: "#64748b", marginTop: 2 }}>
            {p.current_company} &nbsp;·&nbsp; {p.location} &nbsp;·&nbsp; {p.years_of_experience} yrs
          </div>
        </div>

        {/* Score pill */}
        <div style={{
          flexShrink: 0, background: "rgba(99,102,241,.15)",
          border: "1px solid #6366f133", borderRadius: 20,
          padding: "4px 12px", fontSize: 13, fontWeight: 700, color: "#818cf8",
        }}>
          {c.score.toFixed(4)}
        </div>

        {/* Chevron */}
        <div style={{ flexShrink: 0, color: "#6366f1", fontSize: 12, transition: "transform .2s", transform: open ? "rotate(180deg)" : "rotate(0)" }}>
          ▼
        </div>
      </div>

      {/* Expanded body */}
      {open && (
        <div style={{
          background: "#0f0f22",
          border: "1px solid #6366f1",
          borderTop: "none",
          borderRadius: "0 0 12px 12px",
          padding: 20,
        }}>
          {/* Metrics row */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 10, marginBottom: 16 }}>
            <MetricBox icon="🏆" label="Rank"           value={`#${c.rank}`}              color={c.rank <= 3 ? "#f59e0b" : "#6366f1"} />
            <MetricBox icon="📊" label="Score"          value={c.score.toFixed(3)}         color="#8b5cf6" />
            <MetricBox icon="🕐" label="Experience"     value={`${p.years_of_experience}y`} color="#10b981" />
            <MetricBox icon="🔖" label="Recruiter Saves" value={sig.saved_by_recruiters_30d} color="#3b82f6" />
            <MetricBox icon="👁" label="Search Views"   value={sig.search_appearance_30d}   color="#f59e0b" />
          </div>

          {/* Score range bar */}
          <div style={{ marginBottom: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#64748b", marginBottom: 4 }}>
              <span>Match strength in shortlist</span>
              <span style={{ color: "#e2e8f0", fontWeight: 700 }}>{pct}%</span>
            </div>
            <div style={{ background: "#1e1e3a", borderRadius: 6, height: 8 }}>
              <div style={{ width: `${pct}%`, height: 8, borderRadius: 6, background: "linear-gradient(90deg,#6366f1,#a78bfa)", transition: "width .5s" }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#374151", marginTop: 2 }}>
              <span>{minScore.toFixed(3)}</span><span>{maxScore.toFixed(3)}</span>
            </div>
          </div>

          {/* Pills */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 16 }}>
            {pills.map((pill, i) => (
              <span key={i} style={{
                display: "inline-flex", alignItems: "center", gap: 5,
                background: pill.bg, color: pill.fg,
                border: `1px solid ${pill.fg}33`,
                padding: "3px 10px", borderRadius: 20, fontSize: 12, fontWeight: 600,
              }}>{pill.icon} {pill.text}</span>
            ))}
          </div>

          {/* Headline quote */}
          {p.headline && (
            <div style={{
              fontSize: 13, color: "#64748b", fontStyle: "italic",
              padding: "8px 14px", background: "#111128",
              borderLeft: "3px solid #6366f1", borderRadius: "0 6px 6px 0", marginBottom: 16,
            }}>
              &ldquo;{p.headline}&rdquo;
            </div>
          )}

          <hr style={{ border: "none", borderTop: "1px solid #1e1e3a", margin: "4px 0 16px" }} />

          {/* Two-column body */}
          <div style={{ display: "grid", gridTemplateColumns: "55fr 45fr", gap: 24 }}>

            {/* LEFT — analysis + career */}
            <div>
              <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "#4a4a6a", marginBottom: 10 }}>
                🤖 AI Analysis <span style={{ color: "#6366f1", fontWeight: 500, textTransform: "none", letterSpacing: 0, fontSize: 10 }}>· Groq llama-3.3-70b</span>
              </div>
              <div>
                {c.analysis.split("\n").filter(l => l.trim()).map((line, i) => (
                  <div key={i} style={{ display: "flex", gap: 8, marginBottom: 8 }}>
                    <span style={{ color: "#6366f1", flexShrink: 0, marginTop: 2, fontSize: 10 }}>›</span>
                    <span style={{ fontSize: 13, color: "#c4c4e0", lineHeight: 1.6 }}>
                      {line.replace(/^[-•*]\s*/, "")}
                    </span>
                  </div>
                ))}
              </div>

              <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "#4a4a6a", margin: "18px 0 10px" }}>
                💼 Career History
              </div>
              {c.career_history.slice(0, 5).map((job, i) => (
                <div key={i} style={{ display: "flex", gap: 14, marginBottom: 16, position: "relative" }}>
                  <div style={{ flexShrink: 0, width: 16, position: "relative" }}>
                    <div style={{
                      width: 16, height: 16, borderRadius: "50%", marginTop: 2,
                      background: job.is_current ? "#10b981" : "#374151",
                      border: `2px solid ${job.is_current ? "#10b981" : "#2d2d4a"}`,
                    }} />
                    {i < Math.min(c.career_history.length, 5) - 1 && (
                      <div style={{ position: "absolute", left: 7, top: 18, bottom: -4, width: 2, background: "#1e1e3a" }} />
                    )}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: job.is_current ? "#e2e8f0" : "#c4c4e0", display: "flex", alignItems: "center", gap: 8 }}>
                      {job.title}
                      {job.is_current && (
                        <span style={{ background: "rgba(16,185,129,.15)", color: "#34d399", fontSize: 10, padding: "1px 7px", borderRadius: 20, border: "1px solid #34d39944" }}>
                          current
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 12, color: job.is_current ? "#6366f1" : "#94a3b8", margin: "2px 0" }}>
                      {job.company} · {job.industry} · {job.duration_months}mo
                    </div>
                    <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.5 }}>
                      {(job.description ?? "").slice(0, 220)}{(job.description ?? "").length > 220 ? "…" : ""}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* RIGHT — assessments + skills + signals */}
            <div>
              {assessments.length > 0 && (
                <>
                  <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "#4a4a6a", marginBottom: 10 }}>
                    ✅ Verified Assessments
                  </div>
                  {assessments.map(([skill, val]) => (
                    <AssessmentBar key={skill} skill={skill} score={val} />
                  ))}
                  <div style={{ height: 10 }} />
                </>
              )}

              <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "#4a4a6a", marginBottom: 10 }}>
                🔧 Skills
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 16 }}>
                {topSkills.map(sk => {
                  const [bg, fg] = PROF_COLORS[sk.proficiency] ?? PROF_COLORS.beginner;
                  return (
                    <span key={sk.name} title={`${sk.duration_months}mo · ${sk.endorsements} endorsements`} style={{
                      background: bg, color: fg,
                      padding: "4px 10px", borderRadius: 20,
                      fontSize: 12, fontWeight: 500, whiteSpace: "nowrap",
                    }}>
                      {sk.name} <span style={{ opacity: 0.6, fontSize: 10 }}>{sk.endorsements}★</span>
                    </span>
                  );
                })}
              </div>

              <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "#4a4a6a", marginBottom: 10 }}>
                📊 Platform &amp; Engagement
              </div>
              {[
                ["Response rate", `${Math.round(sig.recruiter_response_rate * 100)}%`],
                ["GitHub score",  `${sig.github_activity_score?.toFixed(1)} / 100`],
                ["Last active",   sig.last_active_date ?? "—"],
                ["Work mode",     sig.preferred_work_mode ?? "—"],
              ].map(([label, val]) => (
                <div key={label} style={{
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "6px 0", borderBottom: "1px solid #1a1a30", fontSize: 12,
                }}>
                  <span style={{ color: "#64748b" }}>{label}</span>
                  <span style={{ color: "#e2e8f0", fontWeight: 600 }}>{val}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
