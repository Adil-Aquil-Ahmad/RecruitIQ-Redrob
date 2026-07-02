"use client";

interface Props {
  label: string;
  value: string | number;
  color?: string;
  icon: string;
}

export default function MetricBox({ label, value, color = "#6366f1", icon }: Props) {
  return (
    <div style={{
      background: "#0d0d1f",
      border: "1px solid #1e1e3a",
      borderRadius: 10,
      padding: "12px 10px",
      textAlign: "center",
    }}>
      <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 6 }}>
        {icon} {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
    </div>
  );
}
