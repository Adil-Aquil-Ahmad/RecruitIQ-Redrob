import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "RecruitIQ · India Runs Challenge",
  description: "AI-powered candidate ranking for Senior AI Engineer at Redrob AI. Track 01 — Intelligent Candidate Discovery.",
  openGraph: {
    title: "RecruitIQ · India Runs Challenge",
    description: "Top 100 AI Engineer candidates ranked by a 3-stage ML pipeline.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
