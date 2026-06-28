import { promises as fs } from "fs";
import path from "path";
import type { Candidate } from "./types";
import RankingDashboard from "./components/RankingDashboard";

export default async function Home() {
  const filePath = path.join(process.cwd(), "public", "candidates_data.json");
  const raw = await fs.readFile(filePath, "utf-8");
  const candidates: Candidate[] = JSON.parse(raw);

  const maxScore = candidates[0]?.score ?? 1;
  const minScore = candidates[candidates.length - 1]?.score ?? 0;

  return <RankingDashboard candidates={candidates} maxScore={maxScore} minScore={minScore} />;
}
