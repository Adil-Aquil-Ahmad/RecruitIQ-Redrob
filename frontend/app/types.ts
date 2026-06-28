export interface CareerEntry {
  company: string;
  title: string;
  industry: string;
  duration_months: number;
  is_current: boolean;
  description: string;
  company_size?: string;
}

export interface Skill {
  name: string;
  proficiency: "expert" | "advanced" | "intermediate" | "beginner";
  endorsements: number;
  duration_months: number;
}

export interface Education {
  institution: string;
  degree: string;
  field_of_study: string;
  start_year: number;
  end_year: number;
  tier: string;
}

export interface Profile {
  anonymized_name: string;
  headline: string;
  summary: string;
  location: string;
  country: string;
  years_of_experience: number;
  current_title: string;
  current_company: string;
  current_industry: string;
  current_company_size?: string;
}

export interface Signals {
  saved_by_recruiters_30d: number;
  search_appearance_30d: number;
  recruiter_response_rate: number;
  github_activity_score: number;
  open_to_work_flag: boolean;
  last_active_date: string;
  notice_period_days: number;
  preferred_work_mode: string;
  skill_assessment_scores: Record<string, number>;
}

export interface Candidate {
  candidate_id: string;
  rank: number;
  score: number;
  reasoning: string;
  analysis: string;
  profile: Profile;
  career_history: CareerEntry[];
  skills: Skill[];
  education: Education[];
  redrob_signals: Signals;
}
