import type { Figure } from "../types/common";
import { api } from "./client";

export type Scenario = { name: string; channel: string; description: string; shock: string; source: string };

export type Scenarios = { warning: string; non_monotonic_note: string; doc: string; scenarios: Scenario[] };

export type StressRow = {
  scenario: string;
  channel: string;
  shock: string;
  pd_mean: number;
  flag_rate: number;
  flag_rate_delta: number;
};

export type StressRun = { n_sample: number; threshold: number; rows: StressRow[]; band_chart: Figure };

export const getScenarios = () => api.get<Scenarios>("/api/stress/scenarios");
export const runStress = (n: number) => api.post<StressRun>("/api/stress/run", { n });
