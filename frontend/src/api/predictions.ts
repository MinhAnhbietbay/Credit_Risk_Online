import type { Figure, Row } from "../types/common";
import { api } from "./client";

export type PredictionsResponse = {
  rows: Row[];
  summary: { total: number; n_labeled: number; pd_mean: number | null; flag_rate: number | null };
  auc: number | null;
  auc_note: string | null;
  confusion: Figure | null;
  distribution: Figure | null;
};

export const getPredictions = (limit = 1000) =>
  api.get<PredictionsResponse>(`/api/predictions?limit=${limit}`);
