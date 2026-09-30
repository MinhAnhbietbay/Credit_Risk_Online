import type { Figure, Row } from "../types/common";
import { api } from "./client";

export type RiskSegments = {
  bands: Row[];
  total: number;
  n_labeled: number;
  pd_mean: number | null;
  band_chart: Figure | null;
  calibration_chart: Figure | null;
};

export const getRiskSegments = () => api.get<RiskSegments>("/api/risk-segments");
