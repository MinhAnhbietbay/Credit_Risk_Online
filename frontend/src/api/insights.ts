import type { Figure, Row } from "../types/common";
import type { Counts } from "./catalog";
import { api } from "./client";

export type FeatureInsight = {
  feature: string;
  mo_ta: string;
  buckets: Row[];
  chart: Figure | null;
  sentence: string | null;
  monotonic: boolean | null;
};

export type DefaultRates = {
  counts: Counts;
  available_features: string[];
  picked: string[];
  features: FeatureInsight[];
};

export type ModelComparison = { model_comparison: Row[]; shap_ranking: Row[]; shap_note: string };

// null = để backend chọn mặc định; [] = người dùng bỏ chọn hết.
export const getDefaultRates = (picked: string[] | null) =>
  api.get<DefaultRates>(
    picked === null
      ? "/api/insights/default-rates"
      : `/api/insights/default-rates?features=${encodeURIComponent(picked.join(","))}`,
  );

export const getModelComparison = () => api.get<ModelComparison>("/api/insights/model-comparison");
