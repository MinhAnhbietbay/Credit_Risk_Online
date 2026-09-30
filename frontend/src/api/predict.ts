import type { Figure } from "../types/common";
import type { ActiveModel } from "./catalog";
import { api } from "./client";

export type NumericField = {
  feature: string;
  mo_ta: string;
  categorical: false;
  min: number;
  max: number;
  median: number;
  step: number;
};

export type CategoricalField = {
  feature: string;
  mo_ta: string;
  categorical: true;
  options: { code: number; label: string }[];
};

export type FormField = NumericField | CategoricalField;

export type ApplicantOption = { sk_id_curr: number; label: string };

export type FormConfig = {
  fields: FormField[];
  n_features_total: number;
  applicants: ApplicantOption[];
  active_model: ActiveModel | null;
};

export type TopFeature = { feature: string; value: string; mo_ta: string };
export type ApplicantDetail = { sk_id_curr: number; top_features: TopFeature[] };

export type ShapFactor = {
  feature: string;
  value: string;
  direction: string;
  shap: number;
  mo_ta: string;
};

export type ShapResult = {
  note: string;
  probability: number;
  base_probability: number;
  factors: ShapFactor[];
};

export type PredictResult = {
  pd: number;
  threshold: number;
  flagged: boolean;
  risk_level: string;
  risk_color: string;
  decision: string;
  model_label: string;
  prediction_id: number;
  filled_from_median: { feature: string; mo_ta: string }[];
  n_features_total: number;
  gauge: Figure;
  shap: ShapResult | null;
  shap_error: string | null;
};

export const getFormConfig = () => api.get<FormConfig>("/api/predict/form-config");
export const getApplicant = (sk: number) => api.get<ApplicantDetail>(`/api/predict/applicants/${sk}`);
export const predictExisting = (sk: number) => api.post<PredictResult>("/api/predict", { sk_id_curr: sk });
export const predictManual = (values: Record<string, number>) =>
  api.post<PredictResult>("/api/predict", { values });
