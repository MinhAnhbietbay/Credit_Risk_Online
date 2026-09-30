import { api } from "./client";

export type ActiveModel = {
  id: number;
  model_name: string;
  version: number;
  role: string;
  threshold: number | null;
  feature_set_version: string | null;
  auc: number | null;
  note: string;
};

export type Counts = {
  applicants: number;
  labeled: number;
  predictions: number;
  model_versions: number;
};

export type Status = {
  db_ready: boolean;
  message: string;
  active_model: ActiveModel | null;
  counts: Counts | null;
  feature_set_version: string;
  n_features: number;
};

export const getStatus = () => api.get<Status>("/api/status");
