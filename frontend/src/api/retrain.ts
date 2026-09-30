import type { Row } from "../types/common";
import { api } from "./client";

export type RetrainStatus = {
  progress: { total: number; labeled: number; unlabeled: number };
  available_to_reveal: number;
  can_retrain: boolean;
  reason: string;
  min_training_rows: number;
  min_positives: number;
  pending: Row[];
  models: Row[];
};

export type RevealResult = { n_revealed: number; n_remaining: number; n_default: number };

export type RetrainResult = {
  model_name: string;
  version: number;
  n_train: number;
  n_valid: number;
  n_positives: number;
  auc_train: number;
  auc_valid: number;
  ks_valid: number;
  fit_seconds: number;
  artifact_path: string;
  feature_set_version: string;
  threshold: number;
  overfit_warning: boolean;
};

export type ModelRef = { id: number; model_name: string; version: number; role: string };

export const getRetrainStatus = () => api.get<RetrainStatus>("/api/retrain/status");
export const revealLabels = (n: number) => api.post<RevealResult>("/api/retrain/reveal-labels", { n });
export const setLabel = (prediction_id: number, actual: 0 | 1) =>
  api.post<{ prediction_id: number; actual: number }>("/api/retrain/labels", { prediction_id, actual });
export const runRetrain = (activate: boolean) => api.post<RetrainResult>("/api/retrain/run", { activate });
export const promote = (model_version_id: number) =>
  api.post<ModelRef>("/api/retrain/promote", { model_version_id });
export const registerChampion = () => api.post<ModelRef>("/api/retrain/register-champion");
