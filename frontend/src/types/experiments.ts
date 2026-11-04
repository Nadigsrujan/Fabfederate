export interface ExperimentOption { id: string; status: string; scenarios: string[]; seeds: number[]; default: boolean }
export interface MetricRow {
  method: string; seed: number; scenario: string; split: string; mae: number; rmse: number;
  group_mae: number; worst_client_mae: number | null; count: number; groups: number; units: string; mu: number;
}
export interface RoundRow {
  method: string; seed: number; scenario: string; round: number; train_loss: number;
  validation_mae: number; duration_seconds: number; upload_bytes: number; download_bytes: number; mu: number;
}
export interface ModelInfo { id: string; kind: string; feature_names: string[]; wear_unit: string; selected_round: number | null; validation_mae: number | null }
export interface ClientInfo { id: string; training_groups: number; training_runs: number; training_windows: number }
export interface Study {
  id: string; status: string;
  dataset: { name: string; run_count: number; labeled_runs: number; unlabeled_runs: number; case_count: number;
    feature_count: number; window_count: number; wear_unit: string;
    channels: { name: string; samples: number; nonfinite_samples: number }[];
    splits: { split: string; windows: number; groups: number }[] };
  scenarios: string[]; seeds: number[]; models: ModelInfo[]; metrics: MetricRow[]; rounds: RoundRow[]; clients: ClientInfo[];
  provenance: { source_sha256: string; code_sha256: string; completed_at: string };
  default_scenario: string; default_seed: number;
}
export interface Sample { id: string; run_id: string; group_id: string; window_id: string | number; split: string; wear_target: number; wear_unit: string }
export interface Prediction {
  model: string; sample: { run_id: string; group_id: string; window_id: string | number; split: string } | null;
  predicted_wear: number; measured_wear: number | null; wear_unit: string; features: { name: string; value: number }[];
  explanation: { feature_names: string[]; values: number[]; base_value: number; prediction: number; reconstruction_error: number; runtime_seconds: number } | null;
  limitations: string[];
}
export type Route = "overview" | "experiments" | "clients" | "dataset" | "estimate";
