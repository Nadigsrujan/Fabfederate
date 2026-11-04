# Execution ledger: implementation.md

Spec: ../../implementation.md. The user authorized implementing the complete plan on 4 November 2026.

## Rulings

- Work in the supplied fresh directory, which has no Git repository or existing application. Preserve all source PDFs and the report. Do not move the user's files into a worktree.
- Use Python 3.12 in a project-local `.venv`; the host's default Python 3.9 is below the spec's floor.
- Execute independent modules in parallel with disjoint ownership and explicit contracts; integrate and review centrally.
- All real-data claims require downloaded source inspection and generated measurements. Synthetic fixtures are permitted only for tests and explicitly labeled demonstrations.
- Optional datasets and physical campus work remain conditional. Implement supported offline adapters/protocols without inventing access, labels, or safety approvals.

## Integration contracts

- Shared records are defined in `src/fabfederate/contracts.py`.
- Data pipeline emits a pandas feature table with metadata columns `dataset_id`, `run_id`, `group_id`, `window_id`, `wear_target`, `wear_unit`, `split`, `client_id`, `condition`; numeric predictor columns are explicitly listed.
- `PreparedData` carries `frame`, `feature_names`, `metadata`; train/validation/test rows share a frozen feature schema.
- Model bundle stores model kind/state, preprocessing state, feature names, wear unit, metadata, and version. Model API owns save/load/predict.
- Experiment directory contains `manifest.json`, `metrics.csv`, `predictions.csv`, `rounds.csv`, `features.parquet`, `models/`, `figures/`, and optional `explanations/`.
- Coordinator round payloads contain model arrays, counts, and scalar metadata only; benchmark preparation and held-out evaluation are explicitly separate access roles.

## Work packages

- [ ] Foundation/configuration/CLI/integration and real-data acquisition (root).
- [ ] Dataset adapters, audit, features, group splits, client scenarios, tests (data agent).
- [ ] Models, model bundles, FedAvg/FedProx, Flower integration, tests (model agent).
- [ ] Evaluation, SHAP, Streamlit dashboard, tests (interface agent).
- [ ] Real NASA end-to-end experiment, repeated seeds/scenarios, reports and verification.
- [ ] Independent review, corrections, final evidence and task-status reconciliation.
