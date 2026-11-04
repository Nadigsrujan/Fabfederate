# Models and federated implementation report

Historical development record from the original layout. Current locations and commands are in the [root README](../../README.md).

## Public interface

- `models.train.run_baselines(data, output_dir, config)` returns `mean`, `median`, `random_forest`, `centralized_mlp`, and `local_<client_id>` bundles. Each saves into `output_dir/<method>/`.
- `federated.server.run_federated(data, output_dir, config, algorithm='fedavg', mu=0.0)` returns one selected bundle and saves `output_dir/<algorithm>/`. Use a separate output root per FedProx candidate to retain all candidates.
- `models.bundle.load_bundle(path)` accepts a method directory. `ModelBundle` exposes `feature_names`, `wear_unit`, `model_kind`, `model`, `imputer`, `scaler`, and `metadata`; `predict(frame)` consumes raw named feature columns, `transform(frame)` returns prepared features, and `predict_array(array)` accepts raw features in `feature_names` order for SHAP.
- Prediction rejects missing/duplicate/unknown columns, infinities, nonnumeric features, mismatched declared wear units, and unsupported explicit `frame.attrs['schema_version']`. Known metadata columns may accompany the predictors but are discarded. NaNs use the frozen median imputer. Empty prediction batches return empty arrays.
- Bundles contain `bundle.joblib`, hash/version/schema manifest, and `model_card.md`; MLP baselines additionally save `epochs.csv`. Joblib is executable Python serialization: load only trusted project artifacts. SHA256 detects artifact damage/tampering against the manifest; it is not a cryptographic publisher signature.

## Scientific behavior

Preprocessing is fitted exclusively on pooled train rows. All baseline and federated entry points deterministically fit the same median imputer and standard scaler on this identical partition, then freeze it. Metadata explicitly discloses that pooled public-training statistics are shared; this is a benchmark convenience, not a private deployment protocol. Entirely missing training features are rejected. Train and validation targets must be finite; test feature/target values are never read by training.

The neural architecture is the same CPU PyTorch MLP for central, local, FedAvg and FedProx comparisons. Defaults are `[64,32]`, ReLU, dropout `.1`, Adam `.001`, batch size 32, MSE, central/local epochs 30, FL local epochs 3 and rounds 30; configuration overrides these defaults. Huber is supported. Random Forest uses the same preprocessing; scaling preserves its split ordering. Every local estimator's fit sees only its assigned training rows. Central/local checkpoints use minimum common heldout validation MAE, with that common validation scope disclosed. FL selects among trained rounds only; untrained round-zero validation is observed but excluded from checkpoint selection. Test evaluation belongs to the integration layer.

## Real Flower coordination and payloads

`flwr.server.Server.fit` runs the full round loop, obtains `FitIns` through the actual `FedAvg.configure_fit`, calls actual `ClientProxy` adapters around `NumPyClient.to_client()`, receives serialized Flower parameters and `FitRes`, and performs actual `FedAvg.aggregate_fit` weighted averaging. Our strategy validates updates before calling Flower aggregation. It is not a custom averaging loop relabeled as Flower. Logical clients run in process, in fixed order with one worker, avoiding Ray and global Torch RNG races. Incoming global weights are cloned and detached before FedProx local training. Fresh Adam is created each round. `mu=0` is prediction-identical to matched FedAvg in tests.

Clients return parameters, actual unique training-example count (not multiplied by epochs), and scalar metrics only. The strategy rejects incomplete rounds, duplicated clients, mismatched counts or versions, nonfinite losses/parameters, and mismatched parameter shapes/dtypes. The local data arrays remain on client objects and are not in fit instructions or responses. The coordinator owns a separate heldout validation callback. This architecture provides logical separation inside one benchmark process, not operating-system process isolation, secure aggregation, or privacy guarantees.

Each method writes `rounds.csv` with round, weighted local train loss, validation MAE, measured wall duration, upload/download bytes, client count, and mu. Byte counts measure the serialized NumPy parameter tensors including NumPy headers, excluding scalar/configuration and transport envelopes. Thus the recorded value is an explicit parameter payload estimate, not physical network bandwidth. `payload_manifest.json` records allowed/excluded fields and actual per-client example counts without embedding raw data. Selected bundles record seed, configuration, feature/split identities, architecture, train group/example counts, preprocessing policy, validation selection, Flower version, budgets and provenance from `PreparedData.metadata`.

Installed interfaces inspected: Flower 1.42.0 `Server`, `ClientProxy`, `NumPyClient`, and `FedAvg`. Official references: [Server](https://flower.ai/docs/framework/ref-api/flwr.server.Server.html), [NumPyClient](https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html). Future Flower major versions may require a transport adapter migration; project dependency restricts Flower to `<2`.

## Verification

Tests were authored before implementation; initial collection failed because the proposed modules did not exist. Focused tests exercise known weighted vectors, invalid counts/shapes/dtypes/nonfinite updates, positive and zero proximal penalties, train-only imputation/scaling, independent local fit counts, bundle reload prediction parity, schema/unit/infinite-value rejection, artifact hash failure, and test-row independence. The integration tests run real three-client Flower rounds on explicitly synthetic `PreparedData`, verifying complete round logs, payload counts, saved reload parity, and FedProx mu-zero equivalence. An additional positive-mu Flower test verifies training remains unchanged when heldout test values are destroyed.

Run: `.venv/bin/python -m pytest tests/test_models.py tests/test_federated.py -q`; `.venv/bin/ruff check src/fabfederate/models src/fabfederate/federated tests/test_models.py tests/test_federated.py`.

## Limits and integration ownership

These tests prove mechanics on synthetic fixtures, not a wear-estimation research finding. Dataset quality, group leakage auditing, feasible heterogeneity construction, multi-seed/mu selection, final heldout evaluation, SHAP and dashboard are handled by the other project modules. No empirical superiority claim is encoded in the trainers. CPU logical clients and full participation are intentional current constraints. Client identifiers used for local output directories must contain letters, digits, underscores or hyphens.
