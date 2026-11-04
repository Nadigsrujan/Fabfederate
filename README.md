# FabFederate-CNC

An offline research application for measured CNC milling-tool-wear regression, with case-disjoint data preparation, centralized/local baselines, genuine Flower FedAvg/FedProx training, model-specific SHAP, and a Streamlit dashboard.

The governing specification is [implementation.md](implementation.md). The revised campus/public-dataset architecture takes precedence over the older industry deployment proposal.

## Open the application

The project environment and NASA data have been installed in this workspace. From the project directory:

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open [the local dashboard](http://127.0.0.1:8501). Select **nasa-final** for the reviewed, complete study. Other artifact sets include a smoke run, an efficiency experiment, and a spectral-feature ablation. `nasa-study` is historical and superseded; its few-shot design predates review corrections.

The five screens show dataset coverage, experiment comparisons, client distributions, convergence/cost, and saved-model prediction/explanation. Wear Estimate accepts a recorded held-out feature row or a strict feature CSV; the UI provides a matching column header. Model files are loaded only from trusted project-created artifact folders.

## Fresh installation

Python 3.12 on macOS/Linux is the tested environment. The default macOS system Python 3.9 is too old. On another machine with Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m fabfederate download
```

The lockfile contains the actual tested versions, including the WebSocket compatibility constraint needed by the installed Streamlit/Flower stack. No GPU, CNC machine, account, cloud service, or network connection is needed for experiments after installation and data acquisition. Flower's core `Server.fit` and `FedAvg` strategy run logical clients through an in-process transport; the larger Ray simulation runtime is optional and is not used by the verified default path.

## Reproduce experiments

```bash
# Audit the actual downloaded source and write a machine-readable inventory.
.venv/bin/python -m fabfederate audit --output data/interim/audit

# Prepare case-disjoint windows, features, and deterministic client manifests.
.venv/bin/python -m fabfederate prepare --output data/processed/nasa

# Small genuine-data smoke experiment.
.venv/bin/python -m fabfederate study --config configs/smoke.yaml --output experiments/my-smoke

# Complete study: five scenarios, three training seeds, 30 rounds,
# three local epochs, and validation selection over four FedProx mu values.
.venv/bin/python -m fabfederate study --config configs/default.yaml --output experiments/my-study

# Independent, explicitly budgeted extensions.
.venv/bin/python -m fabfederate study --config configs/efficiency.yaml --output experiments/my-efficiency
.venv/bin/python -m fabfederate study --config configs/spectral.yaml --output experiments/my-spectral

# Regenerate plots and report from stored predictions/metrics, without retraining.
.venv/bin/python -m fabfederate report experiments/my-study
```

Study output directories are never silently overwritten. Use a distinct output path for reproduction. On macOS, Matplotlib's first font discovery may take time; subsequent runs use its cache. For a controlled writable cache, prefix a command with `MPLCONFIGDIR=/tmp/fabfederate-matplotlib`.

Default supervised models are mean, median, Random Forest, centralized MLP, one local MLP per client, FedAvg MLP, and validation-selected FedProx MLP. All neural methods share the declared architecture and train-only pooled-public-data preprocessing. Checkpoints and `mu` are selected using validation only; saved final test predictions are generated after selection freezes. Local models have isolated training rows but share the common public validation/preprocessing policy, which is explicitly recorded.

Few-shot uses the condition-partition evaluation mapping, retains a small number of one client's complete training cases, and excludes that client's withheld cases from **every** method's eligible training union. Its held-out domains remain unchanged. `limited_label_comparison.csv` measures local/federated error on the low-data client's actual supported test groups.

## Artifacts and provenance

Each study writes:

- `manifest.json`: source archive/file digest, executed-source file hashes, exact configuration, library versions, seeds, timing, and status.
- `prepared/<scenario>/`: feature table, schema, audit, fixed split/client manifests, exclusions, and supported held-out domain ownership.
- `runs/<scenario>-seed-<seed>/`: model bundles, validation-selected candidates, selection freeze, per-model evaluation, and SHAP examples.
- `metrics.csv`, `predictions.csv`, `rounds.csv`: full measured outputs and payload accounting.
- `robustness.csv`: deterministic feature-noise stress and missing-input rejection.
- `limited_label_comparison.csv`, `explanation_stability.csv`, `attributions.json`: limited-label and cross-seed attribution evidence.
- `models/`, `features.parquet`: one explicitly identified scenario/seed for dashboard inference.
- `report.md`, `figures/`, `overview_predictions.csv`: generated scientific report and traceable plotted source identities.

The source snapshot is checked again before successful completion; editing executable code during a run invalidates that run. All raw data, derived tables, and model checkpoints are Git-ignored. Retrieval instructions and source cards are committed instead of redistributing archives.

## File-driven inference

Export only the ordered feature columns listed in the selected model manifest into `sample.csv`. Then:

```bash
.venv/bin/python -m fabfederate predict \
  --model experiments/nasa-final/models/fedavg \
  --input sample.csv --output outputs/predictions.csv

.venv/bin/python -m fabfederate explain \
  --model experiments/nasa-final/models/fedavg \
  --input sample.csv --background experiments/nasa-final/features.parquet \
  --output outputs/explanation.json
```

The background table must contain a documented `train` split. The CLI filters it to training rows. SHAP calls the actual selected model together with its saved transform; feature contributions reconstruct that predictor's output within a checked tolerance. SHAP values describe model influence, not physical causation or calibrated uncertainty.

## Tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests app scripts
.venv/bin/python -m pip check
```

Tests use tiny explicitly synthetic/source-shaped fixtures to check mathematical and interface behavior. Research results come from the downloaded NASA archive, never those fixtures. The suite covers label mapping, windows, group isolation, train-only preprocessing, scenario construction, weighted aggregation, proximal loss, actual Flower rounds, reload parity, prediction validation, SHAP reconstruction, report identity, and all five Streamlit views. Verification logs are under `docs/verification/`.

## Additional data and campus validation

[NASA source card](docs/dataset_cards/nasa_milling.md) documents the actual archive, measured labels, six channels, 250 Hz processed signals, and wear in millimetres. Its 167 runs belong to 16 cases; 21 runs have no measured wear. Case-heldout performance is not a verified cross-physical-tool or cross-factory result.

[PHM source card and manifest](docs/dataset_cards/phm2010_milling.md) document the optional functional local CSV adapter. Units, cut IDs, three flute measurements, and explicit `mean`/`max` reduction are preserved. Released-label checks reject c2/c3/c5 challenge labels. Use `configs/phm2010.example.yaml` for audit. Three released labeled cutters cannot support the NASA protocol's three training clients plus separate held-out validation/test cutters. A separate defensible protocol and actual archive acquisition are required before claiming PHM benchmark results.

[Campus protocol](docs/campus_protocol.md) and `configs/campus_manifest.example.json` define the read-only extension. With real supervisor approval and compatible recorded features:

```bash
.venv/bin/python -m fabfederate campus-validate \
  --manifest /path/to/approved/campus_manifest.json \
  --model experiments/nasa-final/models/fedavg --output outputs/campus-validation
```

No campus recordings, approval, actual machine integration, or measured campus accuracy are claimed. Unlabeled exports produce compatibility/shift information only. CNNs, wavelet features, remote federation, and a separate web API remain optional extensions.

## Evidence boundaries

All clients share one computer. Pooled public preprocessing uses training statistics centrally; parameter updates can disclose information. This prototype provides neither independent data custody nor a formal privacy guarantee. Repeated training seeds do not add independent tools. No automatic machine control, universal replacement threshold, guaranteed FL improvement, cost saving, or industrial deployment is established.

Official implementation references are linked in [technical decisions](docs/technical_decisions.md). Review findings and their resolutions are retained in [the scientific review](docs/development/scientific-review.md).
