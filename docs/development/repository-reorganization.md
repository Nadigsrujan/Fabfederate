# Repository reorganization

Goal: make the existing implementation understandable using the user's supplied modular Python monorepo as the reference. Keep one project README, preserve every supplied PDF, dataset and trained model, and retain both working interfaces.

Decisions: Python remains an installable package under `src/fabfederate/`. The local backend is `api/app.py`; data loading, feature extraction and partitioning live together under `data/`; full training orchestration lives under `pipelines/`; SHAP lives under `explainability/`; campus validation lives under `campus_cnc/`. Keep `models/bundle.py` because existing trusted joblib artifacts serialize that module path. Add a real `inference/pipeline.py` for file prediction/explanation, rather than an empty folder.

All React source, components, styles, dependency manifests, build output and tooling belong under `frontend/`. Rename the decorative canvas component to `research-background.tsx` and remove the standalone generic demo. Retain the optional Python dashboard under `dashboard/app.py`, with its configuration next to it.

Generated, immutable study directories move from `experiments/` to `artifacts/experiments/`; inference examples move from `outputs/` to `artifacts/inference/`. Raw NASA files stay at their current paths under `data/raw/nasa/` so the dataset PDF is not moved. No empty CNN, cloud, database or device-control modules are introduced.

## Execution

- [x] Record PDF, dataset and model hashes before moving files.
- [x] Move the complete React project into `frontend/`; update imports, aliases, test paths and CI working directory; verify build/tests.
- [x] Reorganize Python modules and the optional dashboard; migrate every active import and filesystem default; verify all Python tests and lint.
- [x] Extract file inference from the CLI into `inference/pipeline.py` with equivalent validation; verify prediction/explanation using real saved models.
- [x] Move generated studies/examples; verify unchanged content and API discovery from the new default location.
- [x] Consolidate project documentation into one root README plus focused named reports. Move `implementation.md` under `docs/`; update references and record that historical artifact manifests retain their original executed-source paths.
- [x] Add a Makefile, a nonsecret environment template and a conventional requirements entry point. Run the documented commands, structure checks and final browser smoke check.

Evidence: [reorganization verification](../verification/reorganization.md).

## Verification focus

Existing model deserialization must retain compatibility; moving API/study modules must not change repository-root detection; frontend aliases and CSS scanning must resolve within the new frontend root; generated models must not disappear from the selector; historical hashes and every PDF must remain unchanged. Use existing regression tests and real saved-model reload/prediction, plus a read-only file inventory, to verify these boundaries.
