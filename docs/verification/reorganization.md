# Repository migration verification

Verified 5 November 2026 after reorganizing the working project.

- `make test`: 108 Python tests and 12 frontend tests pass. Four existing upstream Starlette/Typer deprecation warnings remain. No tests were skipped or removed to make the migration pass.
- `make check`: Ruff passes; TypeScript checks and the Vite production build pass from `frontend/`.
- `pip check`: no broken requirements.
- `make audit` and `make prepare`: the actual NASA file produces an audit and 1,027 windows with 36 features under `data/interim/nasa/` and `data/processed/nasa/`.
- `make predict` and `make explain`: the existing real feature example completes both workflows. Predicted wear remains exactly `0.2428964078426361` mm.
- A before/after SHA-256 inventory verifies all 492 model payloads, all four PDFs (including the dataset's PDF) and the MATLAB dataset without any mismatch. See [integrity summary](reorganization-integrity.json).
- All nine default main-study models reload from their new artifact locations and reproduce all 4,905 saved validation/test predictions exactly, with maximum absolute difference zero. Historical model manifests are unchanged. See [parity evidence](reorganization-prediction-parity.json).
- The relocated API discovers five complete studies, defaults to `nasa-final`, excludes the superseded `nasa-study`, and serves actual bounded prediction/SHAP.
- Browser smoke check on the restarted services: the React overview retains its layout and loads recorded studies; Wear Estimate uses the saved FedAvg model and returns `0.3224 mm` for `case_01_run_001`, window 0, with measured wear and signed SHAP contributions. Reconstruction error is `3.0e-8`.
- Filesystem checks find one project-authored README and no TypeScript, TSX, HTML or NPM manifests outside `frontend/`. Installed third-party dependencies keep their own package documentation.

Raw command output: [tests](reorganization-tests.txt), [lint and build](reorganization-build.txt). Original research-development logs remain historical records. The current folder map and commands are documented only in the [root README](../../README.md).
