# Data work package report

Historical development record from the original layout. Current locations and commands are in the [root README](../../README.md).

Implemented the owned data, feature and simulation modules without commits.

Exports:

- `data.nasa_loader.load_nasa(path) -> list[SignalRun]`: strict source-shaped
  MATLAB loader, six verified channels, original case/run IDs, missing labels,
  source hashes, measured-target timing, mm units and 250 Hz provenance.
- `data.audit.audit_runs(runs) -> dict`: source/group/label/channel inventory and
  traceable quality issues; finite JSON output.
- `features.statistics.extract_window_features(signals, spectral=False,
  sampling_hz=None)`: amplitude-preserving shared time/spectral definitions.
- `features.pipeline.prepare_dataset(runs, output_dir, config) -> PreparedData`:
  independent-group splits before run-only windows, explicit predictor schema,
  source/label provenance, rejection records, no fitted preprocessing, and
  Parquet/schema/audit/split/client artifacts.
- `simulation.splits.build_group_splits(...)`: deterministic whole-group splits
  with feasibility errors. Fractions refer to the complete group population.
- `simulation.clients.assign_clients(...) -> (frame, manifest)`: IID-ish,
  condition, quantity, label and few-shot allocations of complete training groups;
  exact documented held-out-domain ownership only when unique.

`seed` remains the backwards-compatible common seed. `split_seed` and
`partition_seed` override it independently. `few_shot_groups` is configurable.
Source units cannot be relabeled and confirmed sampling rates cannot be silently
overridden. Metadata fields are explicitly excluded from predictor names.

Source audit: 167 runs; 16 independent cases; 146 measured VB; 21 missing VB;
observed range 0–1.53 mm. All six channels contain 1,509,360 finite samples each;
run lengths are 9,000/15,360. The bundled readme pp. 1–3/7 confirms field meanings,
label timing, processed-data sampling frequency, and target units. The fuller
source card is `docs/dataset_cards/nasa_milling.md`.

Tests-first evidence:

- Initial RED: feature/split collection failed for missing implementation modules.
- Loader RED: data collection failed for missing audit module.
- Independent seed regression RED: partition seed changed but client assignments
  did not, demonstrating the old shared-seed-only behavior.
- GREEN after implementation/fix: `.venv/bin/python -m pytest tests/test_data.py
  tests/test_features.py tests/test_splits.py -q` — **22 passed**.
- Ruff on owned implementation and tests — **all checks passed**.
- Shared full-suite check at handoff: `.venv/bin/python -m pytest -q` —
  **66 passed, 4 third-party deprecation warnings**, 25.76 seconds.

Tests use explicit tiny synthetic source-shaped fixtures; they do not establish
NASA performance. They verify feature formulas, known-frequency extraction,
invalid/constant rejection, unknown labels, original source grouping, invalid
source identity, unit and sampling checks, frozen split seeds, five scenarios,
no group leakage, no held-out-label influence on ownership, and null ownership
for ambiguous/unseen domains.

Real-source smoke preparation at 1,024 samples/stride, seed 42: 1,027 rows,
36 predictors, 482 train / 226 validation / 319 test. There were 141 rejected
constant/numerically degenerate `smcDC` windows and 21 excluded unlabeled runs.
At 2,048 samples/stride: 528 rows, 56 constant-window rejections. All five
scenarios prepared successfully at this size. Generated verification artifacts
are under `/private/tmp/fabfederate-data-validation` and
`/private/tmp/fabfederate-scenario-validation`; the root integration runs create
the permanent experiment outputs.

Limitations: only 16 independent cases; physical cutter identities are not
invented; sensor physical calibration remains unresolved; repeated windows
share run-level measured labels. Cases 4/10 have a held-out condition absent
from the seed-42 training cases and retain null client IDs. PHM acquisition and
campus validation remain conditional; the optional explicit local PHM adapter
is described below. Source cards and per-run audits preserve these limitations.

## Follow-up: controlled limited labels and optional PHM adapter

The reviewed few-shot scenario now starts from the condition baseline, freezes
every held-out ownership association, then excludes complete training groups
from a single target client and from the training union of every method. Other
clients remain unchanged. Tests first failed on missing E4 manifest fields, then
passed with reduced training union and exact unchanged held-out rows. At seed42,
the real NASA 2,048-window scenario targets client_1, retains case02, excludes
cases03/05, and evaluates the frozen target test groups cases11/12/16. It has
174 training,119 validation,161 test windows, and an available E4 comparison.
Manifests include `few_shot_client`, `excluded_training_groups`, `e4_test_groups`,
`e4_status`, `e4_reason`, and retained-group IDs.

Optional `data.phm_loader.load_phm(manifest_path, *, root=None,
edge_reduction=None) -> list[SignalRun]` is implemented. `edge_reduction` must
explicitly be `max` or `mean`. The local JSON manifest maps actual seven-channel
files, original cutter/cut IDs and the three measured flute columns. No archive
or data is downloaded; no filename/order-based label mapping is guessed. The
official PHM2010 page was verified through Firecrawl and confirms channels,
50kHz sampling,10^-3mm source wear, released c1/c4/c6 training labels, and the
incorrect c3 wear-file warning. Test-cutter labels are rejected. Output wear
uses `um`, with every measured edge and conversion provenance retained.

Four PHM tests first failed on the missing module. With the adapter and few-shot
fix implemented, the focused data/features/splits suite is **28 passed** and Ruff
is clean. The detailed source/admission protocol and CLI-ready manifest are in
`docs/dataset_cards/phm2010_milling.md`. Full PHM acquisition/evaluation remains
conditional, and three labeled cutters cannot support a three-client plus
whole-cutter train/validation/test benchmark under the default protocol.

Final shared-suite verification after these changes:
`.venv/bin/python -m pytest -q` — **85 passed, 2 third-party deprecation
warnings**, 34.54 seconds. Owned modules/tests also pass Ruff.
