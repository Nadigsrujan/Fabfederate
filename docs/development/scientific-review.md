# Independent scientific correctness review

Reviewed 4 November 2026, before final remediation. Scope: `implementation.md`
sections 7–10 and 14, the data/features/simulation/model/federation/workflow paths,
and genuine NASA smoke artifacts. This is a review of the inspected implementation,
not a claim that the ongoing full study has finished.

## Findings

### P2 — Few-shot scenario does not preserve the intended limited-label comparison

Location: `src/fabfederate/simulation/clients.py:75–77` and `:85–95`.

The few-shot branch redistributes every training case, then derives new held-out
ownership from that redistribution. It does not restrict one existing client's
labeled groups while preserving that client's predefined evaluation set, as
required by section 8. All original training data still enter the pooled/FL
models. This is a valid quantity-skew stress test, but it cannot establish the
specified controlled limited-label E4 result.

Actual NASA reproduction using `experiments/nasa-smoke/features.parquet` and
`assign_clients(..., seed=42)`:

- IID client 0 owns cases 06, 13 and 15; few-shot client 0 owns only case 06,
  while the other cases are reassigned to other training clients.
- Held-out client owners change for cases 07, 08 and 11.
- The few-shot client 0 has **zero supported test groups**, so a test result for
  the low-data client cannot be obtained from this partition. Overall pooled
  test metrics are not that result.

Smallest correct remedy: explicitly persist E4 infeasibility when the low-data
client has no supported test coverage; keep pooled stress results separately
labeled. To implement the controlled experiment, freeze a defensible baseline
domain partition and its held-out ownership, retain N whole training groups for
client 0, exclude its discarded groups from every method's eligible union, and
retain all other clients' groups. Compare local client 0 and FL on the same
predefined client-0 test groups. No test targets or new ownership heuristic are
needed. Avoid reporting E4 complete based solely on pooled scenario completion.

### P2 — Measured-versus-predicted plot merges different recorded runs

Location: `src/fabfederate/evaluation/report.py:119–120`.

Plot aggregation groups by method, case, window index and wear, omitting
`run_id`. Window indices restart for every run, and two runs can have the same
measured wear. Those separate inputs are then averaged together, despite the
caption claiming only repeated-seed predictions for the same saved window are
averaged. This conceals actual prediction dispersion and individual cases.

Actual reproduction in the NASA smoke FedAvg test rows: 161 distinct saved
windows become 157 plotted points. `case_11`, VB 0.07 mm, window indices 0–3
each combine two different runs. Include `dataset_id` and `run_id` in the point
identity (plus existing group/window keys), averaging only across seeds. The
underlying per-row CSV and numerical error metrics are unaffected.

### P2 — Run artifacts do not identify the executed source implementation

Location: `src/fabfederate/workflow.py:84–90` and
`src/fabfederate/features/pipeline.py` metadata construction.

The study records dataset/config/split/partition hashes and library versions,
but no source-code digest or immutable revision. Feature metadata has a fixed
schema version and a hash of column names, which do not distinguish changed
feature formulas, training behavior, or orchestration. Two studies executed
with different code can therefore carry the same provenance identifiers. This
is especially relevant in this fresh, unversioned workspace and prevents the
specified code/config/data reproducibility claim (sections 7.7 and 12.1).

Record an ordered source-file SHA-256 manifest and aggregate code digest at
study start, or a verifiable immutable revision plus dirty-tree information;
record the Python/runtime platform for interpreting timing. Preserve that
identity with study/model metadata. Existing runs must be labeled with their
actual captured source identity, not retrospectively attributed to code edited
after those runs began.

## Verified strengths and limits

- Case splits occur before run-only windowing; all windows from a case retain
  one split. Metadata and labels are excluded from predictor columns.
- NASA missing labels remain missing. Each accepted window repeats its own
  run's recorded VB without interpolation; the physical-tool identity limitation
  is disclosed rather than replaced with invented cutter IDs.
- The common imputer/scaler fit training rows only. The pooled-public-statistics
  policy is disclosed and shared consistently across neural comparisons.
- Model fitting/checkpoint selection use training/validation values only.
  FedProx mu selection uses validation MAE before test predictions are produced.
- FedAvg weights use eligible example counts, not epoch-multiplied counts.
  FedProx uses the correct `mu/2` squared-distance term against detached incoming
  weights; zero-mu equivalence is covered by the real Flower test.
- Held-out ownership uses condition metadata, not held-out labels; ambiguous
  domains remain unassigned and pooled/group results remain available.
- Local models use common validation data, which is explicitly disclosed as a
  benchmark policy rather than a private per-site validation protocol.

Verification command:

```sh
.venv/bin/pytest tests/test_data.py tests/test_features.py tests/test_splits.py tests/test_models.py tests/test_federated.py tests/test_evaluation.py -q
```

Result: **49 passed**, two upstream Typer/Click deprecation warnings, 41.98 s.
Additional read-only reproductions used genuine saved NASA rows for the two
findings above. No implementation source or experimental result was modified
by this review.

## Remediation re-review

Re-reviewed the three fixes on 4 November 2026 before the final stable study.
**All three findings are resolved in the inspected implementation. No blocker
was found for starting a new final run.** The historical findings above are
retained as an audit trail; superseded studies do not become corrected evidence.

1. **Few-shot comparison:** `_few_shot` now constructs the condition baseline,
   freezes its held-out ownership, removes discarded target-client cases from
   the eligible frame, and leaves other clients unchanged. Client selection and
   the retained matching case use documented conditions, never held-out wear
   values. This is a declared condition-based evaluation design. The workflow
   uses that reduced frame for pooled/local/federated fitting and preprocessing,
   and writes the supported client's comparison to
   `limited_label_comparison.csv`; absent support is explicitly infeasible.
   On the genuine NASA smoke features with seed 42: target client 1 retains
   case 02; cases 03 and 05 are excluded; training cases fall from 8 to 6;
   the frozen test cases for that client are 11, 12 and 16. Direct assertions
   confirmed identical baseline/few-shot held-out rows and ownership, exclusion
   from the preprocessing fit, and an unchanged complete assignment manifest
   after replacing every held-out wear value.
2. **Plot identity:** `_prediction_points` preserves every recorded column
   except prediction and training seed in sample identity, including dataset
   and `run_id`; it rejects duplicate within-seed provenance. The plotted points
   are exported to `overview_predictions.csv`. The genuine smoke FedAvg test
   data now produce all 161 points, with no cross-run collapsing. The regression
   test independently covers repeated wear/window indices in two source runs.
3. **Executed source provenance:** `source_snapshot` hashes ordered Python
   source files and project dependency declarations. The study manifest stores
   the file manifest and aggregate digest at start; model data provenance
   carries that code digest. The workflow compares the source again before
   declaring completion and fails if it changed. The source-change regression
   verifies digest sensitivity. Final scientific evidence must come from a new
   stable run; no retrospective digest should be assigned to superseded runs.

Re-review verification:

```sh
.venv/bin/pytest tests/test_splits.py tests/test_features.py tests/test_evaluation.py tests/test_config.py tests/test_workflow.py -q
```

Result: **48 passed**, two upstream Typer/Click deprecation warnings, 43.32 s.
The checks above establish corrected code behavior, not final-study completion
or favorable experimental outcomes. No implementation files were changed by
this re-review.
