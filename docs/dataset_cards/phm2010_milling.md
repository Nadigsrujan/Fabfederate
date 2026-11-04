# Optional PHM2010 Milling: source card and local admission protocol

The [official PHM2010 challenge description](https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/)
was read through Firecrawl on 4 November 2026. No challenge archive was downloaded;
the page describes each cutter archive as approximately 800 MB. This adapter is
functional for explicitly mapped, independently obtained local CSVs. It does not
establish that the full benchmark has been acquired or evaluated.

Officially verified schema:

| Source item | Published description |
|---|---|
| Independent source groups | Six cutters `c1` through `c6` |
| Released measured training labels | `c1`, `c4`, `c6` |
| Challenge test cutters | `c2`, `c3`, `c5`; no measured labels admitted |
| Important correction | The publisher explicitly rejects the wear file formerly included in `c3.zip` |
| Wear timing and units | Measured after each cut, in `10^-3 mm` |
| Acquisition CSV columns 1–3 | Force X/Y/Z, N |
| Acquisition CSV columns 4–6 | Vibration X/Y/Z, g |
| Acquisition CSV column 7 | Acoustic emission RMS, V |
| Sampling rate | 50,000 Hz/channel |
| Published process conditions | 10,400 RPM, 1,555 mm/min feed, radial depth 0.125 mm, axial depth 0.2 mm |

The [official scoring description](https://phmsociety.org/phm10-data-challenge-scoring/)
confirms three flute measurements and uses their maximum for the original
challenge criterion. FabFederate's task is an explicitly separate wear-regression
task. The adapter requires a chosen reduction of all three measured flutes:
`max` or `mean`. Neither is selected silently. All original edge measurements,
the chosen reduction, and label-file/row provenance are preserved. Output units
are `um`, exactly equal to the source `10^-3 mm`; the numeric conversion factor
is one. PHM wear must not be treated as NASA's millimetre-valued target or pooled
by matching column positions.

The actual archive header layout, exact cut filenames, per-cut row counts,
archive integrity hashes and redistribution terms have **not** been verified.
For this reason there is no filename globbing or guessed label-row alignment.
The user must inspect the local release, record its official source and hashes,
and supply this explicit mapping. Cut numbers must come from the source, not
the order of a directory listing. Wear headers below are illustrative aliases,
not a claim about the uninspected official wear-table headers.

```json
{
  "schema_version": 1,
  "source_url": "https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/",
  "wear_unit": "10^-3 mm",
  "sampling_hz": 50000,
  "signal_header": null,
  "signal_columns": {
    "force_x": 0, "force_y": 1, "force_z": 2,
    "vibration_x": 3, "vibration_y": 4, "vibration_z": 5, "ae_rms": 6
  },
  "cutters": [{
    "cutter_id": "c1",
    "label_file": "inspected/local/wear.csv",
    "label_header": 0,
    "label_columns": {
      "cut_id": "inspected_cut_column",
      "flute_1": "inspected_first_flute_column",
      "flute_2": "inspected_second_flute_column",
      "flute_3": "inspected_third_flute_column"
    },
    "records": [{"cut_id": 1, "signal_file": "inspected/local/acquisition.csv"}]
  }]
}
```

`signal_header`/`label_header` must explicitly be `null` for headerless files or
`0` for one header row. Signal column indices are zero-based and must map all
seven source channels once. Label selectors may be exact header strings or
zero-based column indices. Omit `label_file` for unlabeled cutters; no labels
are generated. A released training cut absent from its label table or with any
missing flute measurement receives a null target. Duplicate cut identities,
duplicate label keys, reused acquisition files, invalid/negative labels,
contradictory units or sample rates, and file paths escaping the declared local
root fail explicitly. Nonfinite signal samples remain auditable and are rejected
by feature-window quality gates; they are never imputed in the source loader.

Python API ready for CLI integration:

```python
from fabfederate.data.phm_loader import load_phm
from fabfederate.data.audit import audit_runs

runs = load_phm("local_phm_manifest.json", edge_reduction="max")
audit = audit_runs(runs)
```

Signature: `load_phm(manifest_path: str | Path, *, root: str | Path | None = None,
edge_reduction: str | None = None) -> list[SignalRun]`. `root` defaults to the
manifest's parent directory, and all data paths must stay inside it. The source
adapter never downloads or extracts archives. It hashes every signal/label input
and the manifest. Source cutter IDs become `group_id`; original cut IDs become
run identifiers. Released challenge membership is provenance, not a fabricated
wear label or predictor.

Before model training, inspect local hashes/terms and verify at least several
cut/measurement alignments manually against the original wear file. Save that
review alongside the generated audit. Keep this benchmark separate from NASA.
Use complete cutter-held-out evaluation and disclose the limited three labeled
cutters. A three-way train/validation/test split leaves only one labeled training
cutter, so three disjoint group-preserving training clients are infeasible under
the default protocol. The adapter does not weaken that grouping to manufacture
a federated experiment. A justified alternate evaluation design needs an explicit
protocol and separate results. No real PHM performance claim is made here.

Tests use tiny, explicitly synthetic source-shaped CSVs. They verify explicit
cut alignment, preserved per-flute measurements, both declared reductions,
missing targets, cutter grouping, test-label exclusion (including the c3 warning),
unit protection, duplicate labels and root confinement. They do not replace the
real-data admission review.
