# NASA PCoE Milling data card

Audited 4 November 2026 against the acquired archive and its bundled `Readme.pdf`.
This describes processed recorded sensor signals and measured flank wear; it makes
no campus, industrial deployment, forecasting, or cross-cutter generalization claim.

| Provenance | Verified value |
|---|---|
| Dataset | NASA PCoE Milling Data Set, Agogino and Goebel (2007), BEST Lab UC Berkeley |
| Catalogue | [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| Acquired archive | [NASA-linked milling archive](https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip) |
| Acquisition UTC | 2026-11-04T13:27:08.079982+00:00 |
| Archive bytes | 14,731,306 |
| Archive SHA-256 | `bdba8d52ec1a1baab24c2be58480e6ac62508c8cc1f8219f47ebde8fc9ebc474` |
| Extracted MATLAB SHA-256 | `71486a857939d1416c86c7cf0c469d5e69c7c30495e01ec8a0e13aafd2a313cb` |
| Original structure | `3. Milling/mill.zip`, `3. Milling/Readme.pdf`; inner ZIP contains `mill.mat` |
| Parser | `fabfederate.data.nasa_loader`, schema v1, SciPy MATLAB reader |
| Terms | Public archive linked by NASA; no separate license file found. Attribution preserved; redistribution rights not established by this audit. |

The acquisition manifest is `data/raw/nasa_source_manifest.json`. Raw data are
read-only inputs; generated artifacts are saved separately.

The `mill` variable contains 167 MATLAB structs from 16 case experiments. Each
case contains repeated cuts/runs under one condition. The split group is the
original case, and the run identifier combines case and source run number. No
physical cutter identifier is generated. The readme describes eight combinations
of settings repeated with a second set of inserts (pp. 1–2); it does not establish
an independently verified physical-cutter ID for every source record. A case-safe
evaluation therefore supports held-out experiment claims with that limitation.

| Source fields | Meaning and handling |
|---|---|
| `case`, `run` | Original integer identifiers; metadata only |
| `VB` | Measured flank wear, millimetres; target only |
| `time` | Duration field restarting for each case; retained as source provenance, excluded from predictors |
| `DOC` | Depth of cut, 0.75 or 1.5 mm |
| `feed` | Feed, 0.25 or 0.5 mm/rev |
| `material` | Original code 1: cast iron; 2: steel (setup text specifies stainless steel J45) |
| `smcAC`, `smcDC` | AC and DC spindle motor current acquisition channels |
| `vib_table`, `vib_spindle` | Vibration acquisition channels at table and spindle |
| `AE_table`, `AE_spindle` | Acoustic emission acquisition channels at table and spindle |

The six channels each contain 1,509,360 values across all 167 runs. Run lengths
are 9,000 or 15,360 samples. All source signal values are finite. No complete run
is constant, but individual windows can be constant in `smcDC`. The source
describes amplification/filtering and RMS processing of vibration and acoustic
emission before acquisition (pp. 2–3); these are not raw high-frequency waveforms.
The documented processed-data sampling rate is **250 Hz** (p. 3). The 100 kHz
number describes the acquisition board's maximum capability, not this table's
sampling rate. Sensor amplitudes are preserved in source acquisition coordinates;
this audit does not infer physical calibration into acceleration/current units.
No digital timestamps or sensor clipping thresholds are supplied as fields.

There are **146 measured VB labels and 21 missing labels**, encoded as NaN in
the actual file. The loader also recognizes an empty VB field as missing. The
observed measured range is 0–1.53 mm. Zero is a measured target, not a missing
sentinel. The readme states VB was measured after selected runs at irregular
intervals (p. 1). A microscope was used, and Figure 6 explicitly labels VB in mm
(p. 7). Each eligible run is associated only with its own recorded VB. No label
interpolation, progression synthesis, or reassignment of later measurements is
performed. A run's windows share its label and are dependent observations.

The baseline feature schema has 36 predictors: mean, RMS, population standard
deviation, population variance, peak-to-peak, and biased Fisher excess kurtosis
for each of six channels, ordered by channel then statistic. Settings define
documented domains; they are not predictors in this initial implementation.
Identifiers, source time, labels, split and client ownership are metadata. No
normalization, filtering, imputation, scaling, or feature selection is fitted
here. The model bundle fits its common preprocessing from training rows only.
Optional spectral summaries require a compatible confirmed sampling rate and
operate on these processed signals; they do not recover raw acoustic frequencies.

Windows stay within a run. A full window is excluded if any channel is constant,
numerically degenerate, nonfinite, too short, or malformed. No padding or partial
tail window is added. Every rejection is logged by run/window with its reason.
Missing-label runs remain in the source audit and case-split manifest but do not
produce supervised feature rows.

A deterministic default split (seed 42, test fraction .2, validation fraction .2,
both fractions of all cases, rounded up) gives 8 training, 4 validation and 4 test
cases. This is a modest 16-group benchmark; hundreds of windows do not increase
the independent case count. Split and partition seeds are separate. Feature
preparation emits `features.parquet`, `schema.json`, `audit.json`,
`split_manifest.json`, and `client_manifest.json` with source hashes/provenance.

Actual preparation verification at nonoverlapping 1,024-sample windows produced
1,027 valid windows from 146 labeled runs: 482 training, 226 validation and 319
test; 141 windows were rejected for constant/numerically degenerate `smcDC`.
At 2,048 samples it produced 528 windows and 56 such exclusions. These are
preparation counts, not trained-model performance measurements.

All five three-client scenarios are feasible at seed 42: random complete-group
IID-ish allocation, exact condition skew, unequal group quantities, training-only
group-label-mean skew, and one group for the few-shot client. Whole cases stay
together. IID-ish allocation does not promise equal sample counts or identical
wear distributions. Group-label averages use distinct runs rather than repeated
window labels. Few-shot starts from the condition partition, freezes its held-out
domain ownership, and retains only the configured number of whole training cases
for one client with supported test coverage. Discarded training cases are removed
from the common eligible training union for every comparison method; other clients
and every held-out row remain unchanged. At seed 42 the supported target is
`client_1`, retaining case 2 and excluding cases 3 and 5, with the same test cases
11/12/16 across methods. The manifest explicitly reports available/infeasible E4
coverage and the reason.

Held-out client ownership uses exact `DOC/feed/material` condition matches only
when the condition has one unique training-client owner; labels play no role in
the held-out mapping. Ambiguous/unseen conditions receive null client IDs.
At seed 42, validation cases 4 and 10 have a condition absent from training and
have no defensible client-specific ownership. They remain in pooled/group
evaluation and are explicitly listed in the client manifest. Synthetic tests
verify ambiguous-domain handling separately.

PHM, campus recordings, additional tools/cutters, independently administered
clients, acquisition calibration, and redistribution permissions are not
established by this source. They require separate audit and admission.
