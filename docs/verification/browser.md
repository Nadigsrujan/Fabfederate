# Browser verification

Verified the running local Streamlit server at http://127.0.0.1:8501 through the in-app browser.

- Data Explorer: realNASA counts, case-disjoint split table, grouped unstacked wear statistics.
- Experiment Comparison: recorded unit-bearing metrics and predictions; no fabricated values.
- Client Heterogeneity: logical client counts, conditions and supported heldout-domain errors.
- Convergence / Cost:30 recorded rounds, valid curves and serialized payload totals; explicit finite nonempty line charts.
- Wear Estimate: selected final FedAvg model, source case_01_run_001/window0, measured0.0000mm, predicted0.3224mm, reconstruction error about3e-08 with32 training reference rows.
- Switching from efficiency to final artifacts resets model selection to the correct directory; no stale model path survives. Unique run/window labels remove ambiguous source records. Regression tests first reproduced the cross-experiment path error, then passed after directory-scoped widget keys.
- Fresh page/browser logs after final reload contain no errors or warnings. Visual layout, labels and keyboard-focusable controls reviewed.

Automated AppTest coverage exercises allfive screens, uploads, actual saved bundles, explanation flow and experiment switching.
