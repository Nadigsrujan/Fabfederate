# Implementation decisions and official references

The user-authorized implementation.md is the governing specification. This records concrete engineering choices rather than claiming additional project results.

## Environment and orchestration

Python3.12 was available in the bundled runtime; the host system Python3.9 was below the specification floor. Dependencies are installed in .venv and frozen to requirements.lock. The installed Streamlit runtime warned about websockets17.4 with Flower-constrained uvicorn0.49, so websockets16.1.1 is explicitly constrained according to that runtime's compatibility guidance. `pip check` validates the resolved environment.

Flower's real Server.fit orchestration and FedAvg strategy are used with in-process ClientProxy objects wrapping NumPyClient. This keeps laptop execution deterministic and avoids a mandatory Ray worker stack while preserving Flower's client/update/aggregation contracts. It is a logical-client transport, not the Ray Simulation Runtime or a physically separated deployment. Local training uses one worker and fixed client order to avoid global Torch RNG races.

- [Flower simulation runtime](https://flower.ai/docs/framework/how-to-run-simulations.html)
- [Flower Server API](https://flower.ai/docs/framework/ref-api/flwr.server.Server.html)
- [Flower NumPyClient API](https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html)
- [Flower FedAvg API](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html)

## Data and evaluation

The downloaded NASA archive's own Readme.pdf is the label/channel/unit authority. Case grouping is preserved before window generation. VB is measured after runs at irregular intervals; NaN VB is excluded rather than interpolated. Sensor amplitude is retained; degenerate windows are excluded with recorded reasons. Fixed split and partition seeds are separate from training seeds.

The initial common imputer/scaler fits only allowed pooled public training rows, including the reduced union in limited-label experiments. This policy is disclosed rather than described as private independent-site preprocessing. Client updates use eligible unique window counts; repeated epochs do not multiply aggregation weights.

- [Official NASA catalogue and linked archive](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [Official optional PHM2010 description](https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/)

Held-out client ownership is assigned only for an exact documented condition with one training-client owner. Ambiguous or unseen conditions remain unavailable for client-specific errors. Few-shot freezes condition-partition heldout ownership, withholds complete target-client groups, and excludes those groups from every method's training union. Model/checkpoint/mu decisions use validation only.

## Explanations and visualization

Permutation SHAP calls the persisted model and preprocessing together on named raw features, uses a bounded training-derived reference, and checks output reconstruction. It explains no surrogate. Cross-seed top-k and sign agreement are descriptive stability measures. Dashboard bars use explicit grouped noninteractive Altair encodings so min/median/max are not incorrectly summed and categorical scale bindings do not produce misleading warnings.

- [SHAP PermutationExplainer](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html)
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing)
- [Altair grouped bar charts](https://altair-viz.github.io/gallery/grouped_bar_chart.html)

Every study records hashes of executed Python files, package declarations and lockfile, then checks that code remained unchanged during execution. Plotted predictions preserve run/case/window identity; averaging only combines the same recorded window across training seeds. Earlier pre-review outputs remain explicitly superseded.
