# Verified implementation and experiment evidence

## Optimization and additional-data results — 6 November 2026

The original NASA results below remain the historical reference. New study artifacts are under `artifacts/experiments/nasa-optimized/` and `artifacts/experiments/uniwear-nuaa/`. All reported errors are test MAE/RMSE in mm, averaged across seeds 11, 22 and 33. Repeated seeds use the same fixed group split and do not constitute new independent experimental data.

| NASA model / study | MAE | RMSE | MAE seed SD |
|---|---:|---:|---:|
| Original FedAvg | 0.176218 | 0.206747 | 0.001883 |
| Optimized FedAvg | 0.173872 | 0.204798 | 0.002890 |
| Optimized FedProx | 0.173865 | 0.204784 | 0.002898 |
| New gradient-boosting comparator | 0.138254 | 0.177923 | 0.000000 |

FedAvg MAE decreases by 1.33% on the established test cases. This is a modest exploratory gain: seed variability overlaps, and the test split has been reused across research iterations. The gradient-boosting result is a separate non-federated comparator, not evidence that FL beats centralized training. Neural settings and per-family classical candidates were selected by validation scores; test errors were not used in those selectors. The app initially selects a learned model by saved validation error, which can differ from the lowest observed test-error model.

### Separate Uniwear NUAA benchmark

[Source and limitations](../dataset_cards/uniwear_nuaa.md): 31,388 publisher-smoothed rows from nine NUAA experiments, three actual signal columns, no timestamp/ID/target predictors, three logical training clients and complete experiment holdouts. Publisher-derived wear targets are not independent original measurements. These errors cannot be compared directly with NASA or used to claim transfer gains.

| NUAA model | MAE | RMSE | MAE seed SD |
|---|---:|---:|---:|
| median | 0.070439 | 0.077386 | 0.000000 |
| centralized_mlp | 0.060689 | 0.067337 | 0.002538 |
| fedavg | 0.058359 | 0.067169 | 0.009307 |
| fedprox | 0.058244 | 0.067067 | 0.009498 |
| random_forest | 0.067867 | 0.076414 | 0.000039 |
| extra_trees | 0.067722 | 0.075209 | 0.000159 |
| hist_gradient_boosting | 0.068945 | 0.076392 | 0.000000 |
| ridge | 0.061858 | 0.068093 | 0.000000 |

Every run saves bundles, validation selections, held-out predictions, client/group manifests and model-specific SHAP. Attribution stability records the actual `top_k` (three on this dataset), so full-set overlap is descriptive and is not proof of explanation robustness.

## Original NASA study

Completed offline public-data research implementation, 4 November 2026. This summary is generated from the saved real NASA study CSVs. The full automatically generated report is [saved study report](../../artifacts/experiments/nasa-final/report.md).

## What was executed

- Five scenarios (IID, condition, quantity, label skew and controlled few-shot), three seeds (11,22,33), nine predictive models per run, 30 federated rounds and three local epochs.
- Validation-only selection over FedProx mu values0,0.001,0.01,0.1. Final tests use four held-out NASA cases, not verified independent physical tools.
- 15 main runs and270 saved validation/test metric records. The default feature table has1027 windows and36 predictors; measured-label provenance and exclusions are audited.
- Separate seed11 FFT-feature and15-round efficiency experiments. FFT extends36 to54 predictors; it analyzes already processed250Hz sensor signals, not raw high-frequency acoustic recordings.
- Model-specific SHAP examples across held-out groups and repeated seeds, plus feature-noise robustness and missing-feature rejection.
- Fresh-process reproduction of IID seed11: all4905 saved prediction rows exactly match the original study. The locked environment is shared; this is not new independent experimental data.
- Run the current correctness checks with `make -f tools/Makefile test` and `make -f tools/Makefile check`; setup and command details are in the [root README](../../README.md).

## Observed held-out errors

Errors are in millimetres. Seed standard deviation measures training variability on the same fixed case split; it is not a physical-tool confidence interval.

| scenario | method | mae_mean | mae_seed_std | rmse_mean |
| --- | --- | --- | --- | --- |
| condition | centralized_mlp | 0.173054 | 0.003490 | 0.203879 |
| condition | fedavg | 0.172988 | 0.003347 | 0.200559 |
| condition | fedprox | 0.173215 | 0.003534 | 0.201911 |
| condition | local_client_0 | 0.172526 | 0.004552 | 0.219519 |
| condition | local_client_1 | 0.160439 | 0.001926 | 0.194492 |
| condition | local_client_2 | 0.181438 | 0.011825 | 0.246803 |
| condition | mean | 0.185673 | 0.000000 | 0.212270 |
| condition | median | 0.170470 | 0.000000 | 0.206701 |
| condition | random_forest | 0.188248 | 0.001745 | 0.224198 |
| few_shot | centralized_mlp | 0.221899 | 0.029991 | 0.515787 |
| few_shot | fedavg | 0.214184 | 0.017294 | 0.435283 |
| few_shot | fedprox | 0.214046 | 0.017069 | 0.433362 |
| few_shot | local_client_0 | 0.176824 | 0.009209 | 0.254396 |
| few_shot | local_client_1 | 0.189478 | 0.015267 | 0.297475 |
| few_shot | local_client_2 | 0.228024 | 0.036099 | 0.612287 |
| few_shot | mean | 0.197289 | 0.000000 | 0.224144 |
| few_shot | median | 0.173918 | 0.000000 | 0.205475 |
| few_shot | random_forest | 0.204481 | 0.004976 | 0.239461 |
| iid | centralized_mlp | 0.173054 | 0.003490 | 0.203879 |
| iid | fedavg | 0.176218 | 0.001883 | 0.206747 |
| iid | fedprox | 0.176529 | 0.002397 | 0.207135 |
| iid | local_client_0 | 0.181920 | 0.010744 | 0.245083 |
| iid | local_client_1 | 0.174145 | 0.001921 | 0.213921 |
| iid | local_client_2 | 0.157928 | 0.001671 | 0.194102 |
| iid | mean | 0.185673 | 0.000000 | 0.212270 |
| iid | median | 0.170470 | 0.000000 | 0.206701 |
| iid | random_forest | 0.188248 | 0.001745 | 0.224198 |
| label | centralized_mlp | 0.173054 | 0.003490 | 0.203879 |
| label | fedavg | 0.173561 | 0.001850 | 0.201258 |
| label | fedprox | 0.173789 | 0.002169 | 0.201504 |
| label | local_client_0 | 0.165555 | 0.002013 | 0.202059 |
| label | local_client_1 | 0.176834 | 0.003598 | 0.229591 |
| label | local_client_2 | 0.181438 | 0.011825 | 0.246803 |
| label | mean | 0.185673 | 0.000000 | 0.212270 |
| label | median | 0.170470 | 0.000000 | 0.206701 |
| label | random_forest | 0.188248 | 0.001745 | 0.224198 |
| quantity | centralized_mlp | 0.173054 | 0.003490 | 0.203879 |
| quantity | fedavg | 0.175709 | 0.001379 | 0.204393 |
| quantity | fedprox | 0.175097 | 0.004689 | 0.206234 |
| quantity | local_client_0 | 0.174999 | 0.005397 | 0.234091 |
| quantity | local_client_1 | 0.188336 | 0.002483 | 0.243342 |
| quantity | local_client_2 | 0.175043 | 0.006602 | 0.208532 |
| quantity | mean | 0.185673 | 0.000000 | 0.212270 |
| quantity | median | 0.170470 | 0.000000 | 0.206701 |
| quantity | random_forest | 0.188248 | 0.001745 | 0.224198 |

## Controlled limited-label result

One target client retains one training case. Its fixed held-out domains cover three cases. The withheld target-client training cases are excluded from every method's eligible training union. All values below average the three recorded training seeds.

| method | mean_mae | mean_rmse |
| --- | --- | --- |
| centralized_mlp | 0.235733 | 0.560522 |
| fedavg | 0.225279 | 0.470225 |
| fedprox | 0.225084 | 0.468082 |
| local_client_1 | 0.202074 | 0.319061 |

For this tested limited-label setting, the local client has lower mean error than the federated methods. The experiments therefore do not support a universal claim that collaboration improves wear prediction. Some held-out regimes produce large errors; this remains a research demonstrator rather than a qualified inspection system.

## Feature and efficiency comparisons

These are matched seed11/IID comparisons. Compute times were collected locally while other checks could run, so timings are descriptive rather than isolated hardware benchmarks. Model payload bytes exclude transport envelopes and scalar metadata.

| experiment | method | mae_mm | rmse_mm | model_payload_bytes |
| --- | --- | --- | --- | --- |
| time-only,30rounds,3localepochs | centralized_mlp | 0.173620 | 0.200817 | unavailable |
| time-only,30rounds,3localepochs | fedavg | 0.176975 | 0.206167 | 3364560.000000 |
| time-only,30rounds,3localepochs | fedprox | 0.176703 | 0.205733 | 3364560.000000 |
| time+FFT,30rounds,3localepochs | centralized_mlp | 0.113829 | 0.151756 | unavailable |
| time+FFT,30rounds,3localepochs | fedavg | 0.122864 | 0.161019 | 4194000.000000 |
| time+FFT,30rounds,3localepochs | fedprox | 0.122864 | 0.161019 | 4194000.000000 |
| time-only,15rounds,1localepoch | centralized_mlp | 0.173620 | 0.200817 | unavailable |
| time-only,15rounds,1localepoch | fedavg | 0.173359 | 0.208176 | 1682280.000000 |
| time-only,15rounds,1localepoch | fedprox | 0.173359 | 0.208176 | 1682280.000000 |

## Evidence boundaries and remaining gates

The implementation is software-complete for the required offline study. Physical campus access, approved acquisition, and measured campus labels remain unavailable. The optional PHM adapter is implemented and tested with source-shaped fixtures; a real PHM benchmark was not executed, and its three released labeled cutters cannot support the default NASA three-client plus separate validation/test protocol. Wavelet/CNN models, a separate API, and remote multi-site federation remain optional and unimplemented.

The exact training-source file digests match local Git commit `f4d8caa`, recorded in the completed study manifests. Later changes only correct dashboard record labels, experiment-scoped selection, and line rendering; the model-training code and saved predictions are unchanged. Earlier `nasa-study` output is explicitly superseded.

See the [NASA card](../dataset_cards/nasa_milling.md), [PHM card](../dataset_cards/phm2010_milling.md), and [technical decisions](../technical_decisions.md). The complete local artifacts are under `artifacts/experiments/nasa-final/`.
