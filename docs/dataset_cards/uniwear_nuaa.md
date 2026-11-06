# Uniwear NUAA derived milling benchmark

Source: [Katulu Uniwear Dataset](https://github.com/katulu-io/uniwear-dataset), pinned commit `66c894f2449f4393126e906e086485a8f9d95171`. Bundled data are published under CC BY 4.0; attribution belongs to Katulu/Orlando Hohmeier/Mehmet Suzen and the original NUAA and PHM2010 dataset authors. No upstream executable code is used.

The downloaded `data/raw/uniwear/uniwear.csv` has 39,903 rows across nine NUAA experiments and three PHM2010 cutters. Its actual CSV contains three shared signals (`force_z`, `vibration_x`, `vibration_y`), a timestamp, wear, and dataset/experiment tags. The parser audits these actual columns rather than assuming all six signals mentioned in the general repository description are present.

The trained `uniwear_nuaa` study uses **only** the 31,388 NUAA rows from W1–W9. Complete experiments are assigned to train/validation/test before fitting any imputer, scaler or model. Three logical training clients receive disjoint complete training experiments. Timestamp, upstream row index, experiment ID, source ID and wear are excluded from predictor features. The PHM2010 subset is excluded: three labeled cutters cannot supply three separate training clients plus cutter-held-out validation and test.

These are approximately 2 Hz publisher-smoothed/downsampled signals and derived wear targets in millimetres. They are not original high-frequency waveforms or 31,388 independent microscope measurements. Upstream smoothing and target interpolation limit interpretations; this is an offline derived-target benchmark, evaluated separately from NASA's measured-label study. It does not demonstrate a gain on NASA or support raw-waveform/online causal claims.

The source manifest records the pinned URL, bytes, locally computed SHA-256, license and processing caveat. Download with `.venv/bin/python -m fabfederate download --dataset uniwear`; reproduce using `configs/uniwear_nuaa.yaml`. Saved model bundles, metrics, splits, hashes and SHAP remain inside `artifacts/experiments/uniwear-nuaa/`.
