# Future campus read-only validation protocol

The campus CNC was unavailable in the governing project sources. This document and the software adapter prepare a later supervised test; they do not constitute permission, capture, or validation evidence.

1. Obtain a real laboratory-supervisor approval record. Identify controller, machining process, allowed read-only exports, safe qualified sensor mounting, and retention/access conditions.
2. Preserve source run/tool/group identifiers, timestamps, actual sensor units, calibration, and sampling rate. Do not change machine settings or safety behavior through this software.
3. Use the saved model's exact feature definitions and input units. A 250 Hz processed benchmark waveform is not interchangeable with a raw high-frequency acoustic waveform. Confirm preprocessing parity explicitly.
4. Collect microscope/qualified measured wear tied to completed runs, with units and measurement timing, if accuracy evaluation is intended. Preserve missing labels as missing.
5. Create an approved feature CSV with ordered model features plus `run_id`, `group_id`, optional `wear_target`, and optional timestamps. Create a JSON manifest using the example in configs; change flags only when the real prerequisites are fulfilled, and reference the actual approval record.
6. Run `fabfederate campus-validate` against a frozen trusted public-data model. The adapter reads existing files, never connects to or controls a CNC controller.
7. Examine exported predictions, measured-label error if available, and a descriptive input-shift indicator based on frozen training-standardized features. This indicator is not calibrated confidence.
8. If later adapting a model, reserve separate complete campus groups for training, validation, and final test; do not adapt on the evaluation pilot and report the same data as unseen.

One physical machine is a domain-validation pilot, not evidence of multi-site federated learning. Without measured labels the only permitted conclusion is input compatibility and unlabeled distribution shift.
