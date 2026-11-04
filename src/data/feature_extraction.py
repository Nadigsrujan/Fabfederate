"""Shared extraction definitions for preparation and recorded-signal inference."""

import numpy as np


def extract_window_features(signals, spectral=False, sampling_hz=None):
    """Population moments and Fisher excess kurtosis (biased moment estimator).

    Reject the full window if any channel is nonfinite or constant. No waveform
    normalization, interpolation, filtering, or label synthesis is performed.
    """
    if not signals:
        raise ValueError("No signal channels")
    if spectral and (sampling_hz is None or not np.isfinite(sampling_hz) or sampling_hz <= 0):
        raise ValueError("Spectral features require a verified positive sampling rate")
    result = {}
    for channel in sorted(signals):
        values = np.asarray(signals[channel], dtype=np.float64)
        if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
            raise ValueError(f"Channel {channel}: requires at least four finite one-dimensional samples")
        variance = float(np.var(values))
        if variance <= np.finfo(float).eps * max(1., float(np.mean(values ** 2))):
            raise ValueError(f"Channel {channel}: constant or numerically degenerate window")
        centered = values - values.mean()
        features = {"mean": float(values.mean()), "rms": float(np.sqrt(np.mean(values ** 2))),
                    "std": float(np.sqrt(variance)), "variance": variance,
                    "peak_to_peak": float(np.ptp(values)),
                    "kurtosis": float(np.mean(centered ** 4) / variance ** 2 - 3)}
        if spectral:
            power = np.abs(np.fft.rfft(centered)) ** 2
            frequencies = np.fft.rfftfreq(len(values), 1 / sampling_hz)
            features.update({"dominant_hz": float(frequencies[np.argmax(power)]),
                             "spectral_centroid_hz": float(np.sum(frequencies * power) / power.sum()),
                             "spectral_power": float(power.sum() / len(values) ** 2)})
        if not np.isfinite(list(features.values())).all():
            raise ValueError(f"Channel {channel}: nonfinite derived features")
        result.update({f"{channel}__{name}": value for name, value in features.items()})
    return result
