"""
Filtering and preprocessing blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np
import warnings

from core.types import BlockOutput, Array
from core.backend import Backend


@dataclass
class BandpassFilter:
    """Bandpass filter for EEG signals"""
    band_hz: Tuple[float, float] = (0.5, 45.0)
    order: int = 4
    zero_phase: bool = True

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        sfreq = float(context["sfreq"])
        signal: Array = context["signal"]
        
        # Validate filter parameters
        nyquist = sfreq / 2.0
        
        # Validate order
        if not (1 <= self.order <= 10):
            raise ValueError(
                f"Filter order {self.order} is invalid. Must be between 1 and 10."
            )
        
        # Validate frequency range
        if not (0 < self.band_hz[0] < self.band_hz[1] < nyquist):
            raise ValueError(
                f"Bandpass range {self.band_hz} is invalid. "
                f"Must satisfy: 0 < low < high < Nyquist ({nyquist:.1f} Hz)"
            )
        
        # Warn about very narrow bands at high order
        bandwidth = self.band_hz[1] - self.band_hz[0]
        if bandwidth < 1.0 and self.order > 4:
            warnings.warn(
                f"Narrow bandwidth ({bandwidth:.2f} Hz) with high order ({self.order}) "
                f"may cause numerical issues"
            )

        # Work in numpy for scipy
        if backend.name == "jax":
            signal_np = backend.to_numpy(signal)
        else:
            signal_np = np.asarray(signal)

        from scipy.signal import butter, sosfiltfilt, sosfilt
        sos = butter(self.order, self.band_hz, btype="band", fs=sfreq, output="sos")

        if self.zero_phase:
            filt = sosfiltfilt(sos, signal_np, axis=1)
        else:
            filt = sosfilt(sos, signal_np, axis=1)

        if backend.name == "jax":
            filt = backend.array(filt)

        return BlockOutput(
            data={"signal": filt},
            metadata={
                "filter": "bandpass",
                "band_hz": self.band_hz,
                "order": self.order,
                "zero_phase": self.zero_phase,
            },
        )


@dataclass
class NotchFilter:
    """Notch filter for line noise removal"""
    freq_hz: float = 60.0
    q: float = 30.0
    zero_phase: bool = True

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        sfreq = float(context["sfreq"])
        signal: Array = context["signal"]
        
        # Validate filter parameters
        nyquist = sfreq / 2.0
        if not (0 < self.freq_hz < nyquist):
            raise ValueError(
                f"Notch frequency {self.freq_hz} Hz is invalid. "
                f"Must be between 0 and Nyquist ({nyquist:.1f} Hz)"
            )
        
        # Validate Q
        if self.q <= 0:
            raise ValueError(f"Notch Q must be positive, got {self.q}")
        
        # Clip Q to reasonable range for stability
        q_safe = min(self.q, 100.0)
        if q_safe != self.q:
            warnings.warn(f"Notch Q={self.q} is very high, clipping to {q_safe} for stability")

        if backend.name == "jax":
            x = backend.to_numpy(signal)
        else:
            x = np.asarray(signal)

        from scipy.signal import iirnotch, filtfilt, lfilter
        b, a = iirnotch(w0=self.freq_hz, Q=q_safe, fs=sfreq)

        y = filtfilt(b, a, x, axis=1) if self.zero_phase else lfilter(b, a, x, axis=1)

        if backend.name == "jax":
            y = backend.array(y)

        return BlockOutput(
            data={"signal": y},
            metadata={
                "filter": "notch",
                "freq_hz": self.freq_hz,
                "q": self.q,
                "zero_phase": self.zero_phase,
            },
        )


@dataclass
class MovingRMS:
    """Moving RMS (amplitude envelope) for smoothing"""
    window_sec: float = 0.25
    replace_signal: bool = False  # If True, replace signal with envelope

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        sfreq = float(context["sfreq"])
        signal: Array = context["signal"]

        if backend.name == "jax":
            x = backend.to_numpy(signal)
        else:
            x = np.asarray(signal)

        w = max(1, int(self.window_sec * sfreq))
        kernel = np.ones(w) / w

        # RMS = sqrt(mean(x^2))
        x2 = x * x
        sm = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, x2)
        y = np.sqrt(sm + 1e-12)

        if backend.name == "jax":
            y = backend.array(y)

        # Return as side-channel by default, or replace signal if requested
        if self.replace_signal:
            return BlockOutput(
                data={"signal": y},
                metadata={"transform": "moving_rms", "window_sec": self.window_sec},
            )
        else:
            return BlockOutput(
                data={"moving_rms": y},
                metadata={"transform": "moving_rms", "window_sec": self.window_sec},
            )


@dataclass
class LowpassFilter:
    """Lowpass filter for smoothing"""
    cutoff_hz: float = 20.0
    order: int = 4
    zero_phase: bool = True

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        sfreq = float(context["sfreq"])
        signal: Array = context["signal"]
        
        # Validate filter parameters
        nyquist = sfreq / 2.0
        
        # Validate order
        if not (1 <= self.order <= 10):
            raise ValueError(
                f"Filter order {self.order} is invalid. Must be between 1 and 10."
            )
        
        # Validate cutoff
        if not (0 < self.cutoff_hz < nyquist):
            raise ValueError(
                f"Lowpass cutoff {self.cutoff_hz} Hz is invalid. "
                f"Must be between 0 and Nyquist ({nyquist:.1f} Hz)"
            )

        if backend.name == "jax":
            x = backend.to_numpy(signal)
        else:
            x = np.asarray(signal)

        # Use SOS for numerical stability
        from scipy.signal import butter, sosfiltfilt, sosfilt
        sos = butter(self.order, self.cutoff_hz, btype="low", fs=sfreq, output="sos")

        y = sosfiltfilt(sos, x, axis=1) if self.zero_phase else sosfilt(sos, x, axis=1)

        if backend.name == "jax":
            y = backend.array(y)

        return BlockOutput(
            data={"signal": y},
            metadata={
                "filter": "lowpass",
                "cutoff_hz": self.cutoff_hz,
                "order": self.order,
                "zero_phase": self.zero_phase,
            },
        )
