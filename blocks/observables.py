"""
Observable and validation blocks for EEG analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
import numpy as np
import warnings

from core.types import BlockOutput, Array
from core.backend import Backend


@dataclass
class TapSignal:
    """Capture signal snapshot without modification (for before/after comparisons)"""
    tag: str

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        # Return as a non-signal component so it doesn't participate in accumulation
        return BlockOutput(
            data={self.tag: context["signal"]},
            metadata={"tap": self.tag}
        )


@dataclass
class BandPower:
    """Compute power in a frequency band"""
    band_hz: Tuple[float, float]
    method: str = "welch"
    nperseg_sec: float = 2.0

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        sfreq = float(context["sfreq"])
        signal: Array = context["signal"]

        if backend.name == "jax":
            x = backend.to_numpy(signal)
        else:
            x = np.asarray(signal)

        from scipy.signal import welch
        nperseg = max(16, int(self.nperseg_sec * sfreq))

        powers = []
        for ch in range(x.shape[0]):
            f, pxx = welch(x[ch], fs=sfreq, nperseg=nperseg)
            mask = (f >= self.band_hz[0]) & (f <= self.band_hz[1])
            bp = np.trapezoid(pxx[mask], f[mask])
            powers.append(bp)

        powers = np.asarray(powers)  # V^2
        # Also provide µV^2 for display
        powers_uV2 = powers * 1e12

        return BlockOutput(
            data={
                "signal": context["signal"],
                "bandpower_v2": powers,
                "bandpower_uV2": powers_uV2,
            },
            metadata={
                "bandpower_band_hz": self.band_hz,
                "bandpower_mean_uV2": float(np.mean(powers_uV2)),
            },
        )


@dataclass
class MultiBandPower:
    """Compute power in multiple frequency bands
    
    Note: Default bands have intentional overlaps:
    - Alpha (8-13 Hz) and Sigma (12-16 Hz) overlap at 12-13 Hz
    - Sigma (12-16 Hz) and Beta (13-30 Hz) overlap at 13 Hz
    
    Band powers are NOT disjoint and should not be summed.
    Sigma typically represents sleep spindles (12-16 Hz).
    """
    bands: List[Tuple[str, float, float]] = None
    method: str = "welch"
    nperseg_sec: float = 2.0

    def __post_init__(self):
        if self.bands is None:
            # Standard EEG bands (with documented overlaps)
            self.bands = [
                ("delta", 0.5, 4.0),
                ("theta", 4.0, 8.0),
                ("alpha", 8.0, 13.0),
                ("sigma", 12.0, 16.0),  # Sleep spindles, overlaps with alpha/beta
                ("beta", 13.0, 30.0),
                ("gamma", 30.0, 99.0),
            ]

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        sfreq = float(context["sfreq"])
        signal: Array = context["signal"]
        
        # Validate method
        if self.method != "welch":
            raise NotImplementedError(
                f"Method '{self.method}' is not implemented. Only 'welch' is currently supported."
            )

        if backend.name == "jax":
            x = backend.to_numpy(signal)
        else:
            x = np.asarray(signal)

        from scipy.signal import welch
        nperseg = max(16, int(self.nperseg_sec * sfreq))

        band_powers = {}
        
        # Compute PSD once per channel (more efficient)
        n_channels = x.shape[0]
        psds = []
        freqs = None
        
        for ch in range(n_channels):
            f, pxx = welch(x[ch], fs=sfreq, nperseg=nperseg)
            if freqs is None:
                freqs = f
            psds.append(pxx)
        
        psds = np.array(psds)  # shape: (n_channels, n_freqs)
        
        # Now integrate power for each band
        for band_name, fmin, fmax in self.bands:
            mask = (freqs >= fmin) & (freqs <= fmax)
            powers = []
            
            for ch in range(n_channels):
                bp = np.trapezoid(psds[ch, mask], freqs[mask])
                powers.append(bp)
            
            powers = np.asarray(powers) * 1e12  # Convert to µV^2
            band_powers[band_name] = {
                "mean": float(np.mean(powers)),
                "per_channel": powers.tolist(),
            }

        return BlockOutput(
            data={"signal": context["signal"]},
            metadata={"band_powers": band_powers},
        )


@dataclass
class ValidationChecks:
    """Validation checks for signal quality"""
    rms_uv_range: Tuple[float, float] = (5.0, 50.0)
    peak2peak_uv_range: Tuple[float, float] = (20.0, 150.0)

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        sig: Array = context["signal"]

        x = backend.to_numpy(sig) if backend.name == "jax" else np.asarray(sig)

        # Check for NaN/Inf (critical validation)
        finite_mask = np.isfinite(x)
        finite_frac_per_channel = np.mean(finite_mask, axis=1)
        overall_finite_frac = float(np.mean(finite_frac_per_channel))
        
        # Warn if significant non-finite values
        if overall_finite_frac < 0.99:
            warnings.warn(
                f"Signal contains non-finite values: {(1-overall_finite_frac)*100:.2f}% "
                f"are NaN or Inf. This indicates filter instability or bad input."
            )
        
        # Hard fail if too many non-finite
        if overall_finite_frac < 0.5:
            raise ValueError(
                f"Signal is mostly non-finite ({(1-overall_finite_frac)*100:.1f}% NaN/Inf). "
                f"Cannot proceed with validation."
            )

        rms_uv = np.sqrt(np.mean(x**2, axis=1)) * 1e6
        p2p_uv = (np.max(x, axis=1) - np.min(x, axis=1)) * 1e6

        ok_rms = (rms_uv >= self.rms_uv_range[0]) & (rms_uv <= self.rms_uv_range[1])
        ok_p2p = (p2p_uv >= self.peak2peak_uv_range[0]) & (
            p2p_uv <= self.peak2peak_uv_range[1]
        )

        return BlockOutput(
            data={"signal": sig},
            metadata={
                "check_rms_uv_mean": float(np.mean(rms_uv)),
                "check_rms_uv_per_channel": rms_uv.tolist(),
                "check_p2p_uv_mean": float(np.mean(p2p_uv)),
                "check_p2p_uv_per_channel": p2p_uv.tolist(),
                "check_rms_uv_ok_frac": float(np.mean(ok_rms)),
                "check_p2p_uv_ok_frac": float(np.mean(ok_p2p)),
                "check_finite_frac": overall_finite_frac,
                "check_finite_per_channel": finite_frac_per_channel.tolist(),
            },
        )


@dataclass
class BandSignalExtractor:
    """Extract filtered signals for each EEG band"""
    bands: List[Tuple[str, float, float]] = None

    def __post_init__(self):
        if self.bands is None:
            self.bands = [
                ("delta", 0.5, 4.0),
                ("theta", 4.0, 8.0),
                ("alpha", 8.0, 13.0),
                ("sigma", 12.0, 16.0),
                ("beta", 13.0, 30.0),
                ("gamma", 30.0, 99.0),
            ]

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        sfreq = float(context["sfreq"])
        signal: Array = context["signal"]

        if backend.name == "jax":
            x = backend.to_numpy(signal)
        else:
            x = np.asarray(signal)

        from scipy.signal import butter, sosfiltfilt

        band_signals = {}
        
        for band_name, fmin, fmax in self.bands:
            sos = butter(4, [fmin, fmax], btype="band", fs=sfreq, output="sos")
            filtered = sosfiltfilt(sos, x, axis=1)
            
            # Convert back to backend array if using JAX
            if backend.name == "jax":
                filtered = backend.array(filtered)
            
            band_signals[f"{band_name}_signal"] = filtered

        return BlockOutput(
            data={"signal": context["signal"], **band_signals},
            metadata={"band_extraction": "complete", "bands": [b[0] for b in self.bands]},
        )
