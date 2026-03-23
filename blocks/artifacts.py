"""
Artifact simulation blocks (EOG, EMG, etc).
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from dataclasses import dataclass, field

from core.types import BlockOutput, Array
from core.backend import Backend, RNGManager


@dataclass
class EOGArtifacts:
    """Eye blink and saccade artifacts"""
    enable: bool = True
    rate_per_min: float = 15.0
    duration_range: Tuple[float, float] = (0.1, 0.3)
    amplitude_range_uv: Tuple[float, float] = (50.0, 150.0)
    # Optional default, then set concrete list in __post_init__
    channels: Optional[List[int]] = None
    add_saccades: bool = False
    saccade_prob: float = 0.3

    def __post_init__(self) -> None:
        if self.channels is None:
            self.channels = [0, 1]

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        if not self.enable:
            return BlockOutput(
                data={"signal": context["signal"]},
                metadata={"eog_enabled": False},
            )

        backend = Backend(context.get("backend", "numpy"))
        rng = RNGManager(backend, seed=hash(str(key)) % (2**31))

        signal: Array = context["signal"]
        duration: float = float(context["duration"])
        sfreq: float = float(context["sfreq"])
        n_samples: int = int(context["n_samples"])
        n_channels: int = int(context["n_channels"])
        times: Array = backend.asarray(context["times"])

        # Work in NumPy for injection (simplifies in-place ops)
        if backend.name == "jax":
            signal = backend.to_numpy(signal)
        signal = np.asarray(signal, dtype=float)

        avg_interval = 60.0 / self.rate_per_min
        events: List[Dict[str, Any]] = []

        t = 0.0
        while t < duration:
            t += float(backend.to_numpy(rng.exponential(avg_interval)))
            if t >= duration:
                break

            dur = float(backend.to_numpy(rng.uniform(self.duration_range[0], self.duration_range[1])))
            amp = float(backend.to_numpy(rng.uniform(self.amplitude_range_uv[0], self.amplitude_range_uv[1]))) * 1e-6

            start_idx = int(t * sfreq)
            dur_samples = int(dur * sfreq)
            end_idx = min(start_idx + dur_samples, n_samples)

            blink_len = end_idx - start_idx
            if blink_len < 2:
                continue

            # Blink shape: x * exp(-5*x)
            x = np.linspace(0.0, 1.0, blink_len)
            blink = amp * x * np.exp(-5.0 * x)

            # Add to frontal channels
            for ch in self.channels or []:
                if 0 <= ch < n_channels:
                    signal[ch, start_idx:end_idx] += blink

            # Add saccades
            if (
                self.add_saccades
                and self.channels
                and len(self.channels) >= 2
                and float(backend.to_numpy(rng.uniform(0.0, 1.0))) < self.saccade_prob
            ):
                ch_left = self.channels[0]
                ch_right = self.channels[1]
                saccade_amp = amp * 0.5
                direction = int(np.asarray(rng.choice([-1, 1])))  # force to Python int

                if 0 <= ch_left < n_channels:
                    signal[ch_left, start_idx:end_idx] += direction * saccade_amp
                if 0 <= ch_right < n_channels and ch_right != ch_left:
                    signal[ch_right, start_idx:end_idx] -= direction * saccade_amp

            events.append(
                {
                    "onset": t,
                    "duration": dur,
                    "type": "blink",
                    "amplitude_uv": amp * 1e6,
                }
            )

        # Convert back to backend array
        if backend.name == "jax":
            signal = backend.array(signal)

        return BlockOutput(
            data={"signal": signal},
            metadata={"eog_enabled": True, "n_blinks": len(events)},
            events=events,
        )


@dataclass
class EMGArtifacts:
    """Muscle artifacts"""
    enable: bool = True
    rate_per_min: float = 8.0
    duration_range: Tuple[float, float] = (0.2, 0.8)
    amplitude_range_uv: Tuple[float, float] = (10.0, 40.0)
    band_hz: Tuple[float, float] = (20.0, 200.0)

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        if not self.enable:
            return BlockOutput(
                data={"signal": context["signal"]},
                metadata={"emg_enabled": False},
            )

        backend = Backend(context.get("backend", "numpy"))
        rng = RNGManager(backend, seed=hash(str(key)) % (2**31))

        signal: Array = context["signal"]
        duration: float = float(context["duration"])
        sfreq: float = float(context["sfreq"])
        n_samples: int = int(context["n_samples"])
        n_channels: int = int(context["n_channels"])

        # Work in NumPy for in-place ops and SciPy filter
        if backend.name == "jax":
            signal = backend.to_numpy(signal)
        signal = np.asarray(signal, dtype=float)

        # Create bandpass filter
        from scipy.signal import butter, sosfiltfilt

        sos = butter(4, self.band_hz, btype="band", fs=sfreq, output="sos")

        avg_interval = 60.0 / self.rate_per_min
        events: List[Dict[str, Any]] = []

        t = 0.0
        while t < duration:
            t += float(backend.to_numpy(rng.exponential(avg_interval)))
            if t >= duration:
                break

            dur = float(backend.to_numpy(rng.uniform(self.duration_range[0], self.duration_range[1])))
            amp_uv = float(backend.to_numpy(rng.uniform(self.amplitude_range_uv[0], self.amplitude_range_uv[1])))
            amp = amp_uv * 1e-6

            start_idx = int(t * sfreq)
            dur_samples = int(dur * sfreq)
            end_idx = min(start_idx + dur_samples, n_samples)

            burst_len = end_idx - start_idx
            if burst_len < 4:
                continue

            # Generate bandpass-filtered noise (mean 0, std=amp)
            emg = np.asarray(rng.normal(0.0, amp, size=burst_len), dtype=float)
            emg = sosfiltfilt(sos, emg)

            # Add to a random channel
            ch = int(np.asarray(rng.integers(0, n_channels)))
            signal[ch, start_idx:end_idx] += emg

            events.append({"onset": t, "duration": dur, "type": "muscle", "channel": ch})

        if backend.name == "jax":
            signal = backend.array(signal)

        return BlockOutput(
            data={"signal": signal},
            metadata={"emg_enabled": True, "n_bursts": len(events)},
            events=events,
        )


@dataclass
class LineNoise:
    """Power line noise (50/60 Hz)"""
    freq_hz: float = 60.0
    amplitude_uv: float = 2.0
    harmonics: int = 2
    jitter_hz: float = 0.05
    phase_drift: bool = False

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        rng = RNGManager(backend, seed=hash(str(key)) % (2**31))

        signal: Array = backend.asarray(context["signal"])
        times: Array = backend.asarray(context["times"])
        n_channels = int(signal.shape[0])

        amplitude_v = float(self.amplitude_uv) * 1e-6
        noise: Array = backend.zeros(signal.shape)

        for h in range(1, self.harmonics + 1):
            for ch in range(n_channels):
                freq_offset = float(backend.to_numpy(rng.uniform(-self.jitter_hz, self.jitter_hz)))
                actual_freq = float(h * self.freq_hz + freq_offset)
                phase = float(backend.to_numpy(rng.uniform(0.0, 2.0 * np.pi)))

                phase_term: Array
                if self.phase_drift:
                    if backend.name == "numpy":
                        phase_term = 2.0 * np.pi * float(backend.to_numpy(rng.normal(0.0, 0.02))) * times
                    else:
                        phase_term = 2.0 * np.pi * rng.normal(0.0, 0.02, size=len(times)) * times
                    phase = 0.0  # incorporate phase via phase_term
                else:
                    phase_term = 0.0

                harm_amp = amplitude_v / float(h)
                component = harm_amp * backend.sin(2.0 * np.pi * actual_freq * times + phase + phase_term)

                if backend.name == "jax":
                    noise = noise.at[ch].add(component)  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    noise[ch] += backend.to_numpy(component)

        signal = signal + noise
        return BlockOutput(
            data={"signal": signal},
            metadata={"line_noise_hz": self.freq_hz, "harmonics": self.harmonics},
        )


@dataclass
class BaselineDrift:
    """Slow baseline drift"""
    amplitude_uv: float = 5.0

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        rng = RNGManager(backend, seed=hash(str(key)) % (2**31))

        signal: Array = backend.asarray(context["signal"])
        times: Array = backend.asarray(context["times"])
        duration = float(context["duration"])
        n_channels = int(signal.shape[0])

        drift_strength = float(self.amplitude_uv) * 1e-6
        n_segments = max(1, int(duration // 2))

        for ch in range(n_channels):
            # random drift control points
            drift_points = np.asarray(rng.normal(0.0, drift_strength, size=n_segments + 1))

            # Interpolate drift
            if backend.name == "numpy":
                drift = np.interp(times, np.linspace(0.0, duration, n_segments + 1), drift_points)
            else:
                drift = backend.interp(  # pyright: ignore[reportAttributeAccessIssue]
                    times,
                    backend.linspace(0.0, duration, n_segments + 1),
                    backend.array(drift_points),
                )

            if backend.name == "jax":
                signal = signal.at[ch].add(drift)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                signal[ch] += backend.to_numpy(drift)

        return BlockOutput(
            data={"signal": signal},
            metadata={"baseline_drift_uv": self.amplitude_uv},
        )
