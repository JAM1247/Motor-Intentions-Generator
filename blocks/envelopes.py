"""
Envelope modulation blocks.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass

from core.types import BlockOutput
from core.backend import Backend, RNGManager


@dataclass
class BurstyEnvelope:
    """Random burst envelope (spindles, k-complexes style)"""
    rate_per_min: float = 12.0
    duration_sec: float = 2.0
    amp_factor: float = 2.5
    baseline_factor: float = 0.3
    normalize_post: bool = True
    target_rms_uv: Optional[float] = None

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        rng = RNGManager(backend, seed=hash(str(key)) % (2**31))

        signal = context["signal"]
        times = context["times"]          # kept for completeness/possible future use
        duration = context["duration"]
        sfreq = context["sfreq"]
        n_channels = context["n_channels"]
        n_samples = context["n_samples"]

        avg_interval = 60.0 / self.rate_per_min
        burst_dur = max(0.05, self.duration_sec)

        envelope = backend.ones((n_channels, n_samples)) * self.baseline_factor
        events: List[Dict[str, Any]] = []

        for ch in range(n_channels):
            # env buffer per-channel
            if backend.name == "numpy":
                env = np.ones(n_samples) * self.baseline_factor
            else:
                env = backend.ones(n_samples) * self.baseline_factor

            t_cur = 0.0
            while t_cur < duration:
                t_cur += rng.exponential(avg_interval)
                if t_cur >= duration:
                    break

                start_idx = int(t_cur * sfreq)
                dur_samples = int(burst_dur * sfreq)
                end_idx = min(start_idx + dur_samples, n_samples)

                burst_len = end_idx - start_idx
                if burst_len >= 2:
                    window = backend.hanning(burst_len)
                    burst_env = self.baseline_factor + self.amp_factor * window
                    env = backend.set_slice(env, slice(start_idx, end_idx), burst_env)

                    events.append({
                        "onset": t_cur,
                        "duration": burst_dur,
                        "channel": ch,
                        "type": "burst",
                    })

            envelope = backend.set_row(envelope, ch, env)

        # apply envelope
        signal = signal * envelope

        # optional normalization
        if self.normalize_post and self.target_rms_uv is not None:
            target_rms_v = self.target_rms_uv * 1e-6
            rms = backend.sqrt(backend.mean(signal ** 2, axis=1, keepdims=True)) + 1e-12
            signal = signal * (target_rms_v / rms)

        return BlockOutput(
            data={"signal": signal},
            metadata={
                "envelope": "bursty",
                "rate_per_min": self.rate_per_min,
                "duration_sec": self.duration_sec,
            },
            events=events,
        )
