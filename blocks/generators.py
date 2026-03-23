

"""
Signal generation blocks.
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
from dataclasses import dataclass

from core.types import BlockOutput, BandSpec, Array
from core.backend import Backend, RNGManager


@dataclass
class BandGenerator:
    """Generate multi-partial band-limited signal"""
    band: BandSpec

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        rng = RNGManager(backend, seed=hash(str(key)) % (2**31))

        n_channels: int = context["n_channels"]
        times: Array = backend.asarray(context["times"])
        n_samples = int(times.shape[0]) if hasattr(times, "shape") else len(times)

        sig: Array = backend.zeros((n_channels, n_samples))

        for ch in range(n_channels):
            ch_sig: Array = backend.zeros(n_samples)
            for _ in range(self.band.num_partials):
                # Force RNG scalars to Python floats for clean typing/broadcasting
                partial_freq = float(backend.to_numpy(rng.uniform(self.band.freq_low, self.band.freq_high)))
                partial_phase = float(backend.to_numpy(rng.uniform(0.0, 2.0 * np.pi)))
                ch_sig = ch_sig + backend.sin(2.0 * np.pi * partial_freq * times + partial_phase)

            if backend.name == "jax":
                # JAX arrays support `.at`; NumPy arrays do not.
                sig = sig.at[ch].set(ch_sig)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                sig[ch] = ch_sig

        # Normalize to target RMS
        sig = backend.asarray(sig)
        target_rms_v = float(self.band.amplitude_uv) * 1e-6
        rms: Array = backend.sqrt(backend.mean(sig ** 2, axis=1, keepdims=True)) + 1e-12
        sig = sig * (target_rms_v / rms)

        return BlockOutput(
            data={"signal": sig},
            metadata={
                "band": self.band.name,
                "freq_range": (self.band.freq_low, self.band.freq_high),
                "target_rms_uv": self.band.amplitude_uv,
            },
        )


@dataclass
class ColoredNoise:
    """Generate colored noise (white, pink, brown)"""
    beta: float = 1.0  # 0=white, 1=pink, 2=brown
    rms_uv: float = 2.0

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        rng = RNGManager(backend, seed=hash(str(key)) % (2**31))

        n_channels: int = context["n_channels"]
        n_samples: int = context["n_samples"]
        sfreq: float = float(context["sfreq"])

        white: Array = backend.asarray(rng.normal(0.0, 1.0, size=(n_channels, n_samples)))

        if self.beta == 0:
            colored: Array = white
        else:
            fft: Array = backend.rfft(white, axis=1)
            freqs: Array = backend.rfftfreq(n_samples, 1.0 / sfreq)

            if backend.name == "numpy":
                # Safe handling of DC bin for NumPy
                freqs_safe = np.where(freqs == 0, 1.0, freqs)
            else:
                # JAX-compatible where via backend.xp
                freqs_safe = backend.xp.where(freqs == 0, 1.0, freqs)

            # Ensure array before exponent
            freqs_safe = backend.asarray(freqs_safe)
            shaping: Array = 1.0 / (freqs_safe ** (self.beta / 2.0))

            if backend.name == "jax":
                fft = fft * shaping[None, :]
            else:
                # keep DC as-is; apply shaping from bin 1 onward
                fft[:, 1:] = fft[:, 1:] / (freqs_safe[1:] ** (self.beta / 2.0))

            colored = backend.irfft(fft, n=n_samples, axis=1)

        # Normalize to target RMS
        colored = backend.asarray(colored)
        target_rms_v = float(self.rms_uv) * 1e-6
        rms: Array = backend.sqrt(backend.mean(colored ** 2, axis=1, keepdims=True)) + 1e-12
        colored = colored * (target_rms_v / rms)

        return BlockOutput(
            data={"signal": colored},
            metadata={"beta": self.beta, "rms_uv": self.rms_uv},
        )




