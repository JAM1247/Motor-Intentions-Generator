

"""
Core types and protocols for the EEG synthesis framework.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Dict, Any, Optional, List, Any

import numpy as np  

# Accept anything array-like (NumPy, JAX, etc.) to keep the type checker relaxed
Array = Any
Key = Any


@dataclass
class BlockOutput:
    """Standard output from any block"""
    data: Dict[str, Array]
    metadata: Optional[Dict[str, Any]] = None
    events: Optional[List[Dict[str, Any]]] = None


class Block(Protocol):
    """Base protocol for all processing blocks"""
    def __call__(self, *, key: Key, context: Dict[str, Any]) -> BlockOutput:
        ...


@dataclass
class BandSpec:
    """Specification for a frequency band"""
    name: str
    freq_low: float
    freq_high: float
    amplitude_uv: float = 10.0
    num_partials: int = 2
    per_channel_random: bool = True
    phase_mode: str = "random"
    fixed_phase: float = 0.0


@dataclass
class EnvelopeSpec:
    """Specification for envelope modulation"""
    mode: str = "constant"
    normalize_post: bool = False
    params: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}


@dataclass
class SignalContext:
    """Context passed through pipeline"""
    duration: float
    sfreq: float
    n_channels: int
    n_samples: int
    times: Array
    nyquist: float
    signals: Dict[str, Array]
    components: Dict[str, Array]
    events: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    @classmethod
    def create(cls, duration: float, sfreq: float, n_channels: int, backend: str = "numpy") -> "SignalContext":
        n_samples = int(round(duration * sfreq))
        nyquist = 0.5 * sfreq
        if backend == "numpy":
            times = np.arange(n_samples) / sfreq
        else:
            import jax.numpy as jnp
            times = jnp.arange(n_samples) / sfreq
        return cls(
            duration=duration,
            sfreq=sfreq,
            n_channels=n_channels,
            n_samples=n_samples,
            times=times,
            nyquist=nyquist,
            signals={},
            components={},
            events=[],
            metadata={"backend": backend},
        )


