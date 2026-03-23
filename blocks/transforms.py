

"""
Signal transformation blocks.
"""

from typing import Dict, Any
import numpy as np
from dataclasses import dataclass

from core.types import BlockOutput
from core.backend import Backend


@dataclass
class PACModulate:
    """Phase-amplitude coupling between frequency bands"""
    low_band_name: str
    high_band_name: str
    strength: float = 0.5
    coupling_fn: str = "cos"
    
    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        
        low_signal = context["components"].get(self.low_band_name)
        high_signal = context["components"].get(self.high_band_name)
        
        if low_signal is None or high_signal is None:
            raise ValueError(f"PAC requires '{self.low_band_name}' and '{self.high_band_name}' in components")
        
        # Extract instantaneous phase from low frequency (analytic signal)
        low_arr = backend.xp.asarray(low_signal)
        analytic_low = backend.hilbert(low_arr, axis=1)
        phase = backend.xp.angle(analytic_low)

        
        # Create modulation
        if self.coupling_fn == "cos":
            modulation = 1.0 + self.strength * backend.cos(phase)
        else:
            modulation = 1.0 + self.strength * np.abs(backend.cos(phase))
        
        # Apply modulation to high frequency
        high_modulated = high_signal * modulation
        
        # Preserve amplitude
        if backend.name == "numpy":
            scale = (np.linalg.norm(high_signal, axis=1, keepdims=True) + 1e-12) / \
                    (np.linalg.norm(high_modulated, axis=1, keepdims=True) + 1e-12)
        else:
            scale = (backend.xp.linalg.norm(high_signal, axis=1, keepdims=True) + 1e-12) / \
                    (backend.xp.linalg.norm(high_modulated, axis=1, keepdims=True) + 1e-12)
        
        high_modulated = high_modulated * scale
        
        return BlockOutput(
            data={"signal": high_modulated, "pac_phase": phase, "pac_modulation": modulation},
            metadata={
                "pac_low": self.low_band_name,
                "pac_high": self.high_band_name,
                "strength": self.strength
            }
        )




