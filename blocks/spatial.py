
"""
Spatial mixing and referencing blocks.
"""

from typing import Dict, Any
from dataclasses import dataclass

from core.types import BlockOutput
from core.backend import Backend


@dataclass
class Reference:
    """Apply EEG reference"""
    ref_type: str = "average"
    
    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        signal = context["signal"]
        
        if self.ref_type == "average":
            ref = backend.mean(signal, axis=0, keepdims=True)
            signal = signal - ref
        
        return BlockOutput(
            data={"signal": signal},
            metadata={"reference": self.ref_type}
        )
    


