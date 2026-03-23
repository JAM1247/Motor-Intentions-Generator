# pipeline/orchestrator.py
"""
Pipeline orchestration and execution.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.types import Block, BlockOutput, SignalContext
from core.backend import Backend, RNGManager
import numpy as np


@dataclass
class BlockStep:
    """A single block in the pipeline"""
    name: str
    block: Block
    params: Dict[str, Any]
    save_component: bool = False
    component_name: str = ""   # always a str
    accumulate: str = "replace"


class Pipeline:
    """Composable EEG signal processing pipeline
    
    Notes:
    - Non-signal outputs overwrite by key; use unique tags to avoid collisions
    - Signal shape is validated after each block
    - Core parameters (duration, sfreq, n_channels) cannot be overridden
    """

    def __init__(self, backend: str = "numpy", seed: int = 42):
        self.backend_name = backend
        self.backend = Backend(backend)
        self.seed = seed
        self.rng_manager = RNGManager(self.backend, seed)
        self.steps: List[BlockStep] = []
        self.context: Optional[SignalContext] = None

    def add(
        self,
        name: str,
        block: Block,
        save_component: bool = False,
        component_name: Optional[str] = None,
        accumulate: str = "replace",
        **params,
    ):
        """Add a block to the pipeline"""
        step = BlockStep(
            name=name,
            block=block,
            params=params,
            save_component=save_component,
            component_name=(component_name or name),  # now guaranteed str
            accumulate=accumulate,
        )
        self.steps.append(step)
        return self

    def run(
        self,
        duration: float,
        sfreq: float,
        n_channels: int,
        initial_signal=None,
        **initial_context,
    ) -> Dict[str, Any]:
        """Execute the pipeline"""
        # Initialize context
        self.context = SignalContext.create(duration, sfreq, n_channels, self.backend_name)

        # Base metadata useful to downstream UIs / exports
        self.context.metadata.update({
            "duration": duration,
            "sfreq": sfreq,
            "n_channels": n_channels,
            "backend": self.backend_name,
            "seed": self.seed,
        })
        
        # Add user-provided context/metadata, but filter out core fields
        core_fields = {"duration", "sfreq", "n_channels", "n_samples", "times", "nyquist", "backend", "signal", "components"}
        for k, v in initial_context.items():
            if k not in core_fields:
                self.context.metadata[k] = v
            else:
                import warnings
                warnings.warn(f"Ignoring initial_context['{k}'] as it conflicts with core pipeline parameter")

        # Initialize signal accumulator with validation
        if initial_signal is None:
            current_signal = self.backend.zeros((n_channels, self.context.n_samples))
        else:
            x = self.backend.asarray(initial_signal)
            if x.shape != (n_channels, self.context.n_samples):
                raise ValueError(
                    f"initial_signal must have shape {(n_channels, self.context.n_samples)}, got {x.shape}"
                )
            current_signal = x

        # Execute pipeline
        for step in self.steps:
            # Validate accumulate mode early
            if step.accumulate not in ["replace", "add", "multiply"]:
                raise ValueError(
                    f"Invalid accumulate mode '{step.accumulate}' for step '{step.name}'. "
                    f"Must be 'replace', 'add', or 'multiply'."
                )
            
            # Generate unique seed for this step (FIX for RNG)
            step_seed = int(self.rng_manager.integers(0, 2**31 - 1))
            
            # Prepare block context (core params last so they're authoritative)
            block_context = {
                **self.context.metadata,  # user metadata first
                **step.params,             # step params override user metadata
                # Core parameters last - these are authoritative
                "signal": current_signal,
                "duration": duration,
                "sfreq": sfreq,
                "n_channels": n_channels,
                "n_samples": self.context.n_samples,
                "times": self.context.times,
                "nyquist": self.context.nyquist,
                "backend": self.backend_name,
                "components": self.context.components,
            }

            # Execute block with unique seed
            output: BlockOutput = step.block(key=step_seed, context=block_context)

            # Process output
            if "signal" in output.data:
                new_signal = output.data["signal"]
                
                # Validate shape (critical for production robustness)
                expected_shape = current_signal.shape
                if hasattr(new_signal, 'shape'):
                    actual_shape = new_signal.shape
                else:
                    actual_shape = np.asarray(new_signal).shape
                
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"Step '{step.name}' produced signal with shape {actual_shape}, "
                        f"expected {expected_shape}"
                    )

                # Accumulate (validation already done above)
                if step.accumulate == "replace":
                    current_signal = new_signal
                elif step.accumulate == "add":
                    current_signal = current_signal + new_signal
                elif step.accumulate == "multiply":
                    current_signal = current_signal * new_signal

            # Save component if requested
            # Note: save_component only saves the "signal" field from output.data
            # For other outputs, they are automatically saved to components
            if step.save_component and "signal" in output.data:
                self.context.components[step.component_name] = output.data["signal"]

            # Save any additional data from the block (overwrite if key exists)
            for data_key, data_val in output.data.items():
                if data_key != "signal":
                    self.context.components[data_key] = data_val

            # Collect events
            if output.events:
                self.context.events.extend(output.events)

            # Update metadata
            self.context.metadata[f"{step.name}_meta"] = output.metadata

        # Store final signal
        self.context.signals["mixed"] = current_signal

        return {
            "signal": current_signal,
            "components": self.context.components,
            "events": self.context.events,
            "metadata": self.context.metadata,
            "times": self.context.times,
            "sfreq": sfreq,
            "n_channels": n_channels,
        }

    def get_mne_raw(self, result: Dict[str, Any], ch_names: Optional[List[str]] = None):
        """Convert pipeline result to MNE Raw object"""
        import mne

        signal = result["signal"]
        sfreq = result["sfreq"]
        n_channels = result["n_channels"]

        # Convert to numpy if needed
        if self.backend_name == "jax":
            signal = self.backend.to_numpy(signal)

        # Generate channel names
        if ch_names is None:
            ch_names = [f"EEG{i+1:02d}" for i in range(n_channels)]

        # Create MNE info
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        # Create Raw object
        raw = mne.io.RawArray(signal, info, verbose=False)

        # Add annotations from events
        if result["events"]:
            onset = [e["onset"] for e in result["events"]]
            duration = [e.get("duration", 0) for e in result["events"]]
            description = [e["type"] for e in result["events"]]
            raw.set_annotations(mne.Annotations(onset, duration, description))

        return raw