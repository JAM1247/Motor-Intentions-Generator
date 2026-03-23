from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from core.backend import Backend
from core.types import BlockOutput


def _linspace_exclusive(start: float, end: float, length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=float)
    if length == 1:
        return np.asarray([end], dtype=float)
    return np.linspace(start, end, length, endpoint=False, dtype=float)


def _class_profiles() -> dict[str, dict[str, tuple[float, float]]]:
    return {
        "right_arm": {
            "left_upper_limb_motor": (0.42, 1.18),
            "right_upper_limb_motor": (0.96, 1.02),
            "left_lower_limb_motor": (0.98, 1.00),
            "right_lower_limb_motor": (0.99, 1.00),
            "left_s1": (0.56, 1.10),
            "left_premotor": (0.70, 1.05),
            "sma": (0.80, 1.10),
            "right_s1": (0.98, 1.01),
            "right_premotor": (0.96, 1.01),
            "frontal_background": (1.02, 1.00),
            "posterior_alpha": (0.98, 1.00),
        },
        "left_arm": {
            "right_upper_limb_motor": (0.42, 1.18),
            "left_upper_limb_motor": (0.96, 1.02),
            "left_lower_limb_motor": (0.99, 1.00),
            "right_lower_limb_motor": (0.98, 1.00),
            "right_s1": (0.56, 1.10),
            "right_premotor": (0.70, 1.05),
            "sma": (0.80, 1.10),
            "left_s1": (0.98, 1.01),
            "left_premotor": (0.96, 1.01),
            "frontal_background": (1.02, 1.00),
            "posterior_alpha": (0.98, 1.00),
        },
        "right_leg": {
            "left_lower_limb_motor": (0.56, 1.14),
            "right_lower_limb_motor": (0.90, 1.04),
            "left_upper_limb_motor": (0.99, 1.00),
            "right_upper_limb_motor": (0.99, 1.00),
            "sma": (0.68, 1.12),
            "left_premotor": (0.90, 1.03),
            "right_premotor": (0.96, 1.01),
            "left_s1": (0.86, 1.04),
            "right_s1": (0.94, 1.02),
            "frontal_background": (1.01, 1.00),
            "posterior_alpha": (0.99, 1.00),
        },
        "left_leg": {
            "right_lower_limb_motor": (0.56, 1.14),
            "left_lower_limb_motor": (0.90, 1.04),
            "left_upper_limb_motor": (0.99, 1.00),
            "right_upper_limb_motor": (0.99, 1.00),
            "sma": (0.68, 1.12),
            "right_premotor": (0.90, 1.03),
            "left_premotor": (0.96, 1.01),
            "right_s1": (0.86, 1.04),
            "left_s1": (0.94, 1.02),
            "frontal_background": (1.01, 1.00),
            "posterior_alpha": (0.99, 1.00),
        },
        "rest": {
            "left_upper_limb_motor": (0.99, 1.00),
            "right_upper_limb_motor": (0.99, 1.00),
            "left_lower_limb_motor": (0.99, 1.00),
            "right_lower_limb_motor": (0.99, 1.00),
            "left_s1": (0.99, 1.00),
            "right_s1": (0.99, 1.00),
            "sma": (1.00, 1.00),
            "left_premotor": (1.00, 1.00),
            "right_premotor": (1.00, 1.00),
            "frontal_background": (1.00, 1.00),
            "posterior_alpha": (1.05, 1.00),
        },
    }


@dataclass
class HemisphericSourceBalance:
    source_pairs: Sequence[tuple[int, int]]

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        signal = context["signal"]
        signal_np = (
            backend.to_numpy(signal) if backend.name == "jax" else np.asarray(signal, dtype=float)
        )
        balanced = np.asarray(signal_np, dtype=float).copy()

        for left_idx, right_idx in self.source_pairs:
            left_rms = float(np.sqrt(np.mean(balanced[left_idx] ** 2)) + 1e-12)
            right_rms = float(np.sqrt(np.mean(balanced[right_idx] ** 2)) + 1e-12)
            target_rms = 0.5 * (left_rms + right_rms)
            balanced[left_idx] *= target_rms / left_rms
            balanced[right_idx] *= target_rms / right_rms

        if backend.name == "jax":
            balanced = backend.array(balanced)

        return BlockOutput(
            data={"signal": balanced},
            metadata={"modulation": "hemispheric_balance"},
        )



@dataclass
class MotorTaskModulation:
    source_index_map: Mapping[str, int]
    trials: Sequence[Mapping[str, Any]]

    def __call__(self, *, key: Any, context: Dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        signal = context["signal"]
        sfreq = float(context["sfreq"])

        signal_np = (
            backend.to_numpy(signal) if backend.name == "jax" else np.asarray(signal, dtype=float)
        )
        envelope = np.ones_like(signal_np, dtype=float)
        profiles = _class_profiles()

        n_samples = int(signal_np.shape[1])

        def _clip_index(value_sec: float) -> int:
            idx = int(round(float(value_sec) * sfreq))
            return max(0, min(idx, n_samples))

        for trial in self.trials:
            label = str(trial["flat_label"])

            cue_idx = _clip_index(float(trial["cue_onset_sec"]))
            imagery_start_idx = _clip_index(float(trial["imagery_start_sec"]))
            imagery_end_idx = _clip_index(float(trial["imagery_end_sec"]))
            end_idx = _clip_index(float(trial["end_sec"]))

            # enforce monotone ordering after clipping
            cue_idx = min(cue_idx, imagery_start_idx, imagery_end_idx, end_idx)
            imagery_start_idx = max(cue_idx, min(imagery_start_idx, imagery_end_idx, end_idx))
            imagery_end_idx = max(imagery_start_idx, min(imagery_end_idx, end_idx))
            end_idx = max(imagery_end_idx, end_idx)

            prep_len = imagery_start_idx - cue_idx
            recovery_len = end_idx - imagery_end_idx
            recovery_mid = imagery_end_idx + recovery_len // 2

            for source_name, source_idx in self.source_index_map.items():
                imagery_factor, rebound_factor = profiles.get(label, {}).get(source_name, (1.0, 1.0))

                if prep_len > 0:
                    envelope[source_idx, cue_idx:imagery_start_idx] = _linspace_exclusive(
                        1.0, imagery_factor, prep_len
                    )

                if imagery_end_idx > imagery_start_idx:
                    envelope[source_idx, imagery_start_idx:imagery_end_idx] = imagery_factor

                if recovery_len > 0:
                    first_start = imagery_end_idx
                    first_end = recovery_mid
                    second_start = recovery_mid
                    second_end = end_idx

                    first_len = first_end - first_start
                    second_len = second_end - second_start

                    if first_len > 0:
                        envelope[source_idx, first_start:first_end] = _linspace_exclusive(
                            imagery_factor, rebound_factor, first_len
                        )

                    if second_len > 0:
                        envelope[source_idx, second_start:second_end] = _linspace_exclusive(
                            rebound_factor, 1.0, second_len
                        )

        modulated = signal_np * envelope
        if backend.name == "jax":
            modulated = backend.array(modulated)

        return BlockOutput(
            data={"signal": modulated},
            metadata={
                "modulation": "motor_task",
                "n_trials": len(self.trials),
                "class_profiles": sorted(profiles.keys()),
            },
        )
