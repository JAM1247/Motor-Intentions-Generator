from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class TrialConfig:
    n_trials: int = 12
    baseline_sec: float = 1.5
    preparation_sec: float = 0.5
    imagery_sec: float = 3.0
    recovery_sec: float = 1.0
    class_cycle: Tuple[str, ...] = (
        "left_arm",
        "right_arm",
        "left_leg",
        "right_leg",
        "rest",
    )

    @property
    def trial_duration_sec(self) -> float:
        return (
            self.baseline_sec
            + self.preparation_sec
            + self.imagery_sec
            + self.recovery_sec
        )


@dataclass(frozen=True)
class LayoutConfig:
    montage_name: str = "motor_21"


@dataclass(frozen=True)
class ProjectionConfig:
    sensor_noise_uv: float = 0.75
    floor_weight: float = 0.02
    normalize_columns: bool = True


@dataclass(frozen=True)
class ExportConfig:
    epoch_start_sec: float = 2.0
    epoch_end_sec: float = 5.0


@dataclass(frozen=True)
class MotorIntentionConfig:
    backend: str = "numpy"
    seed: int = 42
    sfreq: float = 250.0
    reference: str = "average"
    trial: TrialConfig = field(default_factory=TrialConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    include_eog: bool = True
    include_emg: bool = True
    include_line_noise: bool = False
    line_noise_freq_hz: float = 60.0
    line_noise_uv: float = 1.5
    eog_rate_per_min: float = 6.0
    emg_rate_per_min: float = 4.0
    validate_output: bool = True
