from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import warnings

import numpy as np

from blocks.artifacts import EMGArtifacts, EOGArtifacts, LineNoise
from blocks.observables import ValidationChecks
from blocks.spatial import Reference
from motor_intention.configs import MotorIntentionConfig
from motor_intention.decoder_export import (
    build_events,
    build_labels,
    build_trial_metadata,
    extract_epochs,
)
from motor_intention.layouts import ElectrodeLayout
from motor_intention.montages import get_layout
from motor_intention.projection import SourceToSensorProjector
from motor_intention.sources import build_source_pipeline
from motor_intention.trials import MotorTrial, build_trial_schedule, total_duration
from pipeline.orchestrator import Pipeline


@dataclass
class MotorIntentionResult:
    source_signal: np.ndarray
    sensor_signal: np.ndarray
    source_result: dict[str, Any]
    sensor_result: dict[str, Any]
    epochs: np.ndarray
    epoch_times: np.ndarray
    labels: np.ndarray
    trial_metadata: list[dict[str, Any]]
    events: list[dict[str, Any]]
    trials: list[MotorTrial]
    layout: ElectrodeLayout
    mixing_matrix: np.ndarray
    metadata: dict[str, Any]


class MotorIntentionArchitecture:
    def __init__(self, config: MotorIntentionConfig | None = None):
        self.config = config or MotorIntentionConfig()

    def run(self) -> MotorIntentionResult:
        trials = build_trial_schedule(self.config.trial)
        duration = total_duration(trials)

        source_pipeline, source_names = build_source_pipeline(self.config, trials)
        source_result = source_pipeline.run(
            duration=duration,
            sfreq=self.config.sfreq,
            n_channels=len(source_names),
            source_names=source_names,
            trial_schedule=[trial.to_dict() for trial in trials],
        )
        source_signal = np.asarray(source_result["signal"], dtype=float)

        layout = get_layout(self.config.layout.montage_name)
        self._warn_on_reduced_layout(layout)
        projector = SourceToSensorProjector(
            layout=layout,
            source_names=source_names,
            config=self.config.projection,
            backend_name=self.config.backend,
            seed=self.config.seed + 1000,
        )
        projected_signal, mixing_matrix = projector.project(source_signal)

        sensor_pipeline = self._build_sensor_pipeline(layout)
        sensor_result = sensor_pipeline.run(
            duration=duration,
            sfreq=self.config.sfreq,
            n_channels=len(layout),
            initial_signal=projected_signal,
            trial_schedule=[trial.to_dict() for trial in trials],
            channel_names=layout.channel_names,
            montage_name=layout.name,
            source_names=source_names,
        )
        sensor_signal = np.asarray(sensor_result["signal"], dtype=float)

        labels = build_labels(trials)
        trial_metadata = build_trial_metadata(trials)
        events = build_events(trials) + list(sensor_result.get("events", []))
        epochs, epoch_times = extract_epochs(
            sensor_signal,
            trials,
            sfreq=self.config.sfreq,
            start_sec=self.config.export.epoch_start_sec,
            end_sec=self.config.export.epoch_end_sec,
        )

        metadata = {
            "generator_version": "motor_intention_v1",
            "class_schema": list(self.config.trial.class_cycle),
            "flat_labels": labels.tolist(),
            "trial_metadata": trial_metadata,
            "layout_name": layout.name,
            "channel_names": layout.channel_names,
            "source_names": source_names,
            "reference": self.config.reference,
            "projection_info": {
                "sensor_noise_uv": self.config.projection.sensor_noise_uv,
                "floor_weight": self.config.projection.floor_weight,
                "normalize_columns": self.config.projection.normalize_columns,
            },
            "config_snapshot": asdict(self.config),
            "source_metadata": source_result["metadata"],
            "sensor_metadata": sensor_result["metadata"],
        }

        return MotorIntentionResult(
            source_signal=source_signal,
            sensor_signal=sensor_signal,
            source_result=source_result,
            sensor_result=sensor_result,
            epochs=epochs,
            epoch_times=epoch_times,
            labels=labels,
            trial_metadata=trial_metadata,
            events=events,
            trials=trials,
            layout=layout,
            mixing_matrix=mixing_matrix,
            metadata=metadata,
        )

    def _build_sensor_pipeline(self, layout: ElectrodeLayout) -> Pipeline:
        pipe = Pipeline(backend=self.config.backend, seed=self.config.seed + 1)

        if self.config.include_eog:
            pipe.add(
                "eog",
                EOGArtifacts(
                    rate_per_min=self.config.eog_rate_per_min,
                    channels=self._default_eog_channel_indices(layout),
                ),
            )
        if self.config.include_emg:
            nyquist = self.config.sfreq / 2.0
            emg_high = max(30.0, min(110.0, nyquist - 5.0))
            pipe.add(
                "emg",
                EMGArtifacts(
                    rate_per_min=self.config.emg_rate_per_min,
                    band_hz=(20.0, emg_high),
                ),
            )
        if self.config.include_line_noise:
            pipe.add(
                "line_noise",
                LineNoise(
                    freq_hz=self.config.line_noise_freq_hz,
                    amplitude_uv=self.config.line_noise_uv,
                ),
            )

        pipe.add("reference", Reference(ref_type=self.config.reference))

        if self.config.validate_output:
            pipe.add("validation", ValidationChecks())

        return pipe

    @staticmethod
    def _default_eog_channel_indices(layout: ElectrodeLayout) -> list[int]:
        preferred = [name for name in ("FC3", "FC4", "Fz") if name in layout.channel_names]
        if len(preferred) >= 2:
            return layout.indices(preferred[:2])
        return layout.indices(layout.frontal_channels(2))

    @staticmethod
    def _warn_on_reduced_layout(layout: ElectrodeLayout) -> None:
        if layout.name == "motor_14":
            warnings.warn(
                "motor_14 is supported, but it provides less medial coverage for lower-limb separation than motor_21.",
                stacklevel=2,
            )


def simulate_motor_dataset(
    config: MotorIntentionConfig | None = None,
    **overrides: Any,
) -> MotorIntentionResult:
    if config is None:
        config = MotorIntentionConfig(**overrides)
    elif overrides:
        raise ValueError("Pass either a config object or constructor overrides, not both")
    return MotorIntentionArchitecture(config).run()
