from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from blocks.envelopes import BurstyEnvelope
from blocks.generators import BandGenerator, ColoredNoise
from core.backend import Backend
from core.types import BandSpec, BlockOutput
from motor_intention.configs import MotorIntentionConfig
from motor_intention.modulation import HemisphericSourceBalance, MotorTaskModulation
from motor_intention.trials import MotorTrial
from pipeline.orchestrator import Pipeline


@dataclass(frozen=True)
class BandComponent:
    name: str
    low_hz: float
    high_hz: float
    amplitude_uv: float
    num_partials: int = 2


@dataclass(frozen=True)
class SourceDefinition:
    name: str
    region: str
    center_x: float
    center_y: float
    spread: float
    bands: tuple[BandComponent, ...]
    noise_uv: float = 0.0
    bursty: bool = False
    burst_rate_per_min: float = 10.0


@dataclass
class SourceScopedBlock:
    inner_block: Any
    source_indices: tuple[int, ...]
    full_n_channels: int
    tag: str
    merge_with_input: bool = False

    def __call__(self, *, key: Any, context: dict[str, Any]) -> BlockOutput:
        backend = Backend(context.get("backend", "numpy"))
        signal = context["signal"]
        if backend.name == "jax":
            signal_np = np.asarray(backend.to_numpy(signal))
        else:
            signal_np = np.asarray(signal)

        subset = signal_np[list(self.source_indices)]
        scoped_context = dict(context)
        scoped_context["signal"] = backend.array(subset) if backend.name == "jax" else subset
        scoped_context["n_channels"] = len(self.source_indices)

        output = self.inner_block(key=key, context=scoped_context)
        data: dict[str, Any] = {}
        for name, value in output.data.items():
            data[name] = self._expand_output(value, backend, signal_np)

        metadata = dict(output.metadata or {})
        metadata["scoped_block"] = self.tag
        metadata["scoped_sources"] = list(self.source_indices)
        return BlockOutput(data=data, metadata=metadata, events=output.events)

    def _expand_output(self, value: Any, backend: Backend, base_signal: np.ndarray) -> Any:
        if backend.name == "jax":
            value_np = np.asarray(backend.to_numpy(value))
        else:
            value_np = np.asarray(value)

        if value_np.ndim >= 1 and value_np.shape[0] == len(self.source_indices):
            if self.merge_with_input:
                expanded = np.asarray(base_signal, dtype=value_np.dtype).copy()
            else:
                expanded = np.zeros((self.full_n_channels, *value_np.shape[1:]), dtype=value_np.dtype)
            for row_idx, source_idx in enumerate(self.source_indices):
                expanded[source_idx] = value_np[row_idx]
            return backend.array(expanded) if backend.name == "jax" else expanded
        return value


SOURCE_DEFINITIONS = (
    SourceDefinition(
        "left_upper_limb_motor",
        "motor",
        -0.52,
        0.00,
        0.20,
        (
            BandComponent("left_upper_mu", 8.0, 12.0, 18.0, 3),
            BandComponent("left_upper_beta", 18.0, 26.0, 7.0, 2),
        ),
        noise_uv=1.2,
    ),
    SourceDefinition(
        "right_upper_limb_motor",
        "motor",
        0.52,
        0.00,
        0.20,
        (
            BandComponent("right_upper_mu", 8.0, 12.0, 18.0, 3),
            BandComponent("right_upper_beta", 18.0, 26.0, 7.0, 2),
        ),
        noise_uv=1.2,
    ),
    SourceDefinition(
        "left_lower_limb_motor",
        "motor",
        -0.14,
        0.03,
        0.15,
        (
            BandComponent("left_lower_mu", 8.0, 12.0, 14.0, 3),
            BandComponent("left_lower_beta", 16.0, 24.0, 6.0, 2),
        ),
        noise_uv=0.9,
    ),
    SourceDefinition(
        "right_lower_limb_motor",
        "motor",
        0.14,
        0.03,
        0.15,
        (
            BandComponent("right_lower_mu", 8.0, 12.0, 14.0, 3),
            BandComponent("right_lower_beta", 16.0, 24.0, 6.0, 2),
        ),
        noise_uv=0.9,
    ),
    SourceDefinition(
        "sma",
        "motor",
        0.00,
        0.08,
        0.16,
        (
            BandComponent("sma_mu", 9.0, 13.0, 10.0, 2),
            BandComponent("sma_beta", 16.0, 24.0, 8.0, 2),
        ),
        noise_uv=0.8,
    ),
    SourceDefinition(
        "left_premotor",
        "premotor",
        -0.42,
        0.28,
        0.22,
        (
            BandComponent("left_premotor_theta", 4.0, 7.0, 4.0, 2),
            BandComponent("left_premotor_beta", 15.0, 22.0, 6.0, 2),
        ),
        noise_uv=0.8,
    ),
    SourceDefinition(
        "right_premotor",
        "premotor",
        0.42,
        0.28,
        0.22,
        (
            BandComponent("right_premotor_theta", 4.0, 7.0, 4.0, 2),
            BandComponent("right_premotor_beta", 15.0, 22.0, 6.0, 2),
        ),
        noise_uv=0.8,
    ),
    SourceDefinition(
        "left_s1",
        "somatosensory",
        -0.38,
        -0.18,
        0.22,
        (
            BandComponent("left_s1_mu", 8.5, 12.5, 10.0, 2),
            BandComponent("left_s1_beta", 16.0, 24.0, 5.0, 2),
        ),
        noise_uv=0.8,
    ),
    SourceDefinition(
        "right_s1",
        "somatosensory",
        0.38,
        -0.18,
        0.22,
        (
            BandComponent("right_s1_mu", 8.5, 12.5, 10.0, 2),
            BandComponent("right_s1_beta", 16.0, 24.0, 5.0, 2),
        ),
        noise_uv=0.8,
    ),
    SourceDefinition(
        "frontal_background",
        "frontal",
        0.00,
        0.44,
        0.55,
        (
            BandComponent("frontal_theta", 4.0, 7.0, 8.0, 2),
            BandComponent("frontal_beta", 13.0, 20.0, 3.0, 2),
        ),
        noise_uv=1.8,
    ),
    SourceDefinition(
        "posterior_alpha",
        "occipital",
        0.00,
        -0.68,
        0.32,
        (BandComponent("posterior_alpha", 8.0, 12.0, 20.0, 3),),
        noise_uv=1.0,
        bursty=True,
        burst_rate_per_min=10.0,
    ),
)


def source_names() -> list[str]:
    return [source.name for source in SOURCE_DEFINITIONS]


def source_index_map() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(source_names())}


def source_definition_map() -> dict[str, SourceDefinition]:
    return {source.name: source for source in SOURCE_DEFINITIONS}


def _scoped_row(index: int) -> tuple[int, ...]:
    return (index,)


def build_source_pipeline(
    config: MotorIntentionConfig,
    trials: Iterable[MotorTrial],
) -> tuple[Pipeline, list[str]]:
    names = source_names()
    n_sources = len(names)
    source_to_idx = source_index_map()
    pipeline = Pipeline(backend=config.backend, seed=config.seed)

    for source in SOURCE_DEFINITIONS:
        idx = source_to_idx[source.name]
        for component in source.bands:
            generator = BandGenerator(
                BandSpec(
                    name=component.name,
                    freq_low=component.low_hz,
                    freq_high=component.high_hz,
                    amplitude_uv=component.amplitude_uv,
                    num_partials=component.num_partials,
                )
            )
            pipeline.add(
                f"{source.name}_{component.name}",
                SourceScopedBlock(
                    inner_block=generator,
                    source_indices=_scoped_row(idx),
                    full_n_channels=n_sources,
                    tag=f"{source.name}_{component.name}",
                ),
                accumulate="add",
            )

        if source.noise_uv > 0:
            pipeline.add(
                f"{source.name}_noise",
                SourceScopedBlock(
                    inner_block=ColoredNoise(beta=1.0, rms_uv=source.noise_uv),
                    source_indices=_scoped_row(idx),
                    full_n_channels=n_sources,
                    tag=f"{source.name}_noise",
                ),
                accumulate="add",
            )

        if source.bursty:
            pipeline.add(
                f"{source.name}_bursts",
                SourceScopedBlock(
                    inner_block=BurstyEnvelope(
                        rate_per_min=source.burst_rate_per_min,
                        duration_sec=1.8,
                        amp_factor=1.5,
                        baseline_factor=0.65,
                        normalize_post=True,
                        target_rms_uv=source.bands[0].amplitude_uv,
                    ),
                    source_indices=_scoped_row(idx),
                    full_n_channels=n_sources,
                    tag=f"{source.name}_bursts",
                    merge_with_input=True,
                ),
            )

    pipeline.add(
        "hemispheric_balance",
        HemisphericSourceBalance(
            source_pairs=(
                (
                    source_to_idx["left_upper_limb_motor"],
                    source_to_idx["right_upper_limb_motor"],
                ),
                (
                    source_to_idx["left_lower_limb_motor"],
                    source_to_idx["right_lower_limb_motor"],
                ),
                (source_to_idx["left_premotor"], source_to_idx["right_premotor"]),
                (source_to_idx["left_s1"], source_to_idx["right_s1"]),
            )
        ),
    )

    pipeline.add(
        "motor_task_modulation",
        MotorTaskModulation(
            source_index_map=source_to_idx,
            trials=[trial.to_dict() for trial in trials],
        ),
    )

    return pipeline, names
