from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from motor_intention.architecture import MotorIntentionResult
from motor_intention.trials import MotorTrial


PHASE_ORDER = ("baseline", "preparation", "imagery", "recovery")


@dataclass(frozen=True)
class RegionSpec:
    key: str
    display_name: str
    dominant_sources: tuple[str, ...]
    preferred_channels: tuple[str, ...]
    expected_pattern: str
    family: str
    side: str | None = None


REGION_SPECS: dict[str, RegionSpec] = {
    "left_arm": RegionSpec(
        "left_arm",
        "Left Arm Motor",
        ("right_upper_limb_motor", "sma", "right_premotor", "right_s1"),
        ("C4", "FC4", "CP4", "Cz", "C2", "FC2", "CP2"),
        "Right-lateralized sensorimotor emphasis with midline support.",
        "arm",
        "left",
    ),
    "right_arm": RegionSpec(
        "right_arm",
        "Right Arm Motor",
        ("left_upper_limb_motor", "sma", "left_premotor", "left_s1"),
        ("C3", "FC3", "CP3", "Cz", "C1", "FC1", "CP1"),
        "Left-lateralized sensorimotor emphasis with midline support.",
        "arm",
        "right",
    ),
    "left_leg": RegionSpec(
        "left_leg",
        "Left Leg Motor",
        ("right_lower_limb_motor", "sma", "right_premotor", "right_s1"),
        ("Cz", "FCz", "CPz", "C2", "FC2", "CP2", "C1"),
        "Medial motor strip emphasis with slight right-of-midline weighting.",
        "leg",
        "left",
    ),
    "right_leg": RegionSpec(
        "right_leg",
        "Right Leg Motor",
        ("left_lower_limb_motor", "sma", "left_premotor", "left_s1"),
        ("Cz", "FCz", "CPz", "C1", "FC1", "CP1", "C2"),
        "Medial motor strip emphasis with slight left-of-midline weighting.",
        "leg",
        "right",
    ),
    "sma": RegionSpec(
        "sma",
        "SMA / Midline Motor",
        ("sma",),
        ("FCz", "Cz", "CPz"),
        "Strong midline motor support centered on the medial strip.",
        "support",
        None,
    ),
    "left_premotor": RegionSpec(
        "left_premotor",
        "Left Premotor Support",
        ("left_premotor",),
        ("FC3", "FC1", "FC5", "C3"),
        "Preparatory emphasis over left premotor cortex.",
        "support",
        "left",
    ),
    "right_premotor": RegionSpec(
        "right_premotor",
        "Right Premotor Support",
        ("right_premotor",),
        ("FC4", "FC2", "FC6", "C4"),
        "Preparatory emphasis over right premotor cortex.",
        "support",
        "right",
    ),
    "rest": RegionSpec(
        "rest",
        "Rest / Background",
        ("posterior_alpha", "frontal_background"),
        ("Cz", "P3", "P4", "FCz"),
        "Symmetric background state with posterior alpha support.",
        "rest",
        None,
    ),
}

SOURCE_FAMILY_COLORS = {
    "upper_limb_motor": "#FF7A59",
    "lower_limb_motor": "#2EC4B6",
    "sma": "#2DD4FF",
    "premotor": "#7DD3FC",
    "somatosensory": "#A78BFA",
    "background": "#94A3B8",
}


def phase_intervals(trial: MotorTrial) -> dict[str, tuple[float, float]]:
    return {
        "baseline": (trial.start_sec, trial.cue_onset_sec),
        "preparation": (trial.cue_onset_sec, trial.imagery_start_sec),
        "imagery": (trial.imagery_start_sec, trial.imagery_end_sec),
        "recovery": (trial.imagery_end_sec, trial.end_sec),
    }


def infer_phase(trial: MotorTrial, time_sec: float) -> str:
    for phase_name, (start_sec, end_sec) in phase_intervals(trial).items():
        if start_sec <= time_sec < end_sec:
            return phase_name
    return "recovery"


def phase_center(trial: MotorTrial, phase_name: str) -> float:
    start_sec, end_sec = phase_intervals(trial)[phase_name]
    return 0.5 * (start_sec + end_sec)


def region_spec(region_key: str) -> RegionSpec:
    return REGION_SPECS[region_key]


def region_keys() -> list[str]:
    return list(REGION_SPECS.keys())


def default_region_for_trial(trial: MotorTrial) -> str:
    if trial.flat_label in REGION_SPECS:
        return trial.flat_label
    return "rest"


def channel_family(result: MotorIntentionResult, channel_name: str) -> str:
    spec = result.layout.spec(channel_name)
    if spec.is_midline and spec.region in {"motor", "premotor", "somatosensory", "frontal"}:
        return "midline"
    return spec.region


def channel_cluster(result: MotorIntentionResult, channel_name: str) -> str:
    spec = result.layout.spec(channel_name)
    if spec.is_midline:
        if spec.region in {"motor", "premotor", "somatosensory", "frontal"}:
            return "midline_motor"
        return "midline"
    if spec.region in {"motor", "somatosensory"}:
        suffix = "sensorimotor" if spec.region == "motor" else "somatosensory"
        return f"{spec.hemisphere}_{suffix}"
    if spec.region == "premotor":
        return f"{spec.hemisphere}_premotor"
    return spec.region


def source_family(source_name: str) -> str:
    if "upper_limb_motor" in source_name:
        return "upper_limb_motor"
    if "lower_limb_motor" in source_name:
        return "lower_limb_motor"
    if source_name == "sma":
        return "sma"
    if "premotor" in source_name:
        return "premotor"
    if source_name.endswith("_s1"):
        return "somatosensory"
    return "background"


def source_strengths_for_phase(
    result: MotorIntentionResult,
    trial: MotorTrial,
    phase_name: str,
) -> np.ndarray:
    start_sec, end_sec = phase_intervals(trial)[phase_name]
    sfreq = float(result.metadata["config_snapshot"]["sfreq"])
    start_idx = int(round(start_sec * sfreq))
    end_idx = int(round(end_sec * sfreq))
    window = result.source_signal[:, start_idx:end_idx]
    return np.sqrt(np.mean(window**2, axis=1)) * 1e6


def rank_electrodes_for_region(
    result: MotorIntentionResult,
    trial: MotorTrial,
    phase_name: str,
    region_key: str,
) -> pd.DataFrame:
    spec = region_spec(region_key)
    strengths = source_strengths_for_phase(result, trial, phase_name)
    source_idx = {
        name: idx for idx, name in enumerate(result.metadata["source_names"])
    }
    active_indices = [source_idx[name] for name in spec.dominant_sources if name in source_idx]
    if not active_indices:
        active_indices = list(range(result.mixing_matrix.shape[1]))

    static_weights = np.mean(result.mixing_matrix[:, active_indices], axis=1)
    dynamic_weights = result.mixing_matrix[:, active_indices] @ strengths[active_indices]
    frame = pd.DataFrame(
        {
            "channel": result.layout.channel_names,
            "static_weight": static_weights,
            "dynamic_weight": dynamic_weights,
            "region": [entry.region for entry in result.layout.specs],
            "hemisphere": [entry.hemisphere for entry in result.layout.specs],
            "is_midline": [entry.is_midline for entry in result.layout.specs],
        }
    )
    frame["cluster_match"] = frame["channel"].isin(spec.preferred_channels)
    frame["channel_family"] = frame["channel"].map(lambda name: channel_family(result, name))
    frame["cluster"] = frame["channel"].map(lambda name: channel_cluster(result, name))
    frame["neighbor_count"] = frame["channel"].map(lambda name: len(result.layout.neighbors(name)))
    frame["static_norm"] = frame["static_weight"] / (float(np.max(frame["static_weight"])) + 1e-9)
    frame["dynamic_norm"] = frame["dynamic_weight"] / (float(np.max(frame["dynamic_weight"])) + 1e-9)
    frame = frame.sort_values(
        by=["cluster_match", "dynamic_weight", "static_weight"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return frame


def default_visible_channels(
    result: MotorIntentionResult,
    trial: MotorTrial,
    phase_name: str,
    region_key: str,
    limit: int = 8,
) -> list[str]:
    frame = rank_electrodes_for_region(result, trial, phase_name, region_key)
    return frame["channel"].head(limit).tolist()


def dominant_sources_for_region(
    result: MotorIntentionResult,
    trial: MotorTrial,
    phase_name: str,
    region_key: str,
) -> pd.DataFrame:
    spec = region_spec(region_key)
    strengths = source_strengths_for_phase(result, trial, phase_name)
    source_idx = {
        name: idx for idx, name in enumerate(result.metadata["source_names"])
    }
    rows: list[dict[str, object]] = []
    for source_name in result.metadata["source_names"]:
        idx = source_idx[source_name]
        rows.append(
            {
                "source": source_name,
                "source_family": source_family(source_name),
                "strength_uv": float(strengths[idx]),
                "selected": source_name in spec.dominant_sources,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        by=["selected", "strength_uv"],
        ascending=[False, False],
    )
    return frame.reset_index(drop=True)


def lateralization_score(
    result: MotorIntentionResult,
    trial: MotorTrial,
    phase_name: str,
) -> float:
    frame = rank_electrodes_for_region(result, trial, phase_name, default_region_for_trial(trial))
    left = frame.loc[frame["hemisphere"] == "left", "dynamic_weight"].head(4).mean()
    right = frame.loc[frame["hemisphere"] == "right", "dynamic_weight"].head(4).mean()
    return float((left - right) / (left + right + 1e-9))


def midline_emphasis_score(
    result: MotorIntentionResult,
    trial: MotorTrial,
    phase_name: str,
) -> float:
    frame = rank_electrodes_for_region(result, trial, phase_name, default_region_for_trial(trial))
    midline = frame.loc[frame["is_midline"], "dynamic_weight"].head(3).mean()
    lateral = frame.loc[~frame["is_midline"], "dynamic_weight"].head(6).mean()
    return float(midline / (lateral + 1e-9))


def trials_for_label(result: MotorIntentionResult, flat_label: str) -> list[MotorTrial]:
    return [trial for trial in result.trials if trial.flat_label == flat_label]


def aggregate_electrode_importance(
    result: MotorIntentionResult,
    flat_label: str,
    phase_name: str = "imagery",
) -> pd.DataFrame:
    matched_trials = trials_for_label(result, flat_label)
    if not matched_trials:
        raise ValueError(f"No trials found for class '{flat_label}'")

    frames = [
        rank_electrodes_for_region(result, trial, phase_name, default_region_for_trial(trial))
        for trial in matched_trials
    ]
    combined = pd.concat(frames, ignore_index=True)
    summary = (
        combined.groupby(
            ["channel", "region", "hemisphere", "is_midline", "channel_family", "cluster"],
            as_index=False,
        )[["static_weight", "dynamic_weight", "static_norm", "dynamic_norm"]]
        .mean()
    )
    summary["n_trials"] = len(matched_trials)
    summary["flat_label"] = flat_label
    summary = summary.sort_values(by=["dynamic_weight", "static_weight"], ascending=[False, False]).reset_index(drop=True)
    return summary


def compare_electrode_importance(
    result: MotorIntentionResult,
    primary_label: str,
    comparison_label: str,
    phase_name: str = "imagery",
) -> pd.DataFrame:
    primary = aggregate_electrode_importance(result, primary_label, phase_name).rename(
        columns={
            "dynamic_weight": "primary_dynamic_weight",
            "static_weight": "primary_static_weight",
        }
    )
    comparison = aggregate_electrode_importance(result, comparison_label, phase_name).rename(
        columns={
            "dynamic_weight": "comparison_dynamic_weight",
            "static_weight": "comparison_static_weight",
        }
    )
    merged = primary.merge(
        comparison[["channel", "comparison_dynamic_weight", "comparison_static_weight"]],
        on="channel",
        how="outer",
    ).fillna(0.0)
    merged["delta_dynamic_weight"] = (
        merged["primary_dynamic_weight"] - merged["comparison_dynamic_weight"]
    )
    merged["abs_delta"] = np.abs(merged["delta_dynamic_weight"])
    return merged.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def class_summary_frame(
    result: MotorIntentionResult,
    phase_name: str = "imagery",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for flat_label in result.metadata["class_schema"]:
        matched_trials = trials_for_label(result, flat_label)
        if not matched_trials:
            continue
        trial_scores = [
            (
                lateralization_score(result, trial, phase_name),
                midline_emphasis_score(result, trial, phase_name),
            )
            for trial in matched_trials
        ]
        epoch_indices = [trial.trial_id for trial in matched_trials]
        sensor_rms_uv = np.sqrt(np.mean(result.epochs[epoch_indices] ** 2, axis=(1, 2))) * 1e6
        rows.append(
            {
                "flat_label": flat_label,
                "n_trials": len(matched_trials),
                "mean_sensor_rms_uv": float(np.mean(sensor_rms_uv)),
                "mean_lateralization": float(np.mean([item[0] for item in trial_scores])),
                "mean_midline_emphasis": float(np.mean([item[1] for item in trial_scores])),
            }
        )
    return pd.DataFrame(rows)


def decoder_readiness_frame(result: MotorIntentionResult) -> pd.DataFrame:
    checks = {
        "labels": len(result.labels) == len(result.trials),
        "epochs": result.epochs.shape[0] == len(result.trials),
        "events": len(result.events) >= len(result.trials) * 3,
        "channel_names": len(result.metadata.get("channel_names", [])) == len(result.layout.channel_names),
        "source_names": len(result.metadata.get("source_names", [])) == result.source_signal.shape[0],
        "reference": bool(result.metadata.get("reference")),
        "projection_info": bool(result.metadata.get("projection_info")),
        "trial_metadata": len(result.trial_metadata) == len(result.trials),
    }
    return pd.DataFrame(
        {
            "component": list(checks.keys()),
            "ready": list(checks.values()),
        }
    )


def class_balance_frame(result: MotorIntentionResult) -> pd.DataFrame:
    frame = pd.DataFrame(result.trial_metadata)
    return (
        frame.groupby("flat_label", as_index=False)
        .size()
        .rename(columns={"size": "n_trials"})
        .sort_values("flat_label")
        .reset_index(drop=True)
    )
