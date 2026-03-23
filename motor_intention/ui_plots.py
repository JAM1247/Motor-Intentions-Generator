from __future__ import annotations

from io import BytesIO
import json
from typing import Sequence

import altair as alt
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse, Polygon
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from motor_intention.architecture import MotorIntentionResult
from motor_intention.sources import source_definition_map
from motor_intention.trials import MotorTrial
from motor_intention.ui_logic import (
    PHASE_ORDER,
    class_summary_frame,
    compare_electrode_importance,
    decoder_readiness_frame,
    default_region_for_trial,
    dominant_sources_for_region,
    phase_center,
    phase_intervals,
    rank_electrodes_for_region,
    region_spec,
    source_strengths_for_phase,
)

PHASE_COLORS: dict[str, str] = {
    "baseline": "#CAD2D3",
    "preparation": "#F0B44C",
    "imagery": "#D95D39",
    "recovery": "#5C80BC",
}
CLASS_COLORS: dict[str, str] = {
    "left_arm": "#E76F51",
    "right_arm": "#F4A261",
    "left_leg": "#2A9D8F",
    "right_leg": "#457B9D",
    "rest": "#8D99AE",
}
TRIANGULATION_CACHE: dict[tuple[str, tuple[str, ...]], mtri.Triangulation] = {}


def build_timeline_chart(
    trials: Sequence[MotorTrial],
    selected_trial_id: int | None = None,
) -> alt.Chart:
    rows: list[dict[str, object]] = []
    ordered_trials = list(trials)
    for trial in ordered_trials:
        intervals = phase_intervals(trial)
        for phase_name in PHASE_ORDER:
            start_sec, end_sec = intervals[phase_name]
            rows.append(
                {
                    "trial": f"Trial {trial.trial_id}",
                    "trial_id": trial.trial_id,
                    "phase": phase_name,
                    "start": float(start_sec),
                    "end": float(end_sec),
                    "flat_label": trial.flat_label,
                    "selected": bool(trial.trial_id == selected_trial_id),
                }
            )

    frame = pd.DataFrame(rows)
    base = alt.Chart(
        frame
        if not frame.empty
        else pd.DataFrame({"start": [], "end": [], "trial": [], "phase": []})
    )
    if frame.empty:
        return base.mark_bar().properties(height=180)

    return (
        base.mark_bar(
            cornerRadiusEnd=3,
            cornerRadiusTopLeft=3,
            cornerRadiusBottomLeft=3,
        )
        .encode(
            x=alt.X("start:Q", title="Time (s)"),
            x2="end:Q",
            y=alt.Y(
                "trial:N",
                sort=[f"Trial {trial.trial_id}" for trial in ordered_trials],
                title="Trial",
            ),
            color=alt.Color(
                "phase:N",
                scale=alt.Scale(
                    domain=list(PHASE_COLORS.keys()),
                    range=list(PHASE_COLORS.values()),
                ),
                title="Phase",
            ),
            opacity=alt.condition(alt.datum.selected, alt.value(1.0), alt.value(0.55)),
            tooltip=["trial_id:Q", "flat_label:N", "phase:N", "start:Q", "end:Q"],
        )
        .properties(height=max(180, 28 * len(ordered_trials)))
    )


def plot_source_strength_bars(
    result: MotorIntentionResult,
    trial: MotorTrial,
    phase_name: str,
    region_key: str | None = None,
) -> Figure:
    active_region = region_key or default_region_for_trial(trial)
    frame = dominant_sources_for_region(
        result,
        trial,
        _validate_phase(phase_name),
        active_region,
    ).head(8)

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    fig.patch.set_facecolor("#0B1730")
    ax.set_facecolor("#0B1730")

    colors = [
        "#FF7A59"
        if bool(row.selected)
        else "#2DD4FF"
        if "premotor" in str(row.source)
        else "#2EC4B6"
        for row in frame.itertuples()
    ]

    ax.barh(frame["source"], frame["strength_uv"], color=colors, alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("RMS (uV)", color="#D6E3F0")
    ax.set_title(
        f"Source Stack · {region_spec(active_region).display_name}",
        color="#D6E3F0",
    )
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_source_family_stack(
    result: MotorIntentionResult,
    trial: MotorTrial,
    phase_name: str,
    region_key: str,
) -> Figure:
    frame = dominant_sources_for_region(
        result,
        trial,
        _validate_phase(phase_name),
        region_key,
    )
    grouped = frame.groupby("source_family", as_index=False).agg(
        strength_uv=("strength_uv", "sum")
    )
    grouped = grouped.nlargest(len(grouped), columns="strength_uv")

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    fig.patch.set_facecolor("#0B1730")
    ax.set_facecolor("#0B1730")

    colors = [
        "#FF7A59"
        if name == "upper_limb_motor"
        else "#2EC4B6"
        if name == "lower_limb_motor"
        else "#2DD4FF"
        if name == "sma"
        else "#7DD3FC"
        if name == "premotor"
        else "#A78BFA"
        if name == "somatosensory"
        else "#94A3B8"
        for name in grouped["source_family"]
    ]
    ax.bar(grouped["source_family"], grouped["strength_uv"], color=colors, alpha=0.92)
    ax.set_ylabel("RMS (uV)", color="#D6E3F0")
    ax.set_title("Source Family Mix", color="#D6E3F0", fontsize=11)
    _style_axes(ax, rotate_x=18)
    fig.tight_layout()
    return fig


def plot_brain_console(
    result: MotorIntentionResult,
    trial: MotorTrial,
    phase_name: str,
    region_key: str,
) -> Figure:
    strengths = source_strengths_for_phase(result, trial, _validate_phase(phase_name))
    definitions = source_definition_map()
    active_region = region_spec(region_key)
    max_strength = float(np.max(strengths) + 1e-9)

    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    fig.patch.set_facecolor("#071122")
    ax.set_facecolor("#071122")

    ax.add_patch(
        Ellipse(
            (-0.22, 0.0),
            width=0.86,
            height=1.18,
            facecolor="#0B1730",
            edgecolor="#18314F",
            linewidth=2,
        )
    )
    ax.add_patch(
        Ellipse(
            (0.22, 0.0),
            width=0.86,
            height=1.18,
            facecolor="#0B1730",
            edgecolor="#18314F",
            linewidth=2,
        )
    )
    ax.add_patch(
        Ellipse(
            (0.0, 0.58),
            width=0.18,
            height=0.10,
            facecolor="#102341",
            edgecolor="#18314F",
            linewidth=1.2,
        )
    )

    overlay_lookup = {
        "left_arm": Ellipse(
            (0.34, 0.04),
            0.28,
            0.26,
            facecolor="#FF7A59",
            alpha=0.18,
            edgecolor="#FF7A59",
            linewidth=2.2,
        ),
        "right_arm": Ellipse(
            (-0.34, 0.04),
            0.28,
            0.26,
            facecolor="#FF7A59",
            alpha=0.18,
            edgecolor="#FF7A59",
            linewidth=2.2,
        ),
        "left_leg": Ellipse(
            (0.10, 0.02),
            0.26,
            0.22,
            facecolor="#2EC4B6",
            alpha=0.18,
            edgecolor="#2EC4B6",
            linewidth=2.2,
        ),
        "right_leg": Ellipse(
            (-0.10, 0.02),
            0.26,
            0.22,
            facecolor="#2EC4B6",
            alpha=0.18,
            edgecolor="#2EC4B6",
            linewidth=2.2,
        ),
        "sma": Ellipse(
            (0.0, 0.18),
            0.18,
            0.20,
            facecolor="#2DD4FF",
            alpha=0.22,
            edgecolor="#2DD4FF",
            linewidth=2.0,
        ),
        "left_premotor": Ellipse(
            (0.26, 0.30),
            0.24,
            0.16,
            facecolor="#2DD4FF",
            alpha=0.18,
            edgecolor="#2DD4FF",
            linewidth=2.0,
        ),
        "right_premotor": Ellipse(
            (-0.26, 0.30),
            0.24,
            0.16,
            facecolor="#2DD4FF",
            alpha=0.18,
            edgecolor="#2DD4FF",
            linewidth=2.0,
        ),
        "rest": Circle(
            (0.0, -0.15),
            0.16,
            facecolor="#94A3B8",
            alpha=0.15,
            edgecolor="#8FA7BF",
            linewidth=1.6,
        ),
    }
    ax.add_patch(overlay_lookup[region_key])

    for index, source_name in enumerate(result.metadata["source_names"]):
        definition = definitions[source_name]
        alpha = 0.3 + 0.7 * (float(strengths[index]) / max_strength)
        color = (
            "#D95D39"
            if "upper_limb" in source_name
            else "#2A9D8F"
            if "lower_limb" in source_name
            else "#5C80BC"
        )
        ax.scatter(
            definition.center_x,
            definition.center_y,
            s=340 * alpha,
            c=color,
            alpha=alpha,
            edgecolors="#D6E3F0"
            if source_name in active_region.dominant_sources
            else "#1F2937",
            linewidths=1.4 if source_name in active_region.dominant_sources else 0.8,
        )
        ax.text(
            definition.center_x,
            definition.center_y + 0.05,
            source_name.replace("_", "\n"),
            ha="center",
            va="bottom",
            fontsize=7.6,
            color="#D6E3F0",
        )

    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.75, 0.75)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Cortex Console · {active_region.display_name}",
        color="#D6E3F0",
        fontsize=14,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def default_sensor_channels(result: MotorIntentionResult) -> list[str]:
    preferred = ["FC3", "C3", "Cz", "C4", "FC4", "CP3", "CPz", "CP4"]
    selected = [name for name in preferred if name in result.layout.channel_names]
    return selected or result.layout.channel_names[: min(8, len(result.layout.channel_names))]


def plot_signal_wall_figure(
    result: MotorIntentionResult,
    trial: MotorTrial,
    channel_names: Sequence[str],
    view_mode: str = "Selected Epoch",
    *,
    playhead_sec: float | None = None,
    window_sec: float = 2.0,
    live_mode: str = "Selected Trial",
) -> Figure:
    unique_channels = [
        name
        for name in dict.fromkeys(channel_names)
        if name in result.layout.channel_names
    ]
    if not unique_channels:
        fig, ax = plt.subplots(figsize=(6.0, 2.5))
        fig.patch.set_facecolor("#0B1730")
        ax.set_facecolor("#0B1730")
        ax.text(
            0.5,
            0.5,
            "No channels selected",
            ha="center",
            va="center",
            color="#D6E3F0",
        )
        ax.axis("off")
        return fig

    trace_bundle, cursor_x = _signal_trace_bundle(
        result,
        trial,
        unique_channels,
        view_mode,
        playhead_sec=playhead_sec,
        window_sec=window_sec,
        live_mode=live_mode,
    )

    n_channels = len(unique_channels)
    ncols = 2 if n_channels <= 4 else 3 if n_channels <= 6 else 4
    nrows = int(np.ceil(n_channels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 1.95 * nrows),
        squeeze=False,
    )
    fig.patch.set_facecolor("#0B1730")
    axes_flat = list(axes.ravel())

    for ax, channel_name in zip(axes_flat, unique_channels):
        x_values, y_values, suffix, trace_color = trace_bundle[channel_name]
        ax.set_facecolor("#0B1730")
        ax.plot(x_values, y_values, color=trace_color, lw=1.05)
        if cursor_x is not None and len(x_values):
            ax.axvline(cursor_x, color="#F0B44C", lw=1.0, alpha=0.95)
        ax.axhline(0.0, color="#36557A", lw=0.8, alpha=0.7)
        ax.set_title(f"{channel_name} · {suffix}", color="#D6E3F0", fontsize=9, pad=4)
        _style_axes(ax, label_size=6.5)

    for ax in axes_flat[n_channels:]:
        ax.set_facecolor("#0B1730")
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_electrode_ranking_table(
    result: MotorIntentionResult,
    trial: MotorTrial,
    phase_name: str,
    region_key: str,
    top_n: int = 8,
) -> pd.DataFrame:
    frame = rank_electrodes_for_region(
        result,
        trial,
        _validate_phase(phase_name),
        region_key,
    ).head(top_n).copy()
    frame["dynamic_weight"] = frame["dynamic_weight"].round(2)
    frame["static_weight"] = frame["static_weight"].round(3)
    return frame


def plot_electrode_cluster_map(
    result: MotorIntentionResult,
    highlight_channels: Sequence[str] | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    fig.patch.set_facecolor("#0B1730")
    ax.set_facecolor("#0B1730")

    positions = result.layout.coordinates
    highlight_set = {
        name for name in (highlight_channels or []) if name in result.layout.channel_names
    }
    cluster_colors = {
        "left_sensorimotor": "#FF7A59",
        "right_sensorimotor": "#F4A261",
        "midline_motor": "#2DD4FF",
        "left_premotor": "#7DD3FC",
        "right_premotor": "#38BDF8",
        "left_somatosensory": "#A78BFA",
        "right_somatosensory": "#C084FC",
    }

    if result.trials:
        ranking = rank_electrodes_for_region(
            result,
            result.trials[0],
            "imagery",
            default_region_for_trial(result.trials[0]),
        )
        cluster_lookup = dict(zip(ranking["channel"], ranking["cluster"]))
    else:
        cluster_lookup = {}

    for x_coord, y_coord, channel_name in zip(
        positions[:, 0],
        positions[:, 1],
        result.layout.channel_names,
    ):
        cluster_name = str(
            cluster_lookup.get(channel_name, result.layout.spec(channel_name).region)
        )
        color = cluster_colors.get(cluster_name, "#64748B")
        size = 120 if channel_name in highlight_set else 70
        edge = "#F8FAFC" if channel_name in highlight_set else "#0F172A"
        ax.scatter(
            x_coord,
            y_coord,
            s=size,
            c=color,
            edgecolors=edge,
            linewidths=1.2,
            zorder=4,
        )
        if channel_name in highlight_set:
            ax.text(
                x_coord,
                y_coord + 0.05,
                channel_name,
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#D6E3F0",
            )

    _add_head_outline(ax)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.05, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Electrode Cluster Map", color="#D6E3F0")
    fig.tight_layout()
    return fig


def plot_topography(
    result: MotorIntentionResult,
    trial: MotorTrial,
    snapshot_sec: float,
    highlight_channels: Sequence[str] | None = None,
) -> Figure:
    sfreq = float(result.metadata["config_snapshot"]["sfreq"])
    sample_index = int(
        np.clip(
            int(round(snapshot_sec * sfreq)),
            0,
            result.sensor_signal.shape[1] - 1,
        )
    )
    values_uv = result.sensor_signal[:, sample_index] * 1e6
    return _plot_topography_from_values(
        result,
        values_uv,
        title=f"Scalp Snapshot at {snapshot_sec:.2f}s",
        highlight_channels=highlight_channels,
    )


def plot_class_topography(
    result: MotorIntentionResult,
    flat_label: str,
    phase_name: str = "imagery",
    highlight_channels: Sequence[str] | None = None,
) -> Figure:
    phase = _validate_phase(phase_name)
    label = _validate_class_label(result, flat_label)
    values_uv = _class_snapshot_values(result, label, phase) * 1e6
    return _plot_topography_from_values(
        result,
        values_uv,
        title=f"{label.replace('_', ' ').title()} · Mean {phase.title()} Topography",
        highlight_channels=highlight_channels,
    )


def plot_topography_difference(
    result: MotorIntentionResult,
    primary_label: str,
    comparison_label: str,
    phase_name: str = "imagery",
) -> Figure:
    phase = _validate_phase(phase_name)
    first_label = _validate_class_label(result, primary_label)
    second_label = _validate_class_label(result, comparison_label)
    delta_uv = (
        _class_snapshot_values(result, first_label, phase)
        - _class_snapshot_values(result, second_label, phase)
    ) * 1e6
    return _plot_topography_from_values(
        result,
        delta_uv,
        title=f"{first_label.replace('_', ' ').title()} minus {second_label.replace('_', ' ').title()}",
    )


def plot_projection_matrix(result: MotorIntentionResult) -> Figure:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    fig.patch.set_facecolor("#0B1730")
    ax.set_facecolor("#0B1730")
    image = ax.imshow(result.mixing_matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(result.metadata["source_names"])))
    ax.set_xticklabels(
        result.metadata["source_names"],
        rotation=45,
        ha="right",
        fontsize=8,
        color="#D6E3F0",
    )
    ax.set_yticks(range(len(result.layout.channel_names)))
    ax.set_yticklabels(result.layout.channel_names, fontsize=8, color="#D6E3F0")
    ax.set_title("Source-to-Sensor Mixing Matrix", color="#D6E3F0")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85, label="Weight")
    colorbar.ax.tick_params(colors="#D6E3F0")
    colorbar.ax.yaxis.label.set_color("#D6E3F0")
    _style_axes(ax, label_size=8)
    fig.tight_layout()
    return fig


def plot_class_summary_bars(
    result: MotorIntentionResult,
    phase_name: str = "imagery",
) -> Figure:
    frame = class_summary_frame(result, _validate_phase(phase_name))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2))
    fig.patch.set_facecolor("#0B1730")
    metrics = [
        ("mean_sensor_rms_uv", "Mean RMS (uV)"),
        ("mean_lateralization", "Mean Lateralization"),
        ("mean_midline_emphasis", "Midline Emphasis"),
    ]

    if frame.empty:
        for ax in axes:
            ax.set_facecolor("#0B1730")
            ax.text(0.5, 0.5, "No class data", ha="center", va="center", color="#D6E3F0")
            ax.axis("off")
        fig.tight_layout()
        return fig

    colors = [CLASS_COLORS.get(label, "#D6E3F0") for label in frame["flat_label"]]
    for ax, (column_name, title) in zip(axes, metrics):
        ax.set_facecolor("#0B1730")
        ax.bar(frame["flat_label"], frame[column_name], color=colors, alpha=0.9)
        ax.set_title(title, color="#D6E3F0", fontsize=10)
        _style_axes(ax, rotate_x=25, label_size=8)
    fig.tight_layout()
    return fig


def plot_channel_delta_bars(
    result: MotorIntentionResult,
    primary_label: str,
    comparison_label: str,
    phase_name: str = "imagery",
    top_n: int = 8,
) -> Figure:
    phase = _validate_phase(phase_name)
    first_label = _validate_class_label(result, primary_label)
    second_label = _validate_class_label(result, comparison_label)
    frame = compare_electrode_importance(
        result,
        first_label,
        second_label,
        phase,
    ).head(top_n)

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    fig.patch.set_facecolor("#0B1730")
    ax.set_facecolor("#0B1730")

    colors = [
        "#FF7A59" if value >= 0 else "#2EC4B6"
        for value in frame["delta_dynamic_weight"]
    ]
    ax.barh(frame["channel"], frame["delta_dynamic_weight"], color=colors, alpha=0.9)
    ax.axvline(0.0, color="#64748B", linewidth=1.0)
    ax.invert_yaxis()
    ax.set_title(
        f"Channel Delta · {first_label.replace('_', ' ')} vs {second_label.replace('_', ' ')}",
        color="#D6E3F0",
        fontsize=11,
    )
    _style_axes(ax)
    fig.tight_layout()
    return fig


def dominant_source_summary_text(
    result: MotorIntentionResult,
    trial: MotorTrial,
    phase_name: str,
    region_key: str,
    limit: int = 4,
) -> list[str]:
    frame = dominant_sources_for_region(
        result,
        trial,
        _validate_phase(phase_name),
        region_key,
    ).head(limit)
    return [f"{row.source} ({row.strength_uv:.1f} uV)" for row in frame.itertuples()]


def build_decoder_readiness_table(result: MotorIntentionResult) -> pd.DataFrame:
    return decoder_readiness_frame(result)


def make_metadata_json_bytes(result: MotorIntentionResult) -> bytes:
    payload = {
        "metadata": result.metadata,
        "events": result.events,
        "trial_metadata": result.trial_metadata,
    }
    return json.dumps(payload, indent=2, default=_jsonify).encode("utf-8")


def make_epochs_npz_bytes(result: MotorIntentionResult) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        epochs=result.epochs,
        epoch_times=result.epoch_times,
        labels=result.labels,
        channel_names=np.asarray(result.layout.channel_names),
    )
    return buffer.getvalue()


def make_continuous_npz_bytes(result: MotorIntentionResult) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        sensor_signal=result.sensor_signal,
        source_signal=result.source_signal,
        sensor_times=np.asarray(result.sensor_result["times"]),
        source_times=np.asarray(result.source_result["times"]),
        channel_names=np.asarray(result.layout.channel_names),
        source_names=np.asarray(result.metadata["source_names"]),
    )
    return buffer.getvalue()


def _validate_phase(phase_name: str) -> str:
    if phase_name not in PHASE_ORDER:
        raise ValueError(f"Unsupported phase '{phase_name}'")
    return phase_name


def _validate_class_label(result: MotorIntentionResult, flat_label: str) -> str:
    class_schema = {str(label) for label in result.metadata["class_schema"]}
    if flat_label not in class_schema:
        raise ValueError(f"Unknown class label '{flat_label}'")
    return flat_label


def _matching_trials_for_label(
    result: MotorIntentionResult,
    flat_label: str,
) -> list[MotorTrial]:
    return [trial for trial in result.trials if trial.flat_label == flat_label]


def _class_snapshot_values(
    result: MotorIntentionResult,
    flat_label: str,
    phase_name: str,
) -> np.ndarray:
    matching_trials = _matching_trials_for_label(result, flat_label)
    if not matching_trials:
        raise ValueError(f"No trials for class '{flat_label}'")

    sfreq = float(result.metadata["config_snapshot"]["sfreq"])
    sample_indices: list[int] = []
    for trial in matching_trials:
        snapshot_sec = phase_center(trial, phase_name)
        sample_index = int(
            np.clip(
                int(round(snapshot_sec * sfreq)),
                0,
                result.sensor_signal.shape[1] - 1,
            )
        )
        sample_indices.append(sample_index)
    return np.mean(result.sensor_signal[:, sample_indices], axis=1)


def _plot_topography_from_values(
    result: MotorIntentionResult,
    values_uv: np.ndarray,
    *,
    title: str,
    highlight_channels: Sequence[str] | None = None,
) -> Figure:
    positions = result.layout.coordinates
    triangulation = _layout_triangulation(result)

    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    fig.patch.set_facecolor("#0B1730")
    ax.set_facecolor("#0B1730")

    contour = ax.tricontourf(triangulation, values_uv, levels=10, cmap="coolwarm")
    ax.tricontour(
        triangulation,
        values_uv,
        levels=5,
        colors="white",
        linewidths=0.35,
        alpha=0.7,
    )
    ax.scatter(positions[:, 0], positions[:, 1], c="#0F172A", s=18, zorder=5)

    highlight_set = {
        name for name in (highlight_channels or []) if name in result.layout.channel_names
    }
    for x_coord, y_coord, channel_name in zip(
        positions[:, 0],
        positions[:, 1],
        result.layout.channel_names,
    ):
        if channel_name in highlight_set:
            ax.scatter(
                x_coord,
                y_coord,
                c="#FF7A59",
                s=64,
                zorder=6,
                edgecolors="white",
                linewidths=1.0,
            )
            ax.text(
                x_coord,
                y_coord + 0.04,
                channel_name,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#FF7A59",
                fontweight="bold",
            )

    _add_head_outline(ax)
    colorbar = fig.colorbar(contour, ax=ax, shrink=0.78, label="Amplitude (uV)")
    colorbar.ax.tick_params(colors="#D6E3F0")
    colorbar.ax.yaxis.label.set_color("#D6E3F0")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.05, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color="#D6E3F0")
    fig.tight_layout()
    return fig


def _layout_triangulation(result: MotorIntentionResult) -> mtri.Triangulation:
    key = (result.layout.name, tuple(result.layout.channel_names))
    triangulation = TRIANGULATION_CACHE.get(key)
    if triangulation is None:
        positions = result.layout.coordinates
        triangulation = mtri.Triangulation(positions[:, 0], positions[:, 1])
        TRIANGULATION_CACHE[key] = triangulation
    return triangulation


def _safe_trial_epoch_index(result: MotorIntentionResult, trial: MotorTrial) -> int:
    for index, other in enumerate(result.trials):
        if other is trial:
            return index

    trial_id = int(getattr(trial, "trial_id", -1))
    if 0 <= trial_id < len(result.trials):
        return trial_id

    raise ValueError("Could not resolve trial to a valid epoch index.")


def _signal_trace_bundle(
    result: MotorIntentionResult,
    trial: MotorTrial,
    channel_names: Sequence[str],
    view_mode: str,
    *,
    playhead_sec: float | None,
    window_sec: float,
    live_mode: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, str, str]], float | None]:
    sfreq = float(result.metadata["config_snapshot"]["sfreq"])
    bundle: dict[str, tuple[np.ndarray, np.ndarray, str, str]] = {}

    if view_mode == "Class Average":
        label_indices = [
            index
            for index, other in enumerate(result.trials)
            if other.flat_label == trial.flat_label
        ]
        if not label_indices:
            return bundle, None

        x_values = (
            np.asarray(result.epoch_times, dtype=float)
            - float(result.metadata["config_snapshot"]["export"]["epoch_start_sec"])
        )
        trace_color = CLASS_COLORS.get(trial.flat_label, "#D6E3F0")

        for channel_name in channel_names:
            channel_index = result.layout.index(channel_name)
            y_values = np.mean(result.epochs[label_indices, channel_index, :], axis=0) * 1e6
            bundle[channel_name] = (x_values, y_values, "class avg", trace_color)
        return bundle, None

    sensor_times = np.asarray(result.sensor_result["times"], dtype=float)

    if playhead_sec is not None:
        x_values, start_idx, end_idx, cursor_x = _live_window(
            sensor_times,
            center_sec=float(playhead_sec),
            window_sec=float(window_sec),
            lower_bound=float(trial.start_sec)
            if live_mode == "Selected Trial"
            else float(sensor_times[0]),
            upper_bound=float(trial.end_sec)
            if live_mode == "Selected Trial"
            else float(sensor_times[-1]),
            relative_offset=float(trial.start_sec) if live_mode == "Selected Trial" else 0.0,
        )
        trace_color = "#8BDBFF" if view_mode == "Continuous" else "#E2E8F0"
        suffix = "live"

        for channel_name in channel_names:
            channel_index = result.layout.index(channel_name)
            y_values = result.sensor_signal[channel_index, start_idx:end_idx] * 1e6
            bundle[channel_name] = (x_values, y_values, suffix, trace_color)
        return bundle, cursor_x

    if view_mode == "Continuous":
        start_sec = max(0.0, float(trial.start_sec) - 0.25)
        end_sec = min(float(sensor_times[-1]), float(trial.end_sec) + 0.25)
        start_idx = int(round(start_sec * sfreq))
        end_idx = int(round(end_sec * sfreq))
        x_values = sensor_times[start_idx:end_idx]

        for channel_name in channel_names:
            channel_index = result.layout.index(channel_name)
            y_values = result.sensor_signal[channel_index, start_idx:end_idx] * 1e6
            bundle[channel_name] = (x_values, y_values, "continuous", "#8BDBFF")
        return bundle, None

    x_values = (
        np.asarray(result.epoch_times, dtype=float)
        - float(result.metadata["config_snapshot"]["export"]["epoch_start_sec"])
    )
    trial_index = _safe_trial_epoch_index(result, trial)

    for channel_name in channel_names:
        channel_index = result.layout.index(channel_name)
        y_values = result.epochs[trial_index, channel_index, :] * 1e6
        bundle[channel_name] = (x_values, y_values, "epoch", "#E2E8F0")

    return bundle, None


def _live_window(
    sensor_times: np.ndarray,
    *,
    center_sec: float,
    window_sec: float,
    lower_bound: float,
    upper_bound: float,
    relative_offset: float,
) -> tuple[np.ndarray, int, int, float]:
    safe_window = max(float(window_sec), 0.25)
    clamped_center = min(max(float(center_sec), float(lower_bound)), float(upper_bound))
    half_window = 0.5 * safe_window

    start_sec = max(float(lower_bound), clamped_center - half_window)
    end_sec = min(float(upper_bound), clamped_center + half_window)

    if end_sec - start_sec < safe_window:
        if start_sec <= float(lower_bound):
            end_sec = min(float(upper_bound), start_sec + safe_window)
        else:
            start_sec = max(float(lower_bound), end_sec - safe_window)

    start_idx = int(np.searchsorted(sensor_times, start_sec, side="left"))
    end_idx = int(np.searchsorted(sensor_times, end_sec, side="right"))
    end_idx = max(end_idx, start_idx + 2)

    x_values = sensor_times[start_idx:end_idx] - float(relative_offset)
    cursor_x = clamped_center - float(relative_offset)
    return x_values, start_idx, end_idx, cursor_x


def _empty_signal_wall_figure(*, height: int = 520) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=24, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,19,38,0.96)",
        font=dict(color="rgba(214,227,240,0.96)"),
        showlegend=False,
        autosize=True,
        hovermode=False,
    )
    return fig


def _style_axes(ax: Axes, *, rotate_x: int = 0, label_size: float = 7.5) -> None:
    ax.tick_params(axis="x", labelrotation=rotate_x, colors="#D6E3F0", labelsize=label_size)
    ax.tick_params(axis="y", colors="#D6E3F0", labelsize=label_size)
    ax.grid(True, alpha=0.15, color="#2DD4FF")
    for spine in ax.spines.values():
        spine.set_color("#18314F")


def _add_head_outline(ax: Axes) -> None:
    ax.add_patch(Circle((0.0, 0.0), 1.02, fill=False, color="#334155", linewidth=1.5))
    ax.add_patch(
        Polygon(
            [(-0.08, 1.0), (0.0, 1.10), (0.08, 1.0)],
            fill=False,
            color="#334155",
            linewidth=1.2,
        )
    )


def _jsonify(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_signal_wall_plotly(
    result: MotorIntentionResult,
    trial: MotorTrial,
    channels: Sequence[str],
    *,
    view_mode: str,
    playhead_sec: float,
    window_sec: float,
    live_mode: str,
) -> go.Figure:
    selected = [ch for ch in channels if ch in result.layout.channel_names]
    if not selected:
        selected = list(result.layout.channel_names[:4])

    if not selected:
        return _empty_signal_wall_figure()

    trace_bundle, cursor_x = _signal_trace_bundle(
        result,
        trial,
        selected,
        view_mode,
        playhead_sec=float(playhead_sec),
        window_sec=float(window_sec),
        live_mode=live_mode,
    )

    if not trace_bundle:
        return _empty_signal_wall_figure()

    prepared_rows: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    scale_candidates: list[float] = []

    for channel_name in selected:
        if channel_name not in trace_bundle:
            continue

        x_values, y_values, suffix, trace_color = trace_bundle[channel_name]
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)

        if len(x_values) == 0 or len(y_values) == 0:
            continue

        if len(y_values) >= 5:
            kernel = np.array([1, 2, 3, 2, 1], dtype=float)
            kernel = kernel / kernel.sum()
            y_plot = np.convolve(y_values, kernel, mode="same")
        else:
            y_plot = y_values

        prepared_rows.append((channel_name, x_values, y_plot, suffix))
        scale_candidates.append(float(np.percentile(np.abs(y_plot), 98)))

    if not prepared_rows:
        return _empty_signal_wall_figure()

    n_rows = len(prepared_rows)
    wall_y_abs = max(10.0, max(scale_candidates) if scale_candidates else 10.0)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
    )

    for row_idx, (channel_name, x_values, y_plot, suffix) in enumerate(prepared_rows, start=1):
        x_start = float(x_values[0])
        x_end = float(x_values[-1])

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_plot,
                mode="lines",
                line=dict(width=1.8, color="rgba(226,232,240,0.90)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[0.0, 0.0],
                mode="lines",
                line=dict(width=1.0, color="rgba(120,170,220,0.22)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )

        if cursor_x is not None and x_start <= float(cursor_x) <= x_end:
            fig.add_trace(
                go.Scatter(
                    x=[float(cursor_x), float(cursor_x)],
                    y=[-1.15 * wall_y_abs, 1.15 * wall_y_abs],
                    mode="lines",
                    line=dict(width=1.5, color="rgba(245,183,66,0.70)", dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )

        fig.update_yaxes(
            title_text=channel_name,
            range=[-1.15 * wall_y_abs, 1.15 * wall_y_abs],
            row=row_idx,
            col=1,
            showgrid=True,
            gridcolor="rgba(45,212,255,0.10)",
            zeroline=False,
            tickfont=dict(size=10, color="rgba(184,208,232,0.82)"),
            title_font=dict(size=12, color="rgba(214,227,240,0.90)"),
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(45,212,255,0.10)",
            tickfont=dict(size=10, color="rgba(184,208,232,0.82)"),
            showticklabels=(row_idx == n_rows),
            row=row_idx,
            col=1,
        )

        fig.add_annotation(
            x=0.01,
            y=0.92,
            xref=f"x{row_idx}" if row_idx > 1 else "x",
            yref=f"y{row_idx}" if row_idx > 1 else "y",
            text=f"{channel_name} · {suffix}",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=13, color="rgba(214,227,240,0.90)"),
            bgcolor="rgba(0,0,0,0)",
        )

    fig.update_xaxes(title_text="Window Time (s)", row=n_rows, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=max(500, 185 * n_rows),
        margin=dict(l=64, r=20, t=24, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,19,38,0.96)",
        font=dict(color="rgba(214,227,240,0.96)"),
        showlegend=False,
        autosize=True,
        hovermode=False,
        uirevision="signal-wall-live",
    )

    return fig