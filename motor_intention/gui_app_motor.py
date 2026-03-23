from __future__ import annotations

from dataclasses import dataclass
import random
import time
import warnings
from typing import Sequence, TypedDict

import pandas as pd
import streamlit as st

from motor_intention import MotorIntentionArchitecture, MotorIntentionConfig
from motor_intention.architecture import MotorIntentionResult
from motor_intention.configs import ExportConfig, LayoutConfig, ProjectionConfig, TrialConfig
from motor_intention.trials import MotorTrial
from motor_intention.ui_logic import (
    class_balance_frame,
    default_region_for_trial,
    default_visible_channels,
    infer_phase,
    lateralization_score,
    midline_emphasis_score,
    phase_center,
    region_keys,
    region_spec,
)

from motor_intention.ui_plots import (
    build_decoder_readiness_table,
    build_signal_wall_plotly,
    build_timeline_chart,
    default_sensor_channels,
    dominant_source_summary_text,
    make_continuous_npz_bytes,
    make_epochs_npz_bytes,
    make_metadata_json_bytes,
    plot_brain_console,
    plot_channel_delta_bars,
    plot_class_summary_bars,
    plot_class_topography,
    plot_electrode_cluster_map,
    plot_electrode_ranking_table,
    plot_projection_matrix,
    plot_source_family_stack,
    plot_source_strength_bars,
    plot_topography,
    plot_topography_difference,
)


from motor_intention.ui_stick_figure import render_stick_figure_svg


st.set_page_config(
    page_title="Motor Intention Console",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


CLASS_FAMILY: dict[str, str] = {
    "left_arm": "arm",
    "right_arm": "arm",
    "left_leg": "leg",
    "right_leg": "leg",
    "rest": "rest",
}
REGION_FAMILY: dict[str, str] = {
    "left_arm": "arm",
    "right_arm": "arm",
    "left_leg": "leg",
    "right_leg": "leg",
    "sma": "support",
    "left_premotor": "support",
    "right_premotor": "support",
    "rest": "rest",
}
PLAY_SPEEDS: tuple[float, ...] = (0.5, 1.0, 2.0)
WINDOW_OPTIONS: tuple[float, ...] = (1.0, 2.0, 3.0)
LIVE_MODES: tuple[str, ...] = ("Selected Trial", "Continuous")
SIGNAL_VIEWS: tuple[str, ...] = ("Selected Epoch", "Continuous", "Class Average")
PHASE_NAMES: tuple[str, ...] = ("baseline", "preparation", "imagery", "recovery")
GROUP_FILTERS: tuple[str, ...] = (
    "All",
    "Arm Only",
    "Leg Only",
    "Rest Only",
    "Left Only",
    "Right Only",
)


@dataclass(frozen=True)
class ViewModel:
    electrode_frame: pd.DataFrame
    visible_defaults: list[str]
    dominant_sources: list[str]
    lateralization: float
    midline_emphasis: float


class ViewModelPayload(TypedDict):
    electrode_frame: pd.DataFrame
    visible_defaults: list[str]
    dominant_sources: list[str]
    lateralization: float
    midline_emphasis: float


@st.cache_data(
    show_spinner=False,
    hash_funcs={MotorIntentionResult: lambda result: _result_key(result)},
)
def _cached_view_model_payload(
    result: MotorIntentionResult,
    trial_id: int,
    phase_name: str,
    region_key: str,
) -> ViewModelPayload:
    trial = result.trials[trial_id]
    electrode_frame = plot_electrode_ranking_table(
        result,
        trial,
        phase_name,
        region_key,
        top_n=len(result.layout.channel_names),
    )
    return {
        "electrode_frame": electrode_frame,
        "visible_defaults": default_visible_channels(
            result,
            trial,
            phase_name,
            region_key,
            limit=8,
        ),
        "dominant_sources": dominant_source_summary_text(
            result,
            trial,
            phase_name,
            region_key,
        ),
        "lateralization": lateralization_score(result, trial, phase_name),
        "midline_emphasis": midline_emphasis_score(result, trial, phase_name),
    }


def _view_model(
    result: MotorIntentionResult,
    trial: MotorTrial,
    phase_name: str,
    region_key: str,
) -> ViewModel:
    payload = _cached_view_model_payload(result, trial.trial_id, phase_name, region_key)
    return ViewModel(
        electrode_frame=payload["electrode_frame"],
        visible_defaults=payload["visible_defaults"],
        dominant_sources=payload["dominant_sources"],
        lateralization=payload["lateralization"],
        midline_emphasis=payload["midline_emphasis"],
    )






@dataclass(frozen=True)
class SidebarState:
    submitted: bool = False
    new_config: MotorIntentionConfig | None = None
    selected_labels: tuple[str, ...] = ()
    selected_group: str = "All"
    compare_label: str = "rest"
    signal_view: str = "Selected Epoch"
    monitor_slots: int = 4
    follow_trial_region: bool = True


class BuildDefaults(TypedDict):
    class_cycle: list[str]
    n_trials: int
    sfreq: float
    seed: int
    baseline_sec: float
    preparation_sec: float
    imagery_sec: float
    recovery_sec: float
    montage_name: str
    noise_uv: float
    reference: str
    include_eog: bool
    include_emg: bool
    include_line: bool
    epoch_start_sec: float
    epoch_end_sec: float


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top, rgba(16,35,65,0.82) 0%, rgba(7,17,34,0.98) 55%, rgba(4,9,18,1) 100%);
            color: #d6e3f0;
        }
        .block-container {
            padding-top: 0.7rem;
            padding-bottom: 1.8rem;
            max-width: 1680px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(7,19,38,0.98) 0%, rgba(4,14,28,0.98) 100%);
            border-right: 1px solid rgba(45,212,255,0.12);
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding-bottom: 0.9rem;
            border-bottom: 1px solid rgba(45,212,255,0.1);
            margin-bottom: 0.7rem;
        }
        .brand-name {
            font-weight: 800;
            font-size: 1.05rem;
            color: #e8f4ff;
            line-height: 1.15;
        }
        .brand-sub {
            font-size: 0.66rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #7aa4c7;
        }
        .sb-head {
            font-size: 0.67rem;
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #2dd4ff;
            padding: 0.75rem 0 0.35rem;
            border-top: 1px solid rgba(45,212,255,0.12);
            margin-top: 0.4rem;
        }
        .sb-head:first-child {
            border-top: none;
            padding-top: 0;
        }
        .card {
            background: linear-gradient(155deg, rgba(10,22,50,0.97) 0%, rgba(5,12,28,0.97) 100%);
            border: 1px solid rgba(45,212,255,0.1);
            border-radius: 16px;
            padding: 0.95rem 1.05rem 0.9rem;
            margin-bottom: 0.7rem;
            box-shadow: 0 4px 28px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.025);
        }
        .card-title {
            font-weight: 700;
            font-size: 0.84rem;
            letter-spacing: 0.03em;
            color: #d0e4f5;
            margin-bottom: 0.15rem;
        }
        .card-sub {
            font-size: 0.74rem;
            color: #7aa4c7;
            margin-bottom: 0.55rem;
            line-height: 1.45;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(155deg, rgba(10,22,50,0.97) 0%, rgba(5,12,28,0.97) 100%);
            border: 1px solid rgba(45,212,255,0.1);
            border-radius: 13px;
            padding: 0.6rem 0.85rem 0.5rem;
            box-shadow: 0 3px 18px rgba(0,0,0,0.4);
        }
        div[data-testid="stMetric"] > label {
            color: #7aa4c7 !important;
            font-size: 0.64rem !important;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #e8f4ff !important;
            font-size: 1.12rem !important;
            font-weight: 700 !important;
        }
        .strip {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.3rem;
            padding: 0.5rem 0.85rem;
            background: rgba(5,12,26,0.75);
            border: 1px solid rgba(45,212,255,0.1);
            border-radius: 11px;
            margin-bottom: 0.75rem;
        }
        .chip {
            display: inline-flex;
            align-items: center;
            padding: 0.17rem 0.52rem;
            border-radius: 999px;
            border: 1px solid rgba(45,212,255,0.1);
            background: rgba(45,212,255,0.04);
            color: #9bc1e0;
            font-size: 0.68rem;
        }
        .chip-arm { border-color: rgba(255,122,89,0.4); color: #ff7a59; background: rgba(255,122,89,0.06); }
        .chip-leg { border-color: rgba(46,196,182,0.4); color: #2ec4b6; background: rgba(46,196,182,0.06); }
        .chip-sup { border-color: rgba(45,212,255,0.25); color: #7dd3fc; background: rgba(45,212,255,0.05); }
        .phase-track {
            display: flex;
            height: 4px;
            border-radius: 3px;
            overflow: hidden;
            margin: 0.5rem 0 0.8rem;
            border: 1px solid rgba(45,212,255,0.08);
            gap: 2px;
        }
        .ph { flex: 1; border-radius: 2px; opacity: 0.25; }
        .ph.on { opacity: 1; }
        .ph-baseline { background: #94a3b8; }
        .ph-preparation { background: #f0b44c; }
        .ph-imagery { background: #d95d39; }
        .ph-recovery { background: #5c80bc; }
        .src {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.24rem 0.52rem;
            border-radius: 7px;
            margin-bottom: 0.16rem;
            background: rgba(8,18,40,0.6);
            border: 1px solid rgba(45,212,255,0.07);
            font-size: 0.71rem;
        }
        .src-n { color: #b8d0e8; }
        .src-v { color: #2dd4ff; font-variant-numeric: tabular-nums; }
        .empty {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 66vh;
            text-align: center;
            gap: 1rem;
        }
        .empty-title {
            font-weight: 800;
            font-size: 1.9rem;
            color: #d0e4f5;
        }
        .empty-body {
            font-size: 0.9rem;
            max-width: 560px;
            color: #8fa7bf;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _default_config() -> MotorIntentionConfig:
    return MotorIntentionConfig()


def _run_architecture(config: MotorIntentionConfig) -> tuple[MotorIntentionResult, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = MotorIntentionArchitecture(config).run()
    return result, [str(item.message) for item in caught]


@st.cache_data(show_spinner=False)
def _cached_run(config: MotorIntentionConfig) -> tuple[MotorIntentionResult, list[str]]:
    return _run_architecture(config)


def _result_key(result: MotorIntentionResult) -> tuple[object, ...]:
    snapshot = result.metadata["config_snapshot"]
    return (
        result.metadata.get("generator_version"),
        snapshot["seed"],
        snapshot["sfreq"],
        result.metadata["layout_name"],
        len(result.trials),
        len(result.metadata["channel_names"]),
    )


@st.cache_data(
    show_spinner=False,
    hash_funcs={MotorIntentionResult: lambda result: _result_key(result)},
)
def _cached_download_payloads(result: MotorIntentionResult) -> dict[str, bytes]:
    return {
        "metadata": make_metadata_json_bytes(result),
        "epochs": make_epochs_npz_bytes(result),
        "continuous": make_continuous_npz_bytes(result),
    }


def _download_payloads(result: MotorIntentionResult) -> dict[str, bytes]:
    return _cached_download_payloads(result)


def _safe_live_mode(value: str) -> str:
    return value if value in LIVE_MODES else LIVE_MODES[0]


def _safe_signal_view(value: str) -> str:
    return value if value in SIGNAL_VIEWS else SIGNAL_VIEWS[0]


def _safe_window(value: float) -> float:
    return value if value in WINDOW_OPTIONS else WINDOW_OPTIONS[1]


def _safe_play_speed(value: float) -> float:
    return value if value in PLAY_SPEEDS else 1.0


def _safe_monitor_slots(value: int) -> int:
    return value if value in (4, 6, 8) else 4


def _safe_region_key(value: str) -> str:
    return value if value in set(region_keys()) else "rest"


def _clamp_trial_id(result: MotorIntentionResult, trial_id: int) -> int:
    if not result.trials:
        return 0
    return max(0, min(trial_id, len(result.trials) - 1))


def _selected_trial(result: MotorIntentionResult) -> MotorTrial:
    return result.trials[_clamp_trial_id(result, int(st.session_state.mi_trial_id))]


def _trial_for_playhead(result: MotorIntentionResult, playhead_sec: float) -> MotorTrial:
    for trial in result.trials:
        if trial.start_sec <= playhead_sec < trial.end_sec:
            return trial
    return result.trials[-1]


def _trial_bounds(trial: MotorTrial) -> tuple[float, float]:
    return float(trial.start_sec), float(trial.end_sec)


def _record_bounds(result: MotorIntentionResult) -> tuple[float, float]:
    times = result.sensor_result.get("times")
    if times is None or len(times) == 0:
        return 0.0, 0.0
    return float(times[0]), float(times[-1])


def _clamp_playhead(result: MotorIntentionResult, playhead_sec: float) -> float:
    if not result.trials:
        return 0.0
    if st.session_state.mi_live_mode == "Selected Trial":
        start_sec, end_sec = _trial_bounds(_selected_trial(result))
    else:
        start_sec, end_sec = _record_bounds(result)
    if end_sec <= start_sec:
        return start_sec
    return max(start_sec, min(playhead_sec, end_sec))


def _set_playhead(result: MotorIntentionResult, playhead_sec: float) -> None:
    st.session_state.mi_playhead_sec = _clamp_playhead(result, playhead_sec)
    st.session_state.mi_last_tick_wall_time = time.monotonic()


def _reset_playhead(result: MotorIntentionResult) -> None:
    if not result.trials:
        _set_playhead(result, 0.0)
        return
    if st.session_state.mi_live_mode == "Selected Trial":
        _set_playhead(result, _selected_trial(result).start_sec)
    else:
        _set_playhead(result, _record_bounds(result)[0])


def _step_playhead(result: MotorIntentionResult, direction: int) -> None:
    step_sec = max(0.1, float(st.session_state.mi_window_sec) / 5.0)
    _set_playhead(result, float(st.session_state.mi_playhead_sec) + direction * step_sec)


def _jump_to_phase(result: MotorIntentionResult, phase_name: str) -> None:
    if phase_name not in PHASE_NAMES or not result.trials:
        return
    _set_playhead(result, phase_center(_selected_trial(result), phase_name))


def _advance_playhead(result: MotorIntentionResult) -> None:
    now = time.monotonic()
    last_tick = float(st.session_state.mi_last_tick_wall_time)
    st.session_state.mi_last_tick_wall_time = now
    if not bool(st.session_state.mi_playing) or not result.trials:
        return

    delta_sec = max(0.0, now - last_tick) * float(st.session_state.mi_play_speed)
    if delta_sec <= 0.0:
        return

    if st.session_state.mi_live_mode == "Selected Trial":
        trial = _selected_trial(result)
        start_sec, end_sec = _trial_bounds(trial)
        duration = max(end_sec - start_sec, 1e-6)
        offset = (float(st.session_state.mi_playhead_sec) - start_sec + delta_sec) % duration
        st.session_state.mi_playhead_sec = start_sec + offset
        return

    record_start, record_end = _record_bounds(result)
    duration = max(record_end - record_start, 1e-6)
    offset = (float(st.session_state.mi_playhead_sec) - record_start + delta_sec) % duration
    st.session_state.mi_playhead_sec = record_start + offset
    st.session_state.mi_trial_id = _trial_for_playhead(
        result,
        float(st.session_state.mi_playhead_sec),
    ).trial_id


def _initialize_state(result: MotorIntentionResult | None) -> None:
    st.session_state.setdefault("motor_result", result)
    st.session_state.setdefault("motor_config", _default_config())
    st.session_state.setdefault("motor_warnings", [])
    st.session_state.setdefault("mi_trial_id", 0)
    st.session_state.setdefault("mi_region", "rest")
    st.session_state.setdefault("mi_playing", False)
    st.session_state.setdefault("mi_playhead_sec", 0.0)
    st.session_state.setdefault("mi_last_tick_wall_time", time.monotonic())
    st.session_state.setdefault("mi_play_speed", 1.0)
    st.session_state.setdefault("mi_live_mode", "Selected Trial")
    st.session_state.setdefault("mi_window_sec", 2.0)
    st.session_state.setdefault("mi_follow_trial_region", True)
    st.session_state.setdefault("mi_signal_view", SIGNAL_VIEWS[0])
    st.session_state.setdefault("mi_monitor_slots", 4)
    st.session_state.setdefault("mi_compare_label", "rest")

    if result is None or not result.trials:
        st.session_state.mi_trial_id = 0
        st.session_state.mi_region = "rest"
        st.session_state.mi_playing = False
        st.session_state.mi_playhead_sec = 0.0
        st.session_state.mi_last_tick_wall_time = time.monotonic()
        st.session_state.mi_follow_trial_region = True
        st.session_state.mi_signal_view = SIGNAL_VIEWS[0]
        st.session_state.mi_monitor_slots = 4
        st.session_state.mi_compare_label = "rest"
        return

    st.session_state.mi_trial_id = _clamp_trial_id(result, int(st.session_state.mi_trial_id))
    st.session_state.mi_region = _safe_region_key(str(st.session_state.mi_region))
    st.session_state.mi_play_speed = _safe_play_speed(float(st.session_state.mi_play_speed))
    st.session_state.mi_live_mode = _safe_live_mode(str(st.session_state.mi_live_mode))
    st.session_state.mi_window_sec = _safe_window(float(st.session_state.mi_window_sec))
    st.session_state.mi_follow_trial_region = bool(st.session_state.mi_follow_trial_region)
    st.session_state.mi_signal_view = _safe_signal_view(str(st.session_state.mi_signal_view))
    st.session_state.mi_monitor_slots = _safe_monitor_slots(int(st.session_state.mi_monitor_slots))
    st.session_state.mi_playhead_sec = _clamp_playhead(result, float(st.session_state.mi_playhead_sec))
    st.session_state.mi_last_tick_wall_time = time.monotonic()
    if st.session_state.mi_follow_trial_region:
        st.session_state.mi_region = default_region_for_trial(_selected_trial(result))
    valid_compare_labels = [str(label) for label in result.metadata["class_schema"]]
    if str(st.session_state.mi_compare_label) not in valid_compare_labels:
        st.session_state.mi_compare_label = valid_compare_labels[0] if valid_compare_labels else "rest"


def _render_brand() -> None:
    st.markdown(
        """
        <div class="brand">
          <div style="font-size:1.45rem;line-height:1;">🧠</div>
          <div>
            <div class="brand-name">Motor Intention</div>
            <div class="brand-sub">Neural Console</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _build_defaults(config: MotorIntentionConfig) -> BuildDefaults:
    return {
        "class_cycle": list(config.trial.class_cycle),
        "n_trials": int(config.trial.n_trials),
        "sfreq": float(config.sfreq),
        "seed": int(config.seed),
        "baseline_sec": float(config.trial.baseline_sec),
        "preparation_sec": float(config.trial.preparation_sec),
        "imagery_sec": float(config.trial.imagery_sec),
        "recovery_sec": float(config.trial.recovery_sec),
        "montage_name": config.layout.montage_name,
        "noise_uv": float(config.projection.sensor_noise_uv),
        "reference": config.reference,
        "include_eog": bool(config.include_eog),
        "include_emg": bool(config.include_emg),
        "include_line": bool(config.include_line_noise),
        "epoch_start_sec": float(config.export.epoch_start_sec),
        "epoch_end_sec": float(config.export.epoch_end_sec),
    }


def _sidebar_build(active_config: MotorIntentionConfig) -> tuple[bool, MotorIntentionConfig | None]:
    defaults = _build_defaults(active_config)
    st.markdown("<div class='sb-head'>Build</div>", unsafe_allow_html=True)

    with st.form("mi_build_form", clear_on_submit=False):
        class_cycle = tuple(
            st.multiselect(
                "Classes",
                ["left_arm", "right_arm", "left_leg", "right_leg", "rest"],
                default=defaults["class_cycle"],
            )
        )

        top_cols = st.columns(2)
        n_trials = int(top_cols[0].number_input("Trials", min_value=5, max_value=200, value=defaults["n_trials"], step=5))
        seed = int(top_cols[1].number_input("Seed", min_value=0, value=defaults["seed"], step=1))

        mid_cols = st.columns(2)
        sfreq = float(
            mid_cols[0].selectbox(
                "Sampling Rate",
                [128.0, 250.0, 500.0],
                index=[128.0, 250.0, 500.0].index(defaults["sfreq"]) if defaults["sfreq"] in {128.0, 250.0, 500.0} else 1,
            )
        )
        montage_name = str(
            mid_cols[1].selectbox(
                "Montage",
                ["motor_21", "motor_14"],
                index=0 if defaults["montage_name"] == "motor_21" else 1,
            )
        )

        baseline_sec = float(st.slider("Baseline", 0.5, 3.0, defaults["baseline_sec"], 0.1))
        preparation_sec = float(st.slider("Preparation", 0.2, 2.0, defaults["preparation_sec"], 0.1))
        imagery_sec = float(st.slider("Imagery", 1.0, 5.0, defaults["imagery_sec"], 0.1))
        recovery_sec = float(st.slider("Recovery", 0.5, 3.0, defaults["recovery_sec"], 0.1))

        with st.expander("Signal & Epoch", expanded=False):
            art_cols = st.columns(2)
            include_eog = bool(art_cols[0].checkbox("EOG", value=defaults["include_eog"]))
            include_emg = bool(art_cols[1].checkbox("EMG", value=defaults["include_emg"]))
            include_line = bool(art_cols[0].checkbox("Line Noise", value=defaults["include_line"]))
            noise_uv = float(st.slider("Sensor Noise (uV)", 0.1, 5.0, defaults["noise_uv"], 0.1))
            reference = str(
                st.selectbox(
                    "Reference",
                    ["average", "Cz", "linked_mastoids"],
                    index=["average", "Cz", "linked_mastoids"].index(defaults["reference"])
                    if defaults["reference"] in {"average", "Cz", "linked_mastoids"}
                    else 0,
                )
            )
            epoch_start_sec = float(st.slider("Epoch Start", 0.0, 4.0, defaults["epoch_start_sec"], 0.1))
            epoch_end_sec = float(
                st.slider(
                    "Epoch End",
                    min_value=epoch_start_sec + 0.2,
                    max_value=6.0,
                    value=max(defaults["epoch_end_sec"], epoch_start_sec + 0.2),
                    step=0.1,
                )
            )

        submitted = st.form_submit_button("Generate Dataset", type="primary", use_container_width=True)

    if not submitted:
        return False, None
    if not class_cycle:
        st.error("Select at least one class before generating a dataset.")
        return False, None

    config = MotorIntentionConfig(
        sfreq=sfreq,
        seed=seed,
        reference=reference,
        include_eog=include_eog,
        include_emg=include_emg,
        include_line_noise=include_line,
        trial=TrialConfig(
            class_cycle=class_cycle,
            n_trials=n_trials,
            baseline_sec=baseline_sec,
            preparation_sec=preparation_sec,
            imagery_sec=imagery_sec,
            recovery_sec=recovery_sec,
        ),
        layout=LayoutConfig(montage_name=montage_name),
        projection=ProjectionConfig(sensor_noise_uv=noise_uv),
        export=ExportConfig(epoch_start_sec=epoch_start_sec, epoch_end_sec=epoch_end_sec),
    )
    return True, config


def _trial_matches(trial: MotorTrial, group_name: str) -> bool:
    if group_name == "Arm Only":
        return trial.effector_family == "arm"
    if group_name == "Leg Only":
        return trial.effector_family == "leg"
    if group_name == "Rest Only":
        return trial.effector_family == "rest"
    if group_name == "Left Only":
        return trial.side == "left"
    if group_name == "Right Only":
        return trial.side == "right"
    return True


def _sync_trial_after_selection(result: MotorIntentionResult, trial_id: int, follow_trial_region: bool) -> None:
    st.session_state.mi_trial_id = _clamp_trial_id(result, trial_id)
    if st.session_state.mi_live_mode == "Selected Trial":
        trial = _selected_trial(result)
        if not (trial.start_sec <= float(st.session_state.mi_playhead_sec) < trial.end_sec):
            _set_playhead(result, trial.start_sec)
    if follow_trial_region:
        st.session_state.mi_region = default_region_for_trial(_selected_trial(result))


def _render_sidebar(result: MotorIntentionResult | None, active_config: MotorIntentionConfig) -> SidebarState:
    with st.sidebar:
        _render_brand()
        submitted, new_config = _sidebar_build(active_config)

        if result is None or not result.trials:
            return SidebarState(submitted=submitted, new_config=new_config)

        available_labels = [str(label) for label in result.metadata["class_schema"]]

        st.markdown("<div class='sb-head'>Navigate</div>", unsafe_allow_html=True)
        selected_labels = tuple(
            st.multiselect(
                "Classes",
                options=available_labels,
                default=available_labels,
                label_visibility="collapsed",
            )
        )
        selected_group = str(
            st.selectbox(
                "Group",
                options=GROUP_FILTERS,
                index=0,
                label_visibility="collapsed",
            )
        )

        filtered_trials = [
            trial
            for trial in result.trials
            if trial.flat_label in (selected_labels or tuple(available_labels))
            and _trial_matches(trial, selected_group)
        ]
        if not filtered_trials:
            filtered_trials = list(result.trials)

        filtered_ids = [trial.trial_id for trial in filtered_trials]
        current_trial_id = int(st.session_state.mi_trial_id)
        if current_trial_id not in filtered_ids:
            current_trial_id = filtered_ids[0]
            st.session_state.mi_trial_id = current_trial_id

        follow_trial_region = bool(
            st.toggle(
                "Follow Trial Region",
                key="mi_follow_trial_region",
            )
        )

        nav_cols = st.columns(3)
        if nav_cols[0].button("◀ Prev", use_container_width=True):
            idx = filtered_ids.index(current_trial_id)
            _sync_trial_after_selection(result, filtered_ids[max(0, idx - 1)], follow_trial_region)
            current_trial_id = int(st.session_state.mi_trial_id)
        if nav_cols[1].button("Next ▶", use_container_width=True):
            idx = filtered_ids.index(current_trial_id)
            _sync_trial_after_selection(result, filtered_ids[min(len(filtered_ids) - 1, idx + 1)], follow_trial_region)
            current_trial_id = int(st.session_state.mi_trial_id)
        if nav_cols[2].button("⚄ Random", use_container_width=True):
            _sync_trial_after_selection(result, random.choice(filtered_ids), follow_trial_region)
            current_trial_id = int(st.session_state.mi_trial_id)

        selected_trial_id = int(
            st.select_slider(
                "Trial",
                options=filtered_ids,
                value=current_trial_id,
            )
        )
        if selected_trial_id != int(st.session_state.mi_trial_id):
            _sync_trial_after_selection(result, selected_trial_id, follow_trial_region)

        jump_cols = st.columns([3, 2])
        jump_label = str(jump_cols[0].selectbox("Jump To Class", available_labels, label_visibility="collapsed"))
        if jump_cols[1].button("Jump", use_container_width=True):
            first_match = next((trial for trial in filtered_trials if trial.flat_label == jump_label), None)
            if first_match is not None:
                _sync_trial_after_selection(result, first_match.trial_id, follow_trial_region)

        st.markdown("<div class='sb-head'>Phase</div>", unsafe_allow_html=True)
        current_trial = _selected_trial(result)
        current_phase = infer_phase(current_trial, float(st.session_state.mi_playhead_sec))
        st.caption(f"Current phase: **{current_phase.title()}**")
        phase_cols = st.columns(2)
        for index, phase_name in enumerate(PHASE_NAMES):
            if phase_cols[index % 2].button(phase_name.title(), use_container_width=True):
                _jump_to_phase(result, phase_name)

        st.markdown("<div class='sb-head'>Region</div>", unsafe_allow_html=True)
        if follow_trial_region:
            st.session_state.mi_region = default_region_for_trial(_selected_trial(result))
            st.caption(f"Auto region: **{region_spec(st.session_state.mi_region).display_name}**")
        region_options = region_keys()
        region_choice = str(
            st.radio(
                "Region",
                options=region_options,
                format_func=lambda key: region_spec(key).display_name,
                index=region_options.index(_safe_region_key(str(st.session_state.mi_region))),
                disabled=follow_trial_region,
                label_visibility="collapsed",
            )
        )
        if not follow_trial_region:
            st.session_state.mi_region = _safe_region_key(region_choice)

        st.markdown("<div class='sb-head'>Playback</div>", unsafe_allow_html=True)
        play_cols = st.columns(2)
        if play_cols[0].button("Play" if not st.session_state.mi_playing else "Pause", use_container_width=True):
            st.session_state.mi_playing = not bool(st.session_state.mi_playing)
            st.session_state.mi_last_tick_wall_time = time.monotonic()
        if play_cols[1].button("Reset", use_container_width=True):
            st.session_state.mi_playing = False
            _reset_playhead(result)

        step_cols = st.columns(2)
        if step_cols[0].button("◀ Step", use_container_width=True):
            st.session_state.mi_playing = False
            _step_playhead(result, -1)
        if step_cols[1].button("Step ▶", use_container_width=True):
            st.session_state.mi_playing = False
            _step_playhead(result, 1)
        

        if st.session_state.get("_reset_mi_playback", False):
            st.session_state.mi_play_speed = 1.0
            st.session_state.mi_live_mode = "Selected Trial"
            st.session_state.mi_window_sec = 2.0
            st.session_state.mi_last_tick_wall_time = time.monotonic()
            st.session_state._reset_mi_playback = False

        st.selectbox(
            "Speed",
            PLAY_SPEEDS,
            index=PLAY_SPEEDS.index(_safe_play_speed(float(st.session_state.mi_play_speed))),
            key="mi_play_speed",
        )
        previous_live_mode = str(st.session_state.mi_live_mode)
        st.selectbox(
            "Live Mode",
            LIVE_MODES,
            index=LIVE_MODES.index(_safe_live_mode(previous_live_mode)),
            key="mi_live_mode",
        )
        if str(st.session_state.mi_live_mode) != previous_live_mode:
            _reset_playhead(result)
        st.selectbox(
            "Signal Window (s)",
            WINDOW_OPTIONS,
            index=WINDOW_OPTIONS.index(_safe_window(float(st.session_state.mi_window_sec))),
            key="mi_window_sec",
        )

        st.markdown("<div class='sb-head'>Options</div>", unsafe_allow_html=True)
        st.selectbox(
            "Signal View",
            SIGNAL_VIEWS,
            index=SIGNAL_VIEWS.index(_safe_signal_view(str(st.session_state.mi_signal_view))),
            key="mi_signal_view",
        )
        st.select_slider(
            "Monitor Channels",
            options=[4, 6, 8],
            value=int(st.session_state.mi_monitor_slots),
            key="mi_monitor_slots",
        )
        compare_candidates = [label for label in available_labels if label != _selected_trial(result).flat_label]
        compare_default = _safe_compare_label(
            result,
            str(st.session_state.mi_compare_label),
            _selected_trial(result).flat_label,
        )
        st.selectbox(
            "Compare Against",
            options=compare_candidates or available_labels,
            index=(compare_candidates or available_labels).index(compare_default),
            key="mi_compare_label",
        )

    return SidebarState(
        submitted=submitted,
        new_config=new_config,
        selected_labels=selected_labels,
        selected_group=selected_group,
        compare_label=str(st.session_state.mi_compare_label),
        signal_view=_safe_signal_view(str(st.session_state.mi_signal_view)),
        monitor_slots=_safe_monitor_slots(int(st.session_state.mi_monitor_slots)),
        follow_trial_region=follow_trial_region,
    )


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty">
          <div style="font-size:4.5rem;opacity:0.18;">🧠</div>
          <div class="empty-title">Motor Intention Console</div>
          <div class="empty-body">
            Build a synthetic limb-intention EEG dataset and inspect the full chain:
            trial label → latent source modulation → source-to-sensor projection →
            electrode activity → decoder-ready epochs.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Quick Guide", expanded=False):
        columns = st.columns(3)
        columns[0].markdown(
            """
            **Classes**
            - `left_arm` / `right_arm`
            - `left_leg` / `right_leg`
            - `rest`
            """
        )
        columns[1].markdown(
            """
            **Phases**
            - `baseline`
            - `preparation`
            - `imagery`
            - `recovery`
            """
        )
        columns[2].markdown(
            """
            **What to look for**
            - Arms: more lateral sensorimotor emphasis
            - Legs: stronger medial / midline emphasis
            - Rest: more symmetric background structure
            """
        )


def _status_strip(result: MotorIntentionResult, trial: MotorTrial, region_key: str, active_phase: str) -> None:
    class_family = CLASS_FAMILY.get(trial.flat_label, "rest")
    region_family = REGION_FAMILY.get(region_key, "rest")
    class_chip = {"arm": "chip-arm", "leg": "chip-leg"}.get(class_family, "chip")
    region_chip = {"arm": "chip-arm", "leg": "chip-leg", "support": "chip-sup"}.get(region_family, "chip")

    chips = [
        (f"trial {trial.trial_id}", "chip"),
        (trial.flat_label.replace("_", " "), class_chip),
        (region_spec(region_key).display_name, region_chip),
        (f"phase: {active_phase}", "chip"),
        (result.metadata["layout_name"], "chip"),
        (result.metadata["reference"], "chip"),
        (f"{len(result.trials)} trials", "chip"),
        (f"{len(result.metadata['source_names'])} src", "chip"),
        (f"{len(result.layout.channel_names)} ch", "chip"),
    ]
    body = "".join(f"<span class='chip {chip_class}'>{label}</span>" for label, chip_class in chips)
    st.markdown(f"<div class='strip'>{body}</div>", unsafe_allow_html=True)


def _phase_track(active_phase: str) -> None:
    blocks = "".join(
        f"<div class='ph ph-{phase_name}{' on' if phase_name == active_phase else ''}'></div>"
        for phase_name in PHASE_NAMES
    )
    st.markdown(f"<div class='phase-track'>{blocks}</div>", unsafe_allow_html=True)


def _source_list(items: Sequence[str], heading: str) -> None:
    st.markdown(f"<div class='card-title'>{heading}</div>", unsafe_allow_html=True)
    for item in items:
        name, _, suffix = item.partition(" (")
        value = f"({suffix}" if suffix else ""
        st.markdown(
            f"<div class='src'><span class='src-n'>{name}</span><span class='src-v'>{value}</span></div>",
            unsafe_allow_html=True,
        )


def _electrode_list(frame: pd.DataFrame, limit: int = 5) -> None:
    for row in frame.head(limit).itertuples():
        st.markdown(
            f"<div class='src'><span class='src-n'>{row.channel}</span><span class='src-v'>{row.dynamic_weight:.1f}</span></div>",
            unsafe_allow_html=True,
        )


@st.fragment(run_every=0.15)
def _render_live_console_fragment(
    signal_view: str,
    monitor_slots: int,
    follow_trial_region: bool,
) -> None:
    stored_result = st.session_state.get("motor_result")
    if not isinstance(stored_result, MotorIntentionResult) or not stored_result.trials:
        return

    result = stored_result
    _advance_playhead(result)
    playhead_sec = _clamp_playhead(result, float(st.session_state.mi_playhead_sec))
    st.session_state.mi_playhead_sec = playhead_sec

    display_trial = _selected_trial(result)
    if st.session_state.mi_live_mode == "Continuous":
        display_trial = _trial_for_playhead(result, playhead_sec)
        if int(st.session_state.mi_trial_id) != display_trial.trial_id:
            st.session_state.mi_trial_id = display_trial.trial_id

    active_phase = infer_phase(display_trial, playhead_sec)
    region_key = default_region_for_trial(display_trial) if follow_trial_region else _safe_region_key(str(st.session_state.mi_region))
    view_model = _view_model(result, display_trial, active_phase, region_key)

    visible_channels = view_model.visible_defaults[:monitor_slots]
    if not visible_channels:
        visible_channels = default_sensor_channels(result)[:monitor_slots]
    highlight_channels = view_model.electrode_frame["channel"].head(6).tolist()

    _status_strip(result, display_trial, region_key, active_phase)

    metrics = st.columns(6)
    metrics[0].metric("Live Trial", display_trial.flat_label.replace("_", " ").title())
    metrics[1].metric("Region", region_spec(region_key).display_name)
    metrics[2].metric("Playhead", f"{playhead_sec:.2f}s")
    metrics[3].metric("Phase", active_phase.title())
    metrics[4].metric("Lateralization", f"{view_model.lateralization:+.2f}")
    metrics[5].metric("Midline Emphasis", f"{view_model.midline_emphasis:.2f}")

    _phase_track(active_phase)

    col_brain, col_body, col_signal = st.columns([1.0, 0.72, 1.3], gap="medium")

    with col_brain:
        spec = region_spec(region_key)
        st.markdown(
            f"<div class='card'><div class='card-title'>Cortex Console · {spec.display_name}</div>"
            f"<div class='card-sub'>{spec.expected_pattern}</div>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_brain_console(result, display_trial, active_phase, region_key), clear_figure=True)
        source_tabs = st.tabs(["Source RMS", "Family Mix"])
        with source_tabs[0]:
            st.pyplot(plot_source_strength_bars(result, display_trial, active_phase, region_key), clear_figure=True)
        with source_tabs[1]:
            st.pyplot(plot_source_family_stack(result, display_trial, active_phase, region_key), clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_body:
        st.markdown("<div class='card' style='padding-bottom:0.5rem'>", unsafe_allow_html=True)
        spec = region_spec(region_key)
        st.markdown(
            render_stick_figure_svg(
                display_trial.flat_label,
                display_trial.effector_family,
                display_trial.side,
                active_phase,
                emphasis_family=spec.family if spec.family in {"arm", "leg"} else display_trial.effector_family,
                emphasis_side=spec.side or display_trial.side,
            ),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        _source_list(view_model.dominant_sources, "Dominant Sources")
        st.markdown("<div style='margin-top:0.55rem'></div>", unsafe_allow_html=True)
        _electrode_list(view_model.electrode_frame, limit=5)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_signal:
        st.markdown(
            "<div class='card'><div class='card-title'>EEG Monitor Wall</div>"
            "<div class='card-sub'>Live multi-channel view around the current playhead.</div>",
            unsafe_allow_html=True,
        )
        

        signal_fig = build_signal_wall_plotly(
            result,
            display_trial,
            visible_channels,
            view_mode=signal_view,
            playhead_sec=playhead_sec,
            window_sec=float(st.session_state.mi_window_sec),
            live_mode=str(st.session_state.mi_live_mode),
        )

        st.plotly_chart(
            signal_fig,
            config={
                "displayModeBar": False,
                "scrollZoom": False,
                "responsive": True,
            },
        )



        st.markdown("</div>", unsafe_allow_html=True)

    lower_left, lower_right = st.columns([1.0, 1.0], gap="medium")
    with lower_left:
        st.markdown(
            "<div class='card'><div class='card-title'>Scalp Snapshot</div>"
            "<div class='card-sub'>Sensor-space amplitude at the current playhead.</div>",
            unsafe_allow_html=True,
        )
        st.pyplot(
            plot_topography(result, display_trial, playhead_sec, highlight_channels=highlight_channels),
            clear_figure=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with lower_right:
        st.markdown(
            "<div class='card'><div class='card-title'>Electrode Relevance</div>"
            "<div class='card-sub'>Dynamic weight for the live trial, phase, and region.</div>",
            unsafe_allow_html=True,
        )
        rank_tabs = st.tabs(["Ranking", "Clusters"])
        with rank_tabs[0]:
            display_frame = view_model.electrode_frame[
                ["channel", "dynamic_weight", "static_weight", "channel_family", "cluster"]
            ].rename(columns={"dynamic_weight": "current_weight"})
            st.dataframe(display_frame, hide_index=True, use_container_width=True)
        with rank_tabs[1]:
            st.pyplot(plot_electrode_cluster_map(result, highlight_channels=highlight_channels), clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _safe_compare_label(result: MotorIntentionResult, preferred_label: str, primary_label: str) -> str:
    options = [str(label) for label in result.metadata["class_schema"] if str(label) != primary_label]
    if not options:
        return primary_label
    if preferred_label in options:
        return preferred_label
    return options[0]


def _render_compare_tab(result: MotorIntentionResult, default_compare: str) -> None:
    if not result.trials:
        st.info("No trial data available for comparison.")
        return

    schema = [str(label) for label in result.metadata["class_schema"]]
    primary_label = str(st.selectbox("Primary Class", schema, index=0))
    compare_label = _safe_compare_label(result, default_compare, primary_label)
    compare_options = [label for label in schema if label != primary_label] or schema
    comparison_label = str(
        st.selectbox(
            "Compare Against",
            compare_options,
            index=compare_options.index(compare_label) if compare_label in compare_options else 0,
        )
    )
    phase_name = str(st.selectbox("Phase", PHASE_NAMES, index=2))

    left_col, right_col = st.columns(2, gap="medium")
    with left_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.pyplot(plot_class_summary_bars(result, phase_name), clear_figure=True)
        st.pyplot(plot_class_topography(result, primary_label, phase_name), clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.pyplot(plot_topography_difference(result, primary_label, comparison_label, phase_name), clear_figure=True)
        st.pyplot(plot_channel_delta_bars(result, primary_label, comparison_label, phase_name, top_n=8), clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_data_tab(result: MotorIntentionResult) -> None:
    payloads = _download_payloads(result)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Export</div>", unsafe_allow_html=True)
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Metadata JSON",
        data=payloads["metadata"],
        file_name="motor_intention_metadata.json",
        mime="application/json",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "Epochs NPZ",
        data=payloads["epochs"],
        file_name="motor_intention_epochs.npz",
        mime="application/octet-stream",
        use_container_width=True,
    )
    download_cols[2].download_button(
        "Continuous NPZ",
        data=payloads["continuous"],
        file_name="motor_intention_continuous.npz",
        mime="application/octet-stream",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    left_col, right_col = st.columns(2, gap="medium")
    with left_col:
        st.markdown("<div class='card'><div class='card-title'>Decoder Readiness</div>", unsafe_allow_html=True)
        st.dataframe(build_decoder_readiness_table(result), hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right_col:
        st.markdown("<div class='card'><div class='card-title'>Class Balance</div>", unsafe_allow_html=True)
        st.dataframe(class_balance_frame(result), hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Trial Timeline", expanded=False):
        st.altair_chart(
            build_timeline_chart(result.trials, selected_trial_id=int(st.session_state.mi_trial_id)),
            use_container_width=True,
        )
    with st.expander("Projection Matrix", expanded=False):
        st.pyplot(plot_projection_matrix(result), clear_figure=True)
    with st.expander("Trial Metadata", expanded=False):
        st.dataframe(pd.DataFrame(result.trial_metadata), hide_index=True, use_container_width=True)
    with st.expander("Event Log", expanded=False):
        st.dataframe(pd.DataFrame(result.events), hide_index=True, use_container_width=True)
    with st.expander("Effective Config", expanded=False):
        st.json(
            {
                "class_schema": result.metadata["class_schema"],
                "layout_name": result.metadata["layout_name"],
                "reference": result.metadata["reference"],
                "projection_info": result.metadata["projection_info"],
                "source_count": len(result.metadata["source_names"]),
                "channel_count": len(result.metadata["channel_names"]),
                "epoch_window": result.metadata["config_snapshot"]["export"],
                "generator_version": result.metadata.get("generator_version"),
            },
            expanded=False,
        )


def main() -> None:
    _inject_css()

    stored_result = st.session_state.get("motor_result")
    typed_result = stored_result if isinstance(stored_result, MotorIntentionResult) else None
    stored_config = st.session_state.get("motor_config")
    active_config = stored_config if isinstance(stored_config, MotorIntentionConfig) else _default_config()
    st.session_state.motor_config = active_config

    _initialize_state(typed_result)
    sidebar = _render_sidebar(typed_result, active_config)

    if sidebar.submitted and sidebar.new_config is not None:
        st.session_state.motor_config = sidebar.new_config
        with st.spinner("Generating synthetic EEG dataset..."):
            new_result, warning_messages = _cached_run(sidebar.new_config)
        st.session_state.motor_result = new_result
        st.session_state.motor_warnings = warning_messages
        st.session_state.mi_trial_id = 0
        st.session_state.mi_region = default_region_for_trial(new_result.trials[0]) if new_result.trials else "rest"
        st.session_state.mi_playing = False
        st.session_state._reset_mi_playback = True
        st.rerun()
        

    for warning_message in st.session_state.get("motor_warnings", []):
        st.warning(warning_message)

    result = st.session_state.get("motor_result")
    if not isinstance(result, MotorIntentionResult) or not result.trials:
        _render_empty_state()
        return

    _render_live_console_fragment(
        signal_view=sidebar.signal_view,
        monitor_slots=sidebar.monitor_slots,
        follow_trial_region=sidebar.follow_trial_region,
    )

    bottom_tabs = st.tabs(["Compare Classes", "Data & Export"])
    with bottom_tabs[0]:
        _render_compare_tab(result, sidebar.compare_label)
    with bottom_tabs[1]:
        _render_data_tab(result)


if __name__ == "__main__":
    main()
