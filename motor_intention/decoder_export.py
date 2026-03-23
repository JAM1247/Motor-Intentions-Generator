from __future__ import annotations

from typing import Iterable

import numpy as np

from motor_intention.trials import MotorTrial


def build_labels(trials: Iterable[MotorTrial]) -> np.ndarray:
    return np.asarray([trial.flat_label for trial in trials])


def build_trial_metadata(trials: Iterable[MotorTrial]) -> list[dict[str, float | int | str]]:
    return [trial.to_dict() for trial in trials]


def build_events(trials: Iterable[MotorTrial]) -> list[dict[str, float | int | str]]:
    events: list[dict[str, float | int | str]] = []
    for trial in trials:
        events.append(
            {
                "type": "cue",
                "onset": trial.cue_onset_sec,
                "duration": max(0.0, trial.imagery_start_sec - trial.cue_onset_sec),
                "trial_id": trial.trial_id,
                "flat_label": trial.flat_label,
                "effector_family": trial.effector_family,
                "side": trial.side or "",
                "joint_subset": trial.joint_subset or "",
            }
        )
        events.append(
            {
                "type": "imagery",
                "onset": trial.imagery_start_sec,
                "duration": max(0.0, trial.imagery_end_sec - trial.imagery_start_sec),
                "trial_id": trial.trial_id,
                "flat_label": trial.flat_label,
                "effector_family": trial.effector_family,
                "side": trial.side or "",
                "joint_subset": trial.joint_subset or "",
            }
        )
        events.append(
            {
                "type": "recovery",
                "onset": trial.imagery_end_sec,
                "duration": max(0.0, trial.end_sec - trial.imagery_end_sec),
                "trial_id": trial.trial_id,
                "flat_label": trial.flat_label,
                "effector_family": trial.effector_family,
                "side": trial.side or "",
                "joint_subset": trial.joint_subset or "",
            }
        )
    return events


def extract_epochs(
    signal: np.ndarray,
    trials: Iterable[MotorTrial],
    sfreq: float,
    start_sec: float,
    end_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    signal_np = np.asarray(signal, dtype=float)
    epoch_start = int(round(start_sec * sfreq))
    epoch_end = int(round(end_sec * sfreq))
    if epoch_end <= epoch_start:
        raise ValueError("Epoch end must be greater than epoch start")

    epochs: list[np.ndarray] = []
    for trial in trials:
        start_idx = int(round(trial.start_sec * sfreq)) + epoch_start
        end_idx = int(round(trial.start_sec * sfreq)) + epoch_end
        if end_idx > signal_np.shape[1]:
            raise ValueError("Epoch window exceeds signal length")
        epochs.append(signal_np[:, start_idx:end_idx])

    epoch_times = np.arange(epoch_end - epoch_start, dtype=float) / sfreq + start_sec
    return np.stack(epochs, axis=0), epoch_times
