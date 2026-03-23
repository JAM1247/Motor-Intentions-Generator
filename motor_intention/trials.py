from __future__ import annotations

from dataclasses import dataclass

from motor_intention.configs import TrialConfig


@dataclass(frozen=True)
class MotorTrial:
    trial_id: int
    flat_label: str
    effector_family: str
    side: str | None
    joint_subset: str | None
    start_sec: float
    cue_onset_sec: float
    imagery_start_sec: float
    imagery_end_sec: float
    end_sec: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "trial_id": self.trial_id,
            "flat_label": self.flat_label,
            "effector_family": self.effector_family,
            "side": self.side or "",
            "joint_subset": self.joint_subset or "",
            "start_sec": self.start_sec,
            "cue_onset_sec": self.cue_onset_sec,
            "imagery_start_sec": self.imagery_start_sec,
            "imagery_end_sec": self.imagery_end_sec,
            "end_sec": self.end_sec,
        }


def parse_flat_label(label: str) -> tuple[str, str | None, str | None]:
    if label == "rest":
        return "rest", None, None

    parts = label.split("_", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Unsupported limb-intention label '{label}'")

    side, family = parts
    if side not in {"left", "right"} or family not in {"arm", "leg"}:
        raise ValueError(f"Unsupported limb-intention label '{label}'")

    return family, side, None


def build_trial_schedule(config: TrialConfig) -> list[MotorTrial]:
    schedule: list[MotorTrial] = []
    trial_duration = config.trial_duration_sec
    for trial_id in range(config.n_trials):
        flat_label = config.class_cycle[trial_id % len(config.class_cycle)]
        effector_family, side, joint_subset = parse_flat_label(flat_label)
        start_sec = trial_id * trial_duration
        cue_onset_sec = start_sec + config.baseline_sec
        imagery_start_sec = cue_onset_sec + config.preparation_sec
        imagery_end_sec = imagery_start_sec + config.imagery_sec
        end_sec = imagery_end_sec + config.recovery_sec
        schedule.append(
            MotorTrial(
                trial_id=trial_id,
                flat_label=flat_label,
                effector_family=effector_family,
                side=side,
                joint_subset=joint_subset,
                start_sec=start_sec,
                cue_onset_sec=cue_onset_sec,
                imagery_start_sec=imagery_start_sec,
                imagery_end_sec=imagery_end_sec,
                end_sec=end_sec,
            )
        )
    return schedule


def total_duration(schedule: list[MotorTrial]) -> float:
    if not schedule:
        return 0.0
    return schedule[-1].end_sec
