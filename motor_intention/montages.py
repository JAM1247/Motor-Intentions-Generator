from __future__ import annotations

from dataclasses import replace

import numpy as np

from motor_intention.layouts import ElectrodeLayout, ElectrodeSpec


def _with_neighbors(specs: list[ElectrodeSpec], k: int = 4) -> list[ElectrodeSpec]:
    coords = np.asarray([(spec.x, spec.y) for spec in specs], dtype=float)
    updated: list[ElectrodeSpec] = []
    for idx, spec in enumerate(specs):
        dists = np.sum((coords - coords[idx]) ** 2, axis=1)
        order = np.argsort(dists)
        neighbors = tuple(specs[i].name for i in order[1 : k + 1])
        updated.append(replace(spec, neighbors=neighbors))
    return updated


def _motor_21_layout() -> ElectrodeLayout:
    specs = _with_neighbors(
        [
            ElectrodeSpec("FC5", -0.90, 0.35, "left", "premotor", False, "FC6"),
            ElectrodeSpec("FC3", -0.60, 0.35, "left", "premotor", False, "FC4"),
            ElectrodeSpec("FC1", -0.30, 0.35, "left", "premotor", False, "FC2"),
            ElectrodeSpec("FCz", 0.00, 0.38, "midline", "premotor", True, None),
            ElectrodeSpec("FC2", 0.30, 0.35, "right", "premotor", False, "FC1"),
            ElectrodeSpec("FC4", 0.60, 0.35, "right", "premotor", False, "FC3"),
            ElectrodeSpec("FC6", 0.90, 0.35, "right", "premotor", False, "FC5"),
            ElectrodeSpec("C5", -0.90, 0.00, "left", "motor", False, "C6"),
            ElectrodeSpec("C3", -0.60, 0.00, "left", "motor", False, "C4"),
            ElectrodeSpec("C1", -0.30, 0.00, "left", "motor", False, "C2"),
            ElectrodeSpec("Cz", 0.00, 0.02, "midline", "motor", True, None),
            ElectrodeSpec("C2", 0.30, 0.00, "right", "motor", False, "C1"),
            ElectrodeSpec("C4", 0.60, 0.00, "right", "motor", False, "C3"),
            ElectrodeSpec("C6", 0.90, 0.00, "right", "motor", False, "C5"),
            ElectrodeSpec("CP5", -0.90, -0.35, "left", "somatosensory", False, "CP6"),
            ElectrodeSpec("CP3", -0.60, -0.35, "left", "somatosensory", False, "CP4"),
            ElectrodeSpec("CP1", -0.30, -0.35, "left", "somatosensory", False, "CP2"),
            ElectrodeSpec("CPz", 0.00, -0.38, "midline", "somatosensory", True, None),
            ElectrodeSpec("CP2", 0.30, -0.35, "right", "somatosensory", False, "CP1"),
            ElectrodeSpec("CP4", 0.60, -0.35, "right", "somatosensory", False, "CP3"),
            ElectrodeSpec("CP6", 0.90, -0.35, "right", "somatosensory", False, "CP5"),
        ]
    )
    return ElectrodeLayout(specs, name="motor_21")


def _motor_14_layout() -> ElectrodeLayout:
    specs = _with_neighbors(
        [
            ElectrodeSpec("Fz", 0.00, 0.62, "midline", "frontal", True, None),
            ElectrodeSpec("FC3", -0.45, 0.35, "left", "premotor", False, "FC4"),
            ElectrodeSpec("FC4", 0.45, 0.35, "right", "premotor", False, "FC3"),
            ElectrodeSpec("C3", -0.45, 0.00, "left", "motor", False, "C4"),
            ElectrodeSpec("Cz", 0.00, 0.02, "midline", "motor", True, None),
            ElectrodeSpec("C4", 0.45, 0.00, "right", "motor", False, "C3"),
            ElectrodeSpec("CP3", -0.45, -0.30, "left", "somatosensory", False, "CP4"),
            ElectrodeSpec("CP4", 0.45, -0.30, "right", "somatosensory", False, "CP3"),
            ElectrodeSpec("P3", -0.45, -0.55, "left", "parietal", False, "P4"),
            ElectrodeSpec("P4", 0.45, -0.55, "right", "parietal", False, "P3"),
            ElectrodeSpec("P7", -0.82, -0.50, "left", "parietal", False, "P8"),
            ElectrodeSpec("P8", 0.82, -0.50, "right", "parietal", False, "P7"),
            ElectrodeSpec("T7", -0.92, 0.00, "left", "temporal", False, "T8"),
            ElectrodeSpec("T8", 0.92, 0.00, "right", "temporal", False, "T7"),
        ]
    )
    return ElectrodeLayout(specs, name="motor_14")


_LAYOUT_FACTORIES = {
    "motor_21": _motor_21_layout,
    "motor_14": _motor_14_layout,
}


def get_layout(name: str) -> ElectrodeLayout:
    try:
        return _LAYOUT_FACTORIES[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown montage '{name}'") from exc
