from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.backend import Backend, RNGManager
from motor_intention.configs import ProjectionConfig
from motor_intention.layouts import ElectrodeLayout
from motor_intention.sources import source_definition_map


@dataclass
class SourceToSensorProjector:
    layout: ElectrodeLayout
    source_names: list[str]
    config: ProjectionConfig
    backend_name: str = "numpy"
    seed: int = 42

    def build_mixing_matrix(self) -> np.ndarray:
        source_defs = source_definition_map()
        matrix = np.zeros((len(self.layout), len(self.source_names)), dtype=float)
        for col, source_name in enumerate(self.source_names):
            source = source_defs[source_name]
            for row, spec in enumerate(self.layout.specs):
                dx = spec.x - source.center_x
                dy = spec.y - source.center_y
                dist2 = dx * dx + dy * dy
                weight = np.exp(-dist2 / max(2.0 * source.spread * source.spread, 1e-6))
                weight = self._apply_hemisphere_attenuation(
                    weight,
                    source_name,
                    source.center_x,
                    spec.hemisphere,
                    spec.is_midline,
                )
                weight = self._apply_named_boost(weight, source_name, spec.name)
                matrix[row, col] = max(self.config.floor_weight, weight)

        if self.config.normalize_columns:
            col_max = np.max(matrix, axis=0, keepdims=True) + 1e-12
            matrix = matrix / col_max
        return matrix

    def project(self, source_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        backend = Backend(self.backend_name)
        rng = RNGManager(backend, seed=self.seed)
        mixing_matrix = self.build_mixing_matrix()

        if backend.name == "jax":
            mixing = backend.array(mixing_matrix)
            signal = backend.asarray(source_signal)
            projected = backend.xp.matmul(mixing, signal)
            noise = rng.normal(0.0, self.config.sensor_noise_uv * 1e-6, size=projected.shape)
            projected = projected + noise
            return backend.to_numpy(projected), mixing_matrix

        signal_np = np.asarray(source_signal, dtype=float)
        projected_np = mixing_matrix @ signal_np
        noise_np = np.asarray(
            rng.normal(0.0, self.config.sensor_noise_uv * 1e-6, size=projected_np.shape),
            dtype=float,
        )
        projected_np = projected_np + noise_np
        return projected_np, mixing_matrix

    @staticmethod
    def _apply_named_boost(weight: float, source_name: str, electrode_name: str) -> float:
        boost_map = {
            # Upper-limb sources are intentionally more lateral.
            "left_upper_limb_motor": {
                "C3": 1.25,
                "FC3": 1.15,
                "CP3": 1.15,
                "C1": 1.10,
                "FC1": 1.08,
                "CP1": 1.08,
                "Cz": 1.05,
            },
            "right_upper_limb_motor": {
                "C4": 1.25,
                "FC4": 1.15,
                "CP4": 1.15,
                "C2": 1.10,
                "FC2": 1.08,
                "CP2": 1.08,
                "Cz": 1.05,
            },
            # Lower-limb sources are intentionally more medial and midline-heavy.
            "left_lower_limb_motor": {
                "FCz": 1.18,
                "Cz": 1.22,
                "CPz": 1.14,
                "FC1": 1.10,
                "C1": 1.12,
                "CP1": 1.10,
            },
            "right_lower_limb_motor": {
                "FCz": 1.18,
                "Cz": 1.22,
                "CPz": 1.14,
                "FC2": 1.10,
                "C2": 1.12,
                "CP2": 1.10,
            },
            "sma": {"FCz": 1.20, "Cz": 1.20, "CPz": 1.10, "Fz": 1.10},
            "left_premotor": {"FC3": 1.20, "FC1": 1.08, "FC5": 1.08},
            "right_premotor": {"FC4": 1.20, "FC2": 1.08, "FC6": 1.08},
            "left_s1": {"CP3": 1.20, "CP1": 1.08, "C3": 1.08, "CP5": 1.05},
            "right_s1": {"CP4": 1.20, "CP2": 1.08, "C4": 1.08, "CP6": 1.05},
            "frontal_background": {"Fz": 1.10, "FCz": 1.08, "AF7": 1.08, "AF8": 1.08},
            "posterior_alpha": {"P3": 1.10, "P4": 1.10, "P7": 1.05, "P8": 1.05, "O1": 1.20, "Oz": 1.20, "O2": 1.20},
        }
        return weight * boost_map.get(source_name, {}).get(electrode_name, 1.0)

    @staticmethod
    def _apply_hemisphere_attenuation(
        weight: float,
        source_name: str,
        source_center_x: float,
        electrode_hemisphere: str,
        is_midline: bool,
    ) -> float:
        if abs(source_center_x) < 0.05 or is_midline:
            return weight
        cross_factor = 0.72 if "lower_limb" in source_name else 0.45
        same_factor = 1.02 if "lower_limb" in source_name else 1.05
        if source_center_x < 0 and electrode_hemisphere == "right":
            return weight * cross_factor
        if source_center_x > 0 and electrode_hemisphere == "left":
            return weight * cross_factor
        if source_center_x < 0 and electrode_hemisphere == "left":
            return weight * same_factor
        if source_center_x > 0 and electrode_hemisphere == "right":
            return weight * same_factor
        return weight
