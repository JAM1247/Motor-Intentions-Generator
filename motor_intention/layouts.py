from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class ElectrodeSpec:
    name: str
    x: float
    y: float
    hemisphere: str
    region: str
    is_midline: bool
    symmetric_partner: str | None = None
    neighbors: Tuple[str, ...] = ()


class ElectrodeLayout:
    def __init__(self, specs: Iterable[ElectrodeSpec], name: str):
        self.specs = tuple(specs)
        self.name = name
        self._by_name: Dict[str, ElectrodeSpec] = {spec.name: spec for spec in self.specs}
        self._index_by_name: Dict[str, int] = {
            spec.name: idx for idx, spec in enumerate(self.specs)
        }
        if len(self._by_name) != len(self.specs):
            raise ValueError("Electrode names must be unique")

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def channel_names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    @property
    def coordinates(self) -> np.ndarray:
        return np.asarray([(spec.x, spec.y) for spec in self.specs], dtype=float)

    def spec(self, name: str) -> ElectrodeSpec:
        return self._by_name[name]

    def index(self, name: str) -> int:
        return self._index_by_name[name]

    def indices(self, names: Iterable[str]) -> list[int]:
        return [self.index(name) for name in names]

    def neighbors(self, name: str) -> Tuple[str, ...]:
        return self.spec(name).neighbors

    def symmetric_partner(self, name: str) -> str | None:
        return self.spec(name).symmetric_partner

    def frontal_channels(self, count: int = 2) -> list[str]:
        ranked = sorted(self.specs, key=lambda spec: spec.y, reverse=True)
        return [spec.name for spec in ranked[:count]]

    def adjacency_matrix(self) -> np.ndarray:
        n_channels = len(self.specs)
        matrix = np.zeros((n_channels, n_channels), dtype=float)
        for row, spec in enumerate(self.specs):
            for neighbor in spec.neighbors:
                col = self.index(neighbor)
                matrix[row, col] = 1.0
                matrix[col, row] = 1.0
        return matrix
