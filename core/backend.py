

"""
Backend abstraction layer for NumPy/JAX interoperability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, TypeVar, overload
import numpy as np

from core.types import Array

_T = TypeVar("_T")


class Backend:
    """Backend abstraction for array operations (NumPy or JAX)."""

    def __init__(self, name: str = "numpy"):
        self.name = name

        if name == "numpy":
            self.xp = np
            self.fft = np.fft
            self.jax = None  # sentinel for type checkers

        elif name == "jax":
            try:
                import jax
                import jax.numpy as jnp
            except ImportError as e:
                raise ImportError("JAX not installed. Install with: pip install jax jaxlib") from e

            self.xp = jnp
            self.fft = jnp.fft
            self.jax = jax

        else:
            raise ValueError(f"Unknown backend: {name}")

    # ---------- array creators / ops ----------

    def array(self, x: Any, dtype: Any = None) -> Array:
        return self.xp.array(x, dtype=dtype)

    def asarray(self, x: Any) -> Array:
        return self.xp.asarray(x)

    def zeros(self, shape: int | Tuple[int, ...], dtype: Any = None) -> Array:
        return self.xp.zeros(shape, dtype=dtype)

    def ones(self, shape: int | Tuple[int, ...], dtype: Any = None) -> Array:
        return self.xp.ones(shape, dtype=dtype)

    def arange(self, *args: Any, **kwargs: Any) -> Array:
        return self.xp.arange(*args, **kwargs)

    def linspace(self, *args: Any, **kwargs: Any) -> Array:
        return self.xp.linspace(*args, **kwargs)

    def sin(self, x: Array) -> Array:
        return self.xp.sin(x)

    def cos(self, x: Array) -> Array:
        return self.xp.cos(x)

    def exp(self, x: Array) -> Array:
        return self.xp.exp(x)

    def sqrt(self, x: Array) -> Array:
        return self.xp.sqrt(x)

    def mean(self, x: Array, axis: int | Tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return self.xp.mean(x, axis=axis, keepdims=keepdims)

    def sum(self, x: Array, axis: int | Tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return self.xp.sum(x, axis=axis, keepdims=keepdims)

    # ---------- small helpers to unify NumPy/JAX mutation ----------

    def set_row(self, arr: Array, idx: int, value: Array) -> Array:
        """Set arr[idx] = value for both NumPy and JAX arrays."""
        if self.name == "jax":
            return arr.at[idx].set(value)  # pyright: ignore[reportAttributeAccessIssue]
        arr[idx] = value
        return arr

    def add_row(self, arr: Array, idx: int, value: Array) -> Array:
        """Add arr[idx] += value for both NumPy and JAX arrays."""
        if self.name == "jax":
            return arr.at[idx].add(value)  # pyright: ignore[reportAttributeAccessIssue]
        arr[idx] += value
        return arr

    def set_slice(self, arr: Array, sl: slice, value: Array) -> Array:
        """Set arr[sl] = value for both NumPy and JAX arrays (1D or trailing slice)."""
        if self.name == "jax":
            return arr.at[sl].set(value)  # pyright: ignore[reportAttributeAccessIssue]
        arr[sl] = value
        return arr

    # ---------- helpers used by envelopes / artifacts ----------

    def hanning(self, M: int) -> Array:
        if self.name == "numpy":
            return np.hanning(M)
        n = self.xp.arange(M)
        # avoid divide-by-zero when M==1 (rare but safe)
        denom = self.xp.maximum(M - 1, 1)
        return 0.5 - 0.5 * self.xp.cos(2 * self.xp.pi * n / denom)

    def interp(self, x: Array, xp: Array, fp: Array) -> Array:
        """1D linear interpolation. NumPy: np.interp, JAX: jnp.interp."""
        if self.name == "numpy":
            return np.interp(x, xp, fp)  # type: ignore[arg-type]
        return self.xp.interp(x, xp, fp)

    # ---------- FFT ----------

    def rfft(self, x: Array, n: int | None = None, axis: int = -1) -> Array:
        return self.fft.rfft(x, n=n, axis=axis)

    def irfft(self, x: Array, n: int | None = None, axis: int = -1) -> Array:
        return self.fft.irfft(x, n=n, axis=axis)

    def rfftfreq(self, n: int, d: float = 1.0) -> Array:
        return self.fft.rfftfreq(n, d=d)

    # ---------- Hilbert (analytic signal) ----------

    def hilbert(self, x: Array, axis: int = -1) -> Array:
        """
        Analytic signal via FFT-based Hilbert transform (SciPy-like behavior).
        Returns complex array with the same shape as x.

        Notes:
        - The analytic signal z(t) = x(t) + i*H{x(t)}
        - This is what you want for instantaneous phase: phase = angle(z(t))
        """
        xp = self.xp
        x = xp.asarray(x)
        N = int(x.shape[axis])

        Xf = xp.fft.fft(x, axis=axis)

        # multiplier for positive frequencies
        if self.name == "jax":
            h: Any = xp.zeros((N,), dtype=Xf.dtype)

            if N % 2 == 0:
                h = h.at[0].set(1)
                h = h.at[N // 2].set(1)
                h = h.at[1:N // 2].set(2)
            else:
                h = h.at[0].set(1)
                h = h.at[1:(N + 1) // 2].set(2)

        else:
            # build in NumPy with normal indexing, then convert to xp
            h_np = np.zeros((N,), dtype=np.complex128 if np.iscomplexobj(Xf) else np.float64)
            h_np = h_np.astype(np.asarray(Xf).dtype, copy=False)

            if N % 2 == 0:
                h_np[0] = 1
                h_np[N // 2] = 1
                h_np[1:N // 2] = 2
            else:
                h_np[0] = 1
                h_np[1:(N + 1) // 2] = 2

            h = xp.asarray(h_np)

        # reshape to broadcast along axis
        shape = [1] * x.ndim
        shape[axis] = N
        h = xp.reshape(h, shape)

        Xf = Xf * h
        return xp.fft.ifft(Xf, axis=axis)

    # ---------- conversions ----------

    def to_numpy(self, x: Any) -> np.ndarray | Any:
        if self.name == "jax":
            return np.array(x)
        return x


class RNGManager:
    """Manage random number generation across backends."""

    def __init__(self, backend: Backend, seed: int = 42):
        self.backend = backend

        if backend.name == "numpy":
            self._rng: Any = np.random.default_rng(seed)
            self._key: Any = None
        else:
            self._rng = None
            self._key = backend.jax.random.PRNGKey(seed)  # type: ignore[union-attr]

    @staticmethod
    def _shape(size: int | Tuple[int, ...] | None) -> Tuple[int, ...]:
        if size is None:
            return ()
        if isinstance(size, int):
            return (size,)
        return size

    def _split(self) -> Any:
        """Return a JAX subkey. Only used in the JAX branch."""
        self._key, sub = self.backend.jax.random.split(self._key)  # type: ignore[union-attr]
        return sub

    def uniform(self, low: float = 0.0, high: float = 1.0, size: int | Tuple[int, ...] | None = None) -> Array:
        if self.backend.name == "numpy":
            out = self._rng.uniform(low, high, size)  # type: ignore[union-attr]
            return self.backend.asarray(out)

        key: Any = self._split()
        return self.backend.jax.random.uniform(  # type: ignore[union-attr]
            key, shape=self._shape(size), minval=low, maxval=high
        )

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: int | Tuple[int, ...] | None = None) -> Array:
        if self.backend.name == "numpy":
            out = self._rng.normal(loc, scale, size)  # type: ignore[union-attr]
            return self.backend.asarray(out)

        key: Any = self._split()
        return loc + scale * self.backend.jax.random.normal(  # type: ignore[union-attr]
            key, shape=self._shape(size)
        )

    def integers(self, low: int, high: int | None = None, size: int | Tuple[int, ...] | None = None) -> Array:
        if self.backend.name == "numpy":
            out = self._rng.integers(low, high, size)  # type: ignore[union-attr]
            return self.backend.asarray(out)

        key: Any = self._split()
        hi = low + 1 if high is None else high
        return self.backend.jax.random.randint(  # type: ignore[union-attr]
            key, shape=self._shape(size), minval=low, maxval=hi
        )

    @overload
    def choice(self, a: int, size: int | Tuple[int, ...] | None = None, replace: bool = True) -> Array: ...
    @overload
    def choice(self, a: Sequence[_T], size: int | Tuple[int, ...] | None = None, replace: bool = True) -> Array: ...

    def choice(self, a: int | Sequence[_T], size: int | Tuple[int, ...] | None = None, replace: bool = True) -> Array:
        """Choose random samples from a population."""
        if self.backend.name == "numpy":
            import numpy as _np

            population = _np.arange(a) if isinstance(a, int) else _np.asarray(a)
            out = (
                self._rng.choice(population, replace=replace)  # type: ignore[union-attr]
                if size is None
                else self._rng.choice(population, size=size, replace=replace)  # type: ignore[union-attr]
            )
            return self.backend.asarray(out)

        key: Any = self._split()
        arr = self.backend.xp.arange(a) if isinstance(a, int) else self.backend.asarray(a)
        return self.backend.jax.random.choice(  # type: ignore[union-attr]
            key, arr, shape=self._shape(size), replace=replace
        )

    def exponential(self, scale: float = 1.0, size: int | Tuple[int, ...] | None = None) -> Array:
        if self.backend.name == "numpy":
            out = self._rng.exponential(scale, size)  # type: ignore[union-attr]
            return self.backend.asarray(out)

        key: Any = self._split()
        return self.backend.jax.random.exponential(  # type: ignore[union-attr]
            key, shape=self._shape(size)
        ) * scale
