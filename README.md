EEG Synthesis Framework - Architecture Overview

The EEG Synthesis Framework is organized as a modular, pipeline-based system designed to support realistic EEG signal generation, preprocessing, analysis, and visualization. At its core, the framework treats EEG synthesis as a sequence of composable signal transformations rather than a monolithic generator.

I. CORE FRAMEWORK (core/)

backend.py
Provides a unified abstraction layer for numerical backends (NumPy and JAX), standardizing array operations, Fourier transforms, and random number generation.

types.py
Defines foundational data structures and protocols, including:
* Signal context and metadata propagation.
* Block interfaces for stateless processing units.
* Frequency band specifications (BandSpec).

II. PROCESSING COMPONENTS (blocks/)
Blocks are self-contained processing units that operate within the pipeline.

Generators (generators.py)
* Multi-partial band-limited oscillatory signal generation.
* Parameterized colored noise synthesis based on spectral exponents.

Artifacts (artifacts.py)
* Physiologically inspired non-neural signals such as eye blinks (EOG) and muscle activity (EMG).
* Environmental noise including power line interference and baseline drift.

Envelopes (envelopes.py)
* Amplitude modulation via transient burst structures, suitable for simulating phenomena like sleep spindles.

Transforms (transforms.py)
* Cross-frequency interactions, specifically Phase-Amplitude Coupling (PAC).

Spatial Operations (spatial.py)
* Channel-wise transformations including average signal referencing.

Filters (filters.py)
* Standard EEG preprocessing filters (Bandpass, Notch, Lowpass) implemented using numerically stable Second-Order Sections (SOS) via SciPy.

Observables (observables.py)
* Analysis and validation tools, including multi-band power estimation using Welch's method.
* Signal integrity checks (NaN/Inf detection) and diagnostic signal taps.

III. PIPELINE & INFRASTRUCTURE

orchestrator.py
Manages pipeline execution, block ordering, accumulation semantics (add/replace), component tracking, and metadata aggregation.

export.py
Handles serialization and visualization, including CSV/JSON export, MNE-compatible integration, and a comprehensive plotting suite.

IV. USER INTERFACE

gui_app.py
A Streamlit-based interactive dashboard enabling real-time pipeline configuration, signal visualization, artifact injection, and data export.

V. TESTING

test_framework.py
Verifies import integrity and ensures deterministic behavior across runs for a fixed random seed.