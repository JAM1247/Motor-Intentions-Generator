"""
Export and I/O utilities for synthetic EEG data.
"""

import numpy as np
import json
import os
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_to_csv(
    signal: np.ndarray,
    times: np.ndarray,
    filename: str,
    ch_names: Optional[List[str]] = None
):
    """Save signal to CSV file"""
    n_channels = signal.shape[0]
    
    if ch_names is None:
        ch_names = [f"EEG{i+1:02d}" for i in range(n_channels)]
    
    header = "time," + ",".join(ch_names)
    data = np.column_stack([times, signal.T])
    
    np.savetxt(filename, data, delimiter=",", header=header, comments="", fmt="%.9e")
    print(f"Saved CSV: {filename}")


def save_events(events: List[Dict], filename: str):
    """Save events to JSON file"""
    def _jsonify(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type not serializable: {type(obj)}")
    
    with open(filename, "w") as f:
        json.dump(events, f, indent=2, default=_jsonify)
    
    print(f"Saved events: {filename}")


def save_metadata(metadata: Dict[str, Any], filename: str):
    """Save metadata to JSON file with proper numpy serialization"""
    def _jsonify(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # For other types, convert to string as fallback
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)
    
    with open(filename, "w") as f:
        json.dump(metadata, f, indent=2, default=_jsonify)
    
    print(f"Saved metadata: {filename}")


def plot_psd(
    raw,
    fmin: float = 0.5,
    fmax: float = 100.0,
    figsize: Tuple = (12, 6),
    max_channels: int = 8
):
    """Plot power spectral density"""
    psd = raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_plot = min(max_channels, len(raw.ch_names))
    for i in range(n_plot):
        ax.semilogy(freqs, psds[i] * 1e12, alpha=0.7, label=f"Ch{i+1}")
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (μV²/Hz)")
    ax.set_title("Power Spectral Density")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    
    return fig


def plot_timeseries(
    signal: np.ndarray,
    times: np.ndarray,
    channel_idx: int = 0,
    figsize: Tuple = (12, 4)
):
    """Plot single channel time series"""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(times, signal[channel_idx] * 1e6, lw=0.8, color='steelblue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (μV)")
    ax.set_title(f"Channel {channel_idx + 1} — Time Series")
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_stacked_channels(
    signal: np.ndarray,
    times: np.ndarray,
    offset_uv: float = 150.0,
    figsize: Tuple = (14, 8)
):
    """Plot all channels stacked"""
    fig, ax = plt.subplots(figsize=figsize)
    
    n_channels = signal.shape[0]
    
    for i in range(n_channels):
        ax.plot(times, signal[i] * 1e6 - i * offset_uv, lw=0.6, label=f"Ch{i+1}")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (μV, stacked)")
    ax.set_title("All Channels — Stacked View")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=8, fontsize=8, loc="upper right")
    
    return fig


def plot_band_powers(
    band_powers: Dict[str, Dict[str, Any]],
    figsize: Tuple = (10, 6)
):
    """Plot power in EEG bands (like your first image)"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Canonical band order
    canonical_order = ["delta", "theta", "alpha", "sigma", "beta", "gamma"]
    
    # Filter to only bands present in data, in canonical order
    bands = [b for b in canonical_order if b in band_powers]
    powers = [band_powers[band]["mean"] for band in bands]
    
    # Color gamma red, others black
    colors = ["red" if band.lower() == "gamma" else "black" for band in bands]
    
    bars = ax.bar(bands, powers, color=colors, alpha=0.8)
    
    ax.set_xlabel("Bands", fontsize=14)
    ax.set_ylabel("Power", fontsize=14)
    ax.set_title("Power in EEG Bands", fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set ticks and labels properly
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands])
    
    return fig


def plot_filtered_comparison(
    original: np.ndarray,
    filtered: np.ndarray,
    times: np.ndarray,
    channel_idx: int = 0,
    title: str = "Brainwave at 30 Hz",
    figsize: Tuple = (14, 5)
):
    """Plot original vs filtered signal (like your second image)"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to µV for consistency with other plots
    orig_uv = original[channel_idx] * 1e6
    filt_uv = filtered[channel_idx] * 1e6
    
    ax.plot(times, orig_uv, lw=1.5, alpha=0.7, label='Original', color='steelblue')
    ax.plot(times, filt_uv, lw=2, alpha=0.9, label='Filtered', color='darkorange')
    
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Amplitude (μV)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    return fig


def plot_band_filtered_signals(
    band_signals: Dict[str, np.ndarray],
    times: np.ndarray,
    channel_idx: int = 0,
    figsize: Tuple = (14, 12)
):
    """Plot filtered signals for each EEG band (like your fifth image)"""
    bands = [
        ("delta", "Delta Band (0.5-4 Hz) Filtered EEG Signal"),
        ("theta", "Theta Band (4-8 Hz) Filtered EEG Signal"),
        ("alpha", "Alpha Band (8-13 Hz) Filtered EEG Signal"),
        ("sigma", "Sigma Band (12-16 Hz) Filtered EEG Signal"),
        ("beta", "Beta Band (13-30 Hz) Filtered EEG Signal"),
        ("gamma", "Gamma Band (30-99 Hz) Filtered EEG Signal"),
    ]
    
    # Convert times to numpy if needed (for JAX compatibility)
    times = np.asarray(times)
    
    fig, axes = plt.subplots(len(bands), 1, figsize=figsize, sharex=True)
    
    for idx, (band_name, title) in enumerate(bands):
        signal_key = f"{band_name}_signal"
        if signal_key in band_signals:
            # Convert to numpy (handles both numpy and JAX arrays)
            signal = np.asarray(band_signals[signal_key])
            
            # Convert to µV for display
            signal_uv = signal[channel_idx] * 1e6
            
            # Validate lengths match
            if len(signal_uv) != len(times):
                raise ValueError(
                    f"Signal length ({len(signal_uv)}) doesn't match times length ({len(times)}) "
                    f"for band {band_name}. This indicates an upstream error."
                )
            
            axes[idx].plot(times, signal_uv, lw=0.8, color='black')
            axes[idx].set_ylabel("Amplitude (μV)", fontsize=10)
            axes[idx].set_title(title, fontsize=11)
            axes[idx].grid(True, alpha=0.3)
            
            # Set reasonable y-limits based on the data
            axes[idx].set_ylim([-np.abs(signal_uv).max() * 1.2, np.abs(signal_uv).max() * 1.2])
    
    axes[-1].set_xlabel("Time (s)", fontsize=12)  # Fixed: was "Sample"
    plt.tight_layout()
    
    return fig


def create_output_bundle(
    result: Dict[str, Any],
    raw,
    output_dir: str,
    prefix: str = "eeg_synth"
):
    """Create complete output bundle with all files"""
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{prefix}.csv")
    save_to_csv(result["signal"], result["times"], csv_path)
    paths["csv"] = csv_path
    
    # Save events
    if result["events"]:
        events_path = os.path.join(output_dir, f"{prefix}_events.json")
        save_events(result["events"], events_path)
        paths["events"] = events_path
    
    # Save FIF
    fif_path = os.path.join(output_dir, f"{prefix}.fif")
    raw.save(fif_path, overwrite=True, verbose=False)
    paths["fif"] = fif_path
    print(f"Saved FIF: {fif_path}")
    
    # Create figures
    figures = []
    
    fig_psd = plot_psd(raw)
    figures.append(fig_psd)
    
    fig_ts = plot_timeseries(result["signal"], result["times"], channel_idx=0)
    figures.append(fig_ts)
    
    fig_stack = plot_stacked_channels(result["signal"], result["times"])
    figures.append(fig_stack)
    
    # Add band power plot if available
    if "metadata" in result:
        for key, value in result["metadata"].items():
            if isinstance(value, dict) and "band_powers" in value:
                fig_bp = plot_band_powers(value["band_powers"])
                figures.append(fig_bp)
                break
    
    # Add band filtered signals if available
    if "components" in result:
        band_signals = {k: v for k, v in result["components"].items() if "_signal" in k}
        if band_signals:
            fig_bands = plot_band_filtered_signals(band_signals, result["times"])
            figures.append(fig_bands)
    
    # Save figures as PNG
    for i, fig in enumerate(figures):
        png_path = os.path.join(output_dir, f"{prefix}_fig{i+1}.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    # Save PDF
    pdf_path = os.path.join(output_dir, f"{prefix}_figures.pdf")
    
    # Recreate figures for PDF
    figures = [
        plot_psd(raw),
        plot_timeseries(result["signal"], result["times"]),
        plot_stacked_channels(result["signal"], result["times"])
    ]
    
    # Add band power plot if available
    if "metadata" in result:
        for key, value in result["metadata"].items():
            if isinstance(value, dict) and "band_powers" in value:
                figures.append(plot_band_powers(value["band_powers"]))
                break
    
    # Add band filtered signals if available
    if "components" in result:
        band_signals = {k: v for k, v in result["components"].items() if "_signal" in k}
        if band_signals:
            figures.append(plot_band_filtered_signals(band_signals, result["times"]))
    
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    
    paths["pdf"] = pdf_path
    print(f"Saved PDF: {pdf_path}")
    
    print(f"\n✓ Output bundle created: {output_dir}")
    
    return paths