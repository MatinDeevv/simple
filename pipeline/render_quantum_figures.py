"""Render reproducible quantum/physics diagnostic figures from stored artifacts.

These figures are visualizations of existing local research outputs.  They do
not claim physical quantum market dynamics, quantum advantage, or tradability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data_derived"
DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "quantum"
PAIRS = ("EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCNH", "USDCHF", "EURGBP", "EURJPY", "GBPJPY")
NAVY = "#081525"
PANEL = "#0e2237"
INK = "#e8f2fc"
MUTED = "#9fb3c8"
CYAN = "#37d7ff"
AMBER = "#ffbe3d"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": NAVY,
        "axes.facecolor": PANEL,
        "axes.edgecolor": "#49627b",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.labelsize": 10,
    })


def save(fig: plt.Figure, output: Path, name: str) -> Path:
    path = output / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.012, text, color=MUTED, fontsize=8)


def render_lindblad_state_plane(output: Path) -> Path:
    frame = pd.read_parquet(DERIVED / "quantum_lindblad_daily.parquet").dropna()
    probabilities = frame[["p_eurusd", "p_usdjpy", "p_usdcnh"]].to_numpy(dtype=float)
    # Barycentric projection: each row is a qutrit population vector summing to one.
    x = probabilities[:, 1] + 0.5 * probabilities[:, 2]
    y = np.sqrt(3.0) * probabilities[:, 2] / 2.0
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0], [0.0, 0.0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color="#6d8ca9", lw=1.4)
    for share in (0.25, 0.50, 0.75):
        ax.plot([share / 2.0, 1.0 - share / 2.0], [np.sqrt(3.0) * share / 2.0] * 2,
                color="#29455f", lw=0.7, zorder=0)
    ax.plot(x, y, color="#87b8d9", lw=1.0, alpha=0.52, zorder=1)
    points = ax.scatter(x, y, c=frame["measurement_shannon_entropy"], cmap="viridis", s=23,
                        edgecolor="none", zorder=2)
    ax.scatter(x[0], y[0], c=CYAN, s=75, marker="o", edgecolor=INK, linewidth=0.6, zorder=3)
    ax.scatter(x[-1], y[-1], c=AMBER, s=85, marker="D", edgecolor=INK, linewidth=0.6, zorder=3)
    ax.text(-0.03, -0.055, "EURUSD", ha="right", color=INK, weight="bold")
    ax.text(1.03, -0.055, "USDJPY", ha="left", color=INK, weight="bold")
    ax.text(0.5, np.sqrt(3.0) / 2.0 + 0.045, "USDCNH", ha="center", color=INK, weight="bold")
    colorbar = fig.colorbar(points, ax=ax, pad=0.03)
    colorbar.set_label("measurement Shannon entropy")
    ax.set_title("Lindblad qutrit state plane")
    ax.set_xlabel("barycentric probability coordinate")
    ax.set_ylabel("barycentric probability coordinate")
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, 0.98)
    ax.set_xticks([])
    ax.set_yticks([])
    footer(fig, "Stored quantum_lindblad_daily.parquet | data-conditioned likelihood filter; diagnostic only, not physical-market evidence")
    return save(fig, output, "01_lindblad_qutrit_state_plane.png")


def render_reservoir_correlation(output: Path) -> Path:
    frame = pd.read_parquet(DERIVED / "quantum_reservoir_daily.parquet")
    columns = [f"mean_z_q{index}" for index in range(10)]
    correlation = frame[columns].corr().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11.2, 9.4))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    labels = [f"q{index}\n{pair}" for index, pair in enumerate(PAIRS)]
    ax.set_xticks(range(10), labels, rotation=38, ha="right")
    ax.set_yticks(range(10), labels)
    for row in range(10):
        for column in range(10):
            value = correlation[row, column]
            ax.text(column, row, f"{value:+.2f}", ha="center", va="center",
                    color="#07131f" if abs(value) < 0.65 else "white", fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation of daily mean Pauli-Z observables")
    ax.set_title("10-qubit reservoir observable correlation map")
    footer(fig, "Stored quantum_reservoir_daily.parquet | fixed circuit software representation; no quantum advantage or trading claim")
    return save(fig, output, "02_reservoir_observable_correlation.png")


def render_aer_noise_heatmap(output: Path) -> Path:
    frame = pd.read_parquet(DERIVED / "quantum_aer_noise_calibration.parquet").sort_values("timestamp")
    labels: list[str] = []
    rows: list[np.ndarray] = []
    for index in range(10):
        labels.append(f"Z q{index}")
        rows.append((frame[f"noisy_z_q{index}"] - frame[f"ideal_z_q{index}"]).to_numpy(dtype=float))
    for index in range(10):
        right = (index + 1) % 10
        labels.append(f"ZZ q{index}-q{right}")
        rows.append((frame[f"noisy_zz_q{index}_{right}"] - frame[f"ideal_zz_q{index}_{right}"]).to_numpy(dtype=float))
    labels.append("global Z parity")
    rows.append((frame["noisy_global_z_parity"] - frame["ideal_global_z_parity"]).to_numpy(dtype=float))
    errors = np.stack(rows)
    limit = max(float(np.quantile(np.abs(errors), 0.98)), 0.01)
    fig, ax = plt.subplots(figsize=(14, 8.5))
    image = ax.imshow(errors, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.axhline(9.5, color="#d7e7f7", lw=1.1)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    tick_locations = list(range(0, len(frame), max(1, len(frame) // 6)))
    ax.set_xticks(tick_locations, [pd.Timestamp(frame.iloc[index]["timestamp"]).strftime("%d %b\n%H:%M")
                                   for index in tick_locations])
    ax.set_xlabel("causal replay sample")
    ax.set_title("Aer synthetic-noise observable error field")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("noisy minus ideal expectation value")
    footer(fig, "Stored quantum_aer_noise_calibration.parquet | local Aer density-matrix simulation with declared synthetic noise, not QPU calibration")
    return save(fig, output, "03_aer_synthetic_noise_heatmap.png")


def render_causal_coupling_field(output: Path) -> Path:
    frame = pd.read_parquet(DERIVED / "coupling_estimates.parquet")
    latest = frame[frame["update_time"] == frame["update_time"].max()]
    matrix = latest.pivot(index="affected_symbol", columns="source_symbol", values="c_ij_s_minus_2").reindex(
        index=PAIRS, columns=PAIRS).to_numpy(dtype=float)
    off_diagonal = matrix[~np.eye(len(PAIRS), dtype=bool)]
    limit = max(float(np.quantile(np.abs(off_diagonal), 0.98)), 1.0e-9)
    fig, ax = plt.subplots(figsize=(11.4, 9.5))
    image = ax.imshow(matrix * 1.0e5, cmap="RdBu_r", vmin=-limit * 1.0e5, vmax=limit * 1.0e5,
                      interpolation="nearest")
    ax.set_xticks(range(10), PAIRS, rotation=45, ha="right")
    ax.set_yticks(range(10), PAIRS)
    ax.set_xlabel("source instrument j")
    ax.set_ylabel("affected instrument i")
    ax.set_title(f"Latest causal specific-acceleration coupling field Cᵢⱼ ({pd.Timestamp(latest['update_time'].iloc[0]).date()})")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Cᵢⱼ × 10⁵  [s⁻²]")
    footer(fig, "Stored coupling_estimates.parquet | causal daily estimate in the project state model; not an execution signal")
    return save(fig, output, "04_causal_coupling_field.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    style()
    paths = [
        render_lindblad_state_plane(output),
        render_reservoir_correlation(output),
        render_aer_noise_heatmap(output),
        render_causal_coupling_field(output),
    ]
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
