"""Render reproducible quantum/physics diagnostic figures from stored artifacts.

These figures are visualizations of existing local research outputs.  They do
not claim physical quantum market dynamics, quantum advantage, or tradability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on optional environment
    raise SystemExit(
        "matplotlib is required. Use .venv-quantum\\Scripts\\python.exe "
        "engine\\visualization\\render_quantum_figures.py after installing requirements-quantum.txt."
    ) from exc
import numpy as np
import pandas as pd

from engine.core.contracts import canonical_pair_order


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"
DEFAULT_OUTPUT = ROOT / "artifacts" / "figures" / "quantum"
PAIRS = canonical_pair_order(ROOT)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output: Path, paths: list[Path]) -> Path:
    sources = [DERIVED / "quantum_lindblad_daily.parquet", DERIVED / "quantum_reservoir_daily.parquet",
               DERIVED / "quantum_aer_noise_calibration.parquet", DERIVED / "coupling_estimates.parquet"]
    mps_sources = [
        DERIVED / "quantum_mps_chi08_5k" / "quantum_mps_minute.parquet",
        DERIVED / "quantum_mps_chi16_5k" / "quantum_mps_minute.parquet",
        DERIVED / "quantum_mps_chi32_1k" / "quantum_mps_minute.parquet",
        DERIVED / "quantum_mps_chi64_500" / "quantum_mps_minute.parquet",
    ]
    sources.extend(path for path in mps_sources if path.exists())
    payload = {
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "instrument_index_order": list(PAIRS),
        "source_artifacts": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in sources
        },
        "output_sha256": {path.name: sha256_file(path) for path in paths},
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest


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


def render_mps_qutrit_correlation_tensor(output: Path) -> Path:
    """Render 30 local qutrit Born channels from the completed chi=64 replay."""
    path = DERIVED / "quantum_mps_chi64_500" / "quantum_mps_minute.parquet"
    frame = pd.read_parquet(path)
    frame = frame[frame["reason"] == "updated"].copy()
    columns = [f"born_{state}_{pair.lower()}" for pair in PAIRS
               for state in ("down", "neutral", "up")]
    correlation = frame[columns].corr().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(14.2, 12.0))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    site_ticks = [3 * index + 1 for index in range(len(PAIRS))]
    ax.set_xticks(site_ticks, PAIRS, rotation=45, ha="right")
    ax.set_yticks(site_ticks, PAIRS)
    for boundary in range(3, len(columns), 3):
        ax.axhline(boundary - 0.5, color=NAVY, lw=1.0)
        ax.axvline(boundary - 0.5, color=NAVY, lw=1.0)
    ax.set_xlabel("local qutrit Born channels: down / neutral / up")
    ax.set_ylabel("local qutrit Born channels: down / neutral / up")
    ax.set_title("χ=64 MPS qutrit Born-probability correlation tensor (30 channels)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation across 474 causal updates")
    footer(fig, "Stored quantum_mps_chi64_500 replay | 10 qutrit sites yield 30 measured local-population channels; experimental non-promotion")
    return save(fig, output, "05_mps_qutrit_correlation_tensor.png")


def mps_prefix_statistics(chi: int, path: Path) -> dict[str, float]:
    frame = pd.read_parquet(path)
    start = int(frame["row_index"].min())
    prefix = frame[(frame["row_index"] < start + 500) & (frame["reason"] == "updated")]
    if len(prefix) != 474:
        raise RuntimeError(f"chi={chi} does not contain the required common 500-row prefix")
    return {
        "chi": float(chi),
        "discarded_weight_total": float(prefix["discarded_weight"].sum()),
        "max_schmidt_entropy": float(prefix["max_schmidt_entropy"].max()),
        "max_norm_error": float(prefix["norm_error"].max()),
    }


def render_mps_bond_ladder(output: Path) -> Path:
    """Show the exact same-prefix truncation reduction across chi values."""
    sources = {
        8: DERIVED / "quantum_mps_chi08_5k" / "quantum_mps_minute.parquet",
        16: DERIVED / "quantum_mps_chi16_5k" / "quantum_mps_minute.parquet",
        32: DERIVED / "quantum_mps_chi32_1k" / "quantum_mps_minute.parquet",
        64: DERIVED / "quantum_mps_chi64_500" / "quantum_mps_minute.parquet",
    }
    rows = [mps_prefix_statistics(chi, path) for chi, path in sources.items()]
    chis = np.asarray([row["chi"] for row in rows])
    discarded = np.asarray([row["discarded_weight_total"] for row in rows])
    entropy = np.asarray([row["max_schmidt_entropy"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6))
    axes[0].plot(chis, discarded, marker="o", color=CYAN, lw=2.2, ms=8)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(chis, [str(int(value)) for value in chis])
    axes[0].set_xlabel("maximum MPS bond dimension χ")
    axes[0].set_ylabel("cumulative discarded weight (474 shared updates)")
    axes[0].set_title("Truncation falls as retained bond capacity increases")
    axes[0].grid(alpha=0.22, color="#6f8da7")
    axes[1].plot(chis, entropy, marker="D", color=AMBER, lw=2.2, ms=8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(chis, [str(int(value)) for value in chis])
    axes[1].set_xlabel("maximum MPS bond dimension χ")
    axes[1].set_ylabel("maximum sampled Schmidt entropy")
    axes[1].set_title("Higher χ retains more entanglement structure")
    axes[1].grid(alpha=0.22, color="#6f8da7")
    fig.suptitle("10-qutrit MPS / TEBD bond-dimension convergence ladder", fontsize=17, fontweight="bold")
    footer(fig, "Same 500-row causal prefix and fixed seed for χ={8,16,32,64}; χ=64 still has nonzero truncation, so convergence is incomplete and non-promotable")
    return save(fig, output, "06_mps_bond_dimension_ladder.png")


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
    if (DERIVED / "quantum_mps_chi64_500" / "quantum_mps_minute.parquet").exists():
        paths.extend((render_mps_qutrit_correlation_tensor(output), render_mps_bond_ladder(output)))
    paths.append(write_manifest(output, paths))
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
