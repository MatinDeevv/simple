"""Render a 50-chart, provenance-backed quant-research visual pack.

The dark, high-contrast presentation is intentionally editorial, but every
figure is calculated from a stored local artifact and remains diagnostic-only.
It makes no claim of quantum advantage, market ontology, alpha, or tradability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.core.contracts import canonical_pair_order


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"
DEFAULT_OUTPUT = ROOT / "artifacts" / "figures" / "quantum" / "aura_pack_v1"
PAIRS = canonical_pair_order(ROOT)
N = len(PAIRS)
TOTAL_CHARTS = 50

NAVY = "#061321"
PANEL = "#0d2237"
INK = "#ecf5ff"
MUTED = "#9eb5cc"
GRID = "#55718b"
CYAN = "#37d7ff"
AMBER = "#ffbd3c"
VIOLET = "#bd8cff"
CORAL = "#ff7979"
TEAL = "#55e0b3"
PALETTE = (CYAN, AMBER, VIOLET, CORAL, TEAL, "#7da5ff", "#f08bc3", "#c8ef5e", "#ff9855", "#7ce6de")


@dataclass(frozen=True)
class Chart:
    number: int
    title: str
    source: str
    path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": NAVY,
        "axes.facecolor": PANEL,
        "axes.edgecolor": "#4d6b85",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 10,
    })


class Pack:
    """Own chart numbering, consistent styling, and immutable provenance."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.charts: list[Chart] = []
        self.sources: set[Path] = set()

    def figure(self, slug: str, title: str, source: Path, *, size: tuple[float, float] = (11.8, 7.2),
               layout: tuple[int, int] = (1, 1)) -> tuple[plt.Figure, np.ndarray]:
        number = len(self.charts) + 1
        fig, axes = plt.subplots(*layout, figsize=size, squeeze=False)
        fig.suptitle(f"AURA QUANT RESEARCH  /  {number:02d} OF {TOTAL_CHARTS:02d}",
                     color=CYAN, x=0.015, y=0.987, ha="left", fontsize=9, fontweight="bold")
        fig.text(0.985, 0.987, "CAUSAL ARTIFACT · DIAGNOSTIC ONLY", color=AMBER,
                 ha="right", va="top", fontsize=8, fontweight="bold")
        self._pending = (number, slug, title, source, fig)
        return fig, axes

    def finish(self, fig: plt.Figure, source: Path) -> Path:
        number, slug, title, expected_source, expected_fig = self._pending
        if expected_fig is not fig or expected_source != source:
            raise RuntimeError("chart provenance does not match its pending figure")
        fig.text(0.012, 0.012, f"Source: {source.relative_to(ROOT).as_posix()} · observational research visual; no alpha or promotion claim",
                 color=MUTED, fontsize=7.7)
        path = self.output / f"{number:02d}_{slug}.png"
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        self.charts.append(Chart(number, title, source.relative_to(ROOT).as_posix(), path))
        self.sources.add(source)
        return path


def polish(ax: plt.Axes, title: str, *, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, color=GRID)


def heatmap(ax: plt.Axes, values: np.ndarray, labels: list[str], title: str, *,
            cmap: str = "coolwarm", symmetric: bool = False, annotate: bool = False) -> None:
    limit = float(np.nanquantile(np.abs(values), 0.98)) if symmetric else None
    if symmetric:
        limit = max(limit or 0.0, 1.0e-12)
        image = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest", aspect="auto")
    else:
        image = ax.imshow(values, cmap=cmap, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title(title, loc="left", pad=14)
    if annotate and len(labels) <= 10:
        for row in range(len(labels)):
            for column in range(len(labels)):
                ax.text(column, row, f"{values[row, column]:+.2f}", ha="center", va="center",
                        color="#061321" if abs(values[row, column]) < 0.65 else "white", fontsize=6.4)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.035)


def coupling_tensor() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = DERIVED / "coupling_estimates.parquet"
    frame = pd.read_parquet(path)
    columns = pd.MultiIndex.from_product([range(N), range(N)])
    wide = frame.pivot(index="update_time", columns=["affected_index", "source_index"], values="c_ij_s_minus_2")
    wide = wide.reindex(columns=columns).sort_index()
    return frame, pd.DatetimeIndex(wide.index), wide.to_numpy(dtype=float).reshape(-1, N, N)


def mps_prefix_stats() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    locations = {
        8: DERIVED / "quantum_mps_chi08_5k" / "quantum_mps_minute.parquet",
        16: DERIVED / "quantum_mps_chi16_5k" / "quantum_mps_minute.parquet",
        32: DERIVED / "quantum_mps_chi32_1k" / "quantum_mps_minute.parquet",
        64: DERIVED / "quantum_mps_chi64_500" / "quantum_mps_minute.parquet",
    }
    discarded: list[float] = []
    entropies: list[float] = []
    for _chi, path in locations.items():
        frame = pd.read_parquet(path)
        start = int(frame["row_index"].min())
        prefix = frame[(frame["row_index"] < start + 500) & (frame["reason"] == "updated")]
        if len(prefix) != 474:
            raise RuntimeError("bond-dimension comparison lost its common causal prefix")
        discarded.append(float(prefix["discarded_weight"].sum()))
        entropies.append(float(prefix["max_schmidt_entropy"].max()))
    return np.asarray(list(locations)), np.asarray(discarded), np.asarray(entropies)


def render_coupling(pack: Pack) -> None:
    source = DERIVED / "coupling_estimates.parquet"
    diagnostics_source = DERIVED / "coupling_diagnostics.parquet"
    frame, times, matrices = coupling_tensor()
    diagnostics = pd.read_parquet(diagnostics_source).sort_values("update_time")
    latest, mean, std = matrices[-1], matrices.mean(axis=0), matrices.std(axis=0)
    for slug, title, matrix, cmap, symmetric in (
        ("coupling_latest_field", "Latest causal coupling field Cᵢⱼ", latest, "RdBu_r", True),
        ("coupling_mean_field", "Mean causal coupling field across all stored updates", mean, "RdBu_r", True),
        ("coupling_variability_field", "Coupling-field temporal variability (standard deviation)", std, "magma", False),
    ):
        fig, axes = pack.figure(slug, title, source)
        heatmap(axes[0, 0], matrix * 1e5, list(PAIRS), title, cmap=cmap, symmetric=symmetric, annotate=slug != "coupling_variability_field")
        axes[0, 0].set_xlabel("source instrument j")
        axes[0, 0].set_ylabel("affected instrument i")
        pack.finish(fig, source)

    fig, axes = pack.figure("coupling_absolute_strength", "Mean absolute coupling strength over time", source)
    axes[0, 0].plot(times, np.mean(np.abs(matrices), axis=(1, 2)) * 1e5, color=CYAN, lw=1.3)
    polish(axes[0, 0], "Mean |Cᵢⱼ| over causal daily updates", xlabel="update time", ylabel="mean |Cᵢⱼ| × 10⁵ [s⁻²]")
    pack.finish(fig, source)

    fig, axes = pack.figure("coupling_max_and_change", "Coupling field amplitude and update-to-update change", diagnostics_source)
    ax = axes[0, 0]
    ax.plot(diagnostics["update_time"], diagnostics["matrix_abs_max_s_minus_2"] * 1e5, color=CYAN, label="max |Cᵢⱼ|")
    ax.plot(diagnostics["update_time"], diagnostics["matrix_delta_fro_s_minus_2"] * 1e5, color=AMBER, label="Δ Frobenius")
    polish(ax, "Causal coupling scale diagnostics", xlabel="update time", ylabel="× 10⁵ [s⁻²]")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    pack.finish(fig, diagnostics_source)

    fig, axes = pack.figure("coupling_condition_number", "Coupling estimator condition number", diagnostics_source)
    axes[0, 0].semilogy(diagnostics["update_time"], np.maximum(diagnostics["condition_number"], 1.0), color=VIOLET, lw=1.1)
    polish(axes[0, 0], "Window conditioning through causal updates", xlabel="update time", ylabel="condition number (log scale)")
    pack.finish(fig, diagnostics_source)

    triangle_columns = [("t1_pre_abs_corr", "t1_post_abs_corr", "triangle 1"),
                        ("t2_pre_abs_corr", "t2_post_abs_corr", "triangle 2"),
                        ("t3_pre_abs_corr", "t3_post_abs_corr", "triangle 3")]
    fig, axes = pack.figure("triangle_constraint_effect", "Triangle-residual constraint effect", diagnostics_source)
    x = np.arange(len(triangle_columns))
    before = [float(diagnostics[left].mean()) for left, _right, _name in triangle_columns]
    after = [float(diagnostics[right].mean()) for _left, right, _name in triangle_columns]
    axes[0, 0].bar(x - 0.18, before, width=0.36, color=CORAL, label="before constraint")
    axes[0, 0].bar(x + 0.18, after, width=0.36, color=TEAL, label="after constraint")
    axes[0, 0].set_xticks(x, [name for _left, _right, name in triangle_columns])
    polish(axes[0, 0], "Mean absolute arithmetic-residual correlation", ylabel="mean absolute correlation")
    axes[0, 0].legend(frameon=False)
    pack.finish(fig, diagnostics_source)

    for slug, title, values in (
        ("coupling_outgoing_latest", "Latest outgoing coupling strength by source", np.sum(np.abs(latest), axis=0)),
        ("coupling_incoming_latest", "Latest incoming coupling strength by affected instrument", np.sum(np.abs(latest), axis=1)),
    ):
        order = np.argsort(values)
        fig, axes = pack.figure(slug, title, source)
        axes[0, 0].barh(np.asarray(PAIRS)[order], values[order] * 1e5, color=[PALETTE[index] for index in order])
        polish(axes[0, 0], title, xlabel="sum |C| × 10⁵ [s⁻²]")
        pack.finish(fig, source)

    off_diagonal = [(abs(latest[row, column]), row, column, latest[row, column])
                    for row in range(N) for column in range(N) if row != column]
    top = sorted(off_diagonal, reverse=True)[:15]
    labels = [f"{PAIRS[row]} ← {PAIRS[column]}" for _strength, row, column, _value in top][::-1]
    values = [value * 1e5 for _strength, _row, _column, value in top][::-1]
    fig, axes = pack.figure("coupling_top_edges", "Largest latest directed coupling edges", source)
    axes[0, 0].barh(labels, values, color=[CORAL if value < 0 else CYAN for value in values])
    polish(axes[0, 0], "Directed Cᵢⱼ edges, sorted by absolute magnitude", xlabel="Cᵢⱼ × 10⁵ [s⁻²]")
    pack.finish(fig, source)


def render_lindblad(pack: Pack) -> None:
    source = DERIVED / "quantum_lindblad_daily.parquet"
    frame = pd.read_parquet(source).sort_values("timestamp")
    probabilities = frame[["p_eurusd", "p_usdjpy", "p_usdcnh"]].to_numpy(dtype=float)
    names = ("EURUSD", "USDJPY", "USDCNH")
    fig, axes = pack.figure("lindblad_populations", "Lindblad qutrit state populations", source)
    for index, name in enumerate(names):
        axes[0, 0].plot(frame["timestamp"], probabilities[:, index], label=name, color=PALETTE[index], lw=1.45)
    polish(axes[0, 0], "Three-state population trajectory", xlabel="timestamp", ylabel="Born probability")
    axes[0, 0].legend(frameon=False, ncol=3)
    pack.finish(fig, source)

    fig, axes = pack.figure("lindblad_state_plane", "Lindblad qutrit state plane", source, size=(10.5, 8.0))
    x = probabilities[:, 1] + 0.5 * probabilities[:, 2]
    y = np.sqrt(3.0) * probabilities[:, 2] / 2.0
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0], [0.0, 0.0]])
    axes[0, 0].plot(triangle[:, 0], triangle[:, 1], color=MUTED)
    axes[0, 0].plot(x, y, color="#77a9ca", lw=0.8, alpha=0.6)
    points = axes[0, 0].scatter(x, y, c=frame["measurement_shannon_entropy"], cmap="viridis", s=25)
    axes[0, 0].text(-0.02, -0.04, "EURUSD", ha="right")
    axes[0, 0].text(1.02, -0.04, "USDJPY", ha="left")
    axes[0, 0].text(0.5, 0.91, "USDCNH", ha="center")
    axes[0, 0].set_aspect("equal"); axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    axes[0, 0].set_title("Barycentric probability trajectory", loc="left", pad=14)
    fig.colorbar(points, ax=axes[0, 0], label="measurement Shannon entropy")
    pack.finish(fig, source)

    for slug, title, values, ylabel, color, logy in (
        ("lindblad_entropy", "Lindblad measurement entropy", frame["measurement_shannon_entropy"], "Shannon entropy", AMBER, False),
        ("lindblad_min_eigenvalue", "Lindblad minimum density eigenvalue", frame["min_eigenvalue"], "minimum eigenvalue", TEAL, False),
        ("lindblad_completeness_error", "Lindblad instrument-completeness error", np.abs(frame["instrument_completeness_error"]), "absolute error", VIOLET, True),
    ):
        fig, axes = pack.figure(slug, title, source)
        values = np.asarray(values, dtype=float)
        if logy:
            axes[0, 0].semilogy(frame["timestamp"], np.maximum(values, 1e-18), color=color, lw=1.2)
        else:
            axes[0, 0].plot(frame["timestamp"], values, color=color, lw=1.35)
        polish(axes[0, 0], title, xlabel="timestamp", ylabel=ylabel)
        pack.finish(fig, source)

    fig, axes = pack.figure("lindblad_trace_hermiticity", "Lindblad trace and Hermiticity errors", source)
    axes[0, 0].semilogy(frame["timestamp"], np.maximum(np.abs(frame["trace_error"]), 1e-18), color=CYAN, label="trace error")
    axes[0, 0].semilogy(frame["timestamp"], np.maximum(np.abs(frame["hermitian_error"]), 1e-18), color=CORAL, label="Hermiticity error")
    polish(axes[0, 0], "Numerical invariants", xlabel="timestamp", ylabel="absolute error (log scale)")
    axes[0, 0].legend(frameon=False)
    pack.finish(fig, source)

    fig, axes = pack.figure("lindblad_purity_proxy", "Lindblad diagonal purity proxy", source)
    axes[0, 0].plot(frame["timestamp"], np.square(probabilities).sum(axis=1), color=CYAN, lw=1.35)
    polish(axes[0, 0], "Σᵢ pᵢ² from measured qutrit populations", xlabel="timestamp", ylabel="diagonal purity proxy")
    pack.finish(fig, source)


def render_trajectories(pack: Pack) -> None:
    source = DERIVED / "quantum_trajectories_daily.parquet"
    frame = pd.read_parquet(source).sort_values("timestamp")
    probabilities = frame[["p_eurusd_mean", "p_usdjpy_mean", "p_usdcnh_mean"]].to_numpy(dtype=float)
    fig, axes = pack.figure("trajectory_populations", "Quantum-trajectory ensemble populations", source)
    for index, name in enumerate(("EURUSD", "USDJPY", "USDCNH")):
        axes[0, 0].plot(frame["timestamp"], probabilities[:, index], label=name, color=PALETTE[index])
    polish(axes[0, 0], "Ensemble mean local populations", xlabel="timestamp", ylabel="mean probability")
    axes[0, 0].legend(frameon=False, ncol=3)
    pack.finish(fig, source)
    for slug, title, column, color, logy in (
        ("trajectory_purity", "Trajectory density purity", "density_purity_mean", CYAN, False),
        ("trajectory_entropy", "Trajectory Born entropy", "born_shannon_entropy_mean_probability", AMBER, False),
        ("trajectory_jumps", "Trajectory dephasing jump events by day", "jump_events", VIOLET, False),
        ("trajectory_min_eigenvalue", "Trajectory minimum eigenvalue", "min_eigenvalue", TEAL, False),
    ):
        fig, axes = pack.figure(slug, title, source)
        values = np.asarray(frame[column], dtype=float)
        if logy:
            axes[0, 0].semilogy(frame["timestamp"], np.maximum(np.abs(values), 1e-18), color=color)
        elif slug == "trajectory_jumps":
            axes[0, 0].bar(frame["timestamp"], values, width=0.75, color=color, alpha=0.85)
        else:
            axes[0, 0].plot(frame["timestamp"], values, color=color, lw=1.35)
        polish(axes[0, 0], title, xlabel="timestamp", ylabel=column.replace("_", " "))
        pack.finish(fig, source)
    fig, axes = pack.figure("trajectory_entropy_purity", "Trajectory entropy versus purity", source)
    scatter = axes[0, 0].scatter(frame["born_shannon_entropy_mean_probability"], frame["density_purity_mean"],
                                 c=np.arange(len(frame)), cmap="plasma", s=24)
    polish(axes[0, 0], "State-mixing diagnostic", xlabel="Born Shannon entropy", ylabel="density purity")
    fig.colorbar(scatter, ax=axes[0, 0], label="chronological day index")
    pack.finish(fig, source)


def render_mps(pack: Pack) -> None:
    source = DERIVED / "quantum_mps_chi64_500" / "quantum_mps_minute.parquet"
    frame = pd.read_parquet(source)
    frame = frame[frame["reason"] == "updated"].reset_index(drop=True)
    activity = [f"activity_{pair.lower()}" for pair in PAIRS]
    neutral = [f"born_neutral_{pair.lower()}" for pair in PAIRS]
    born = [f"born_{state}_{pair.lower()}" for pair in PAIRS for state in ("down", "neutral", "up")]
    timeline = np.arange(len(frame))
    fig, axes = pack.figure("mps_activity_lines", "χ=64 MPS local activity allocation", source)
    for index, pair in enumerate(PAIRS):
        axes[0, 0].plot(timeline, frame[activity[index]], color=PALETTE[index], lw=0.8, label=pair)
    polish(axes[0, 0], "Normalized local Born non-neutral mass", xlabel="causal update index", ylabel="activity share")
    axes[0, 0].legend(frameon=False, ncol=5, fontsize=7)
    pack.finish(fig, source)
    fig, axes = pack.figure("mps_activity_heatmap", "χ=64 MPS activity field", source, size=(13, 7.5))
    heatmap(axes[0, 0], frame[activity].to_numpy(dtype=float).T[:, ::5], list(PAIRS), "Pair activity through the causal prefix", cmap="magma")
    axes[0, 0].set_xlabel("every fifth causal update"); axes[0, 0].set_ylabel("instrument")
    pack.finish(fig, source)
    composition = np.column_stack((
        frame[[f"born_down_{pair.lower()}" for pair in PAIRS]].mean(axis=1),
        frame[neutral].mean(axis=1),
        frame[[f"born_up_{pair.lower()}" for pair in PAIRS]].mean(axis=1),
    ))
    fig, axes = pack.figure("mps_state_composition", "χ=64 MPS average local qutrit composition", source)
    axes[0, 0].stackplot(timeline, composition.T, labels=("down", "neutral", "up"), colors=(CORAL, MUTED, TEAL), alpha=0.9)
    polish(axes[0, 0], "Average local Born composition", xlabel="causal update index", ylabel="mean probability")
    axes[0, 0].legend(frameon=False, ncol=3)
    pack.finish(fig, source)
    for slug, title, column, color, logy in (
        ("mps_mean_schmidt_entropy", "χ=64 MPS mean Schmidt entropy", "mean_schmidt_entropy", AMBER, False),
        ("mps_max_schmidt_entropy", "χ=64 MPS maximum Schmidt entropy", "max_schmidt_entropy", VIOLET, False),
        ("mps_discarded_weight", "χ=64 MPS per-update discarded weight", "discarded_weight", CORAL, True),
        ("mps_bond_dimension", "χ=64 MPS observed bond dimension", "max_bond_dimension", CYAN, False),
    ):
        fig, axes = pack.figure(slug, title, source)
        values = np.asarray(frame[column], dtype=float)
        if logy:
            axes[0, 0].semilogy(timeline, np.maximum(values, 1e-18), color=color, lw=0.85)
        else:
            axes[0, 0].plot(timeline, values, color=color, lw=1.0)
        polish(axes[0, 0], title, xlabel="causal update index", ylabel=column.replace("_", " "))
        pack.finish(fig, source)
    fig, axes = pack.figure("mps_truncation_distribution", "χ=64 MPS truncation distribution", source)
    axes[0, 0].hist(np.log10(np.maximum(frame["discarded_weight"], 1e-18)), bins=45, color=CORAL, alpha=0.85)
    polish(axes[0, 0], "Log discarded weight per TEBD update", xlabel="log₁₀(discarded weight)", ylabel="update count")
    pack.finish(fig, source)
    labels = [f"{pair}\n{state}" for pair in PAIRS for state in ("down", "neutral", "up")]
    fig, axes = pack.figure("mps_qutrit_correlation_tensor", "χ=64 MPS qutrit Born-probability correlation tensor", source, size=(14, 12))
    heatmap(axes[0, 0], frame[born].corr().to_numpy(dtype=float), labels, "30 local qutrit population channels", symmetric=True)
    for boundary in range(3, len(labels), 3):
        axes[0, 0].axhline(boundary - 0.5, color=NAVY, lw=1.0); axes[0, 0].axvline(boundary - 0.5, color=NAVY, lw=1.0)
    pack.finish(fig, source)
    chis, discarded, entropy = mps_prefix_stats()
    fig, axes = pack.figure("mps_bond_ladder", "10-qutrit MPS / TEBD bond-dimension ladder", source, size=(14.5, 6.5), layout=(1, 2))
    axes[0, 0].plot(chis, discarded, marker="o", color=CYAN, lw=2.2, ms=8); axes[0, 0].set_xscale("log", base=2); axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(chis, [str(int(value)) for value in chis]); polish(axes[0, 0], "Truncation falls with retained bond capacity", xlabel="maximum bond χ", ylabel="cumulative discarded weight")
    axes[0, 1].plot(chis, entropy, marker="D", color=AMBER, lw=2.2, ms=8); axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_xticks(chis, [str(int(value)) for value in chis]); polish(axes[0, 1], "Higher χ retains more state structure", xlabel="maximum bond χ", ylabel="maximum Schmidt entropy")
    pack.finish(fig, source)


def render_reservoir(pack: Pack) -> None:
    source = DERIVED / "quantum_reservoir_daily.parquet"
    frame = pd.read_parquet(source).sort_values("timestamp")
    z_columns = [f"mean_z_q{index}" for index in range(N)]
    zz_columns = [f"mean_zz_ring_q{index}_q{(index + 1) % N}" for index in range(N)]
    for slug, title, columns, labels, cmap in (
        ("reservoir_z_field", "10-qubit reservoir Pauli-Z field", z_columns, list(PAIRS), "coolwarm"),
        ("reservoir_zz_field", "10-qubit reservoir ring-ZZ field", zz_columns, [f"q{index}-q{(index + 1) % N}" for index in range(N)], "coolwarm"),
    ):
        fig, axes = pack.figure(slug, title, source, size=(13, 7.5))
        heatmap(axes[0, 0], frame[columns].to_numpy(dtype=float).T, labels, title, cmap=cmap, symmetric=True)
        axes[0, 0].set_xlabel("daily update index"); axes[0, 0].set_ylabel("observable")
        pack.finish(fig, source)
    for slug, title, column, color, logy in (
        ("reservoir_global_parity", "Reservoir global Z parity", "mean_global_z_parity", VIOLET, False),
        ("reservoir_norm_error", "Reservoir state-norm error", "mean_state_norm_error", CORAL, True),
        ("reservoir_oos_brier", "Reservoir daily OOS Brier score", "oos_brier_score", AMBER, False),
    ):
        subset = frame.dropna(subset=[column])
        fig, axes = pack.figure(slug, title, source)
        values = np.asarray(subset[column], dtype=float)
        if logy:
            axes[0, 0].semilogy(subset["timestamp"], np.maximum(values, 1e-18), color=color)
        else:
            axes[0, 0].plot(subset["timestamp"], values, color=color, lw=1.3)
        polish(axes[0, 0], title, xlabel="timestamp", ylabel=column.replace("_", " "))
        pack.finish(fig, source)
    all_columns = z_columns + zz_columns + ["mean_global_z_parity"]
    fig, axes = pack.figure("reservoir_observable_correlation", "Reservoir observable correlation matrix", source, size=(13, 11))
    labels = [f"Z{index}" for index in range(N)] + [f"ZZ{index}" for index in range(N)] + ["parity"]
    heatmap(axes[0, 0], frame[all_columns].corr().to_numpy(dtype=float), labels, "21 measured reservoir observables", symmetric=True)
    pack.finish(fig, source)


def render_kernel(pack: Pack) -> None:
    source = DERIVED / "quantum_kernel_daily.parquet"
    frame = pd.read_parquet(source).sort_values("date")
    fig, axes = pack.figure("kernel_brier_comparison", "Quantum-kernel Brier comparison", source)
    for column, label, color in (("model_brier", "kernel model", CYAN), ("prior_brier", "frozen prior", AMBER), ("uniform_brier", "uniform", MUTED)):
        axes[0, 0].plot(frame["date"], frame[column], label=label, color=color, lw=1.3)
    polish(axes[0, 0], "Daily categorical-probability Brier scores", xlabel="date", ylabel="Brier score")
    axes[0, 0].legend(frameon=False, ncol=3)
    pack.finish(fig, source)
    fig, axes = pack.figure("kernel_brier_delta", "Kernel model minus frozen-prior Brier", source)
    delta = frame["model_brier"] - frame["prior_brier"]
    axes[0, 0].axhline(0.0, color=MUTED, lw=0.8); axes[0, 0].fill_between(frame["date"], delta, 0.0, where=delta >= 0, color=CORAL, alpha=0.5); axes[0, 0].fill_between(frame["date"], delta, 0.0, where=delta < 0, color=TEAL, alpha=0.5)
    polish(axes[0, 0], "Positive means the kernel is worse", xlabel="date", ylabel="model Brier − prior Brier")
    pack.finish(fig, source)
    fig, axes = pack.figure("kernel_top1_volume", "Kernel top-1 accuracy and scored volume", source)
    ax = axes[0, 0]; twin = ax.twinx(); ax.plot(frame["date"], frame["top1_accuracy"], color=CYAN, label="top-1 accuracy"); twin.bar(frame["date"], frame["oos_scored_minutes"], color=AMBER, alpha=0.38, label="scored minutes")
    polish(ax, "Daily OOS diagnostic coverage", xlabel="date", ylabel="top-1 accuracy"); twin.set_ylabel("OOS scored minutes", color=AMBER)
    pack.finish(fig, source)


def render_aer(pack: Pack) -> None:
    source = DERIVED / "quantum_aer_noise_calibration.parquet"
    frame = pd.read_parquet(source).sort_values("timestamp")
    fig, axes = pack.figure("aer_error_envelope", "Aer synthetic-noise error envelope", source)
    axes[0, 0].plot(frame["timestamp"], frame["observable_mean_absolute_error"], color=CYAN, label="mean absolute error")
    axes[0, 0].plot(frame["timestamp"], frame["observable_max_absolute_error"], color=CORAL, label="max absolute error")
    polish(axes[0, 0], "Ideal-versus-noisy observable deviation", xlabel="sample timestamp", ylabel="absolute expectation error")
    axes[0, 0].legend(frameon=False)
    pack.finish(fig, source)
    z_columns = [f"z_{pair.lower()}" for pair in PAIRS]
    fig, axes = pack.figure("aer_input_field", "Aer causal standardized-input field", source, size=(13, 7.5))
    heatmap(axes[0, 0], frame[z_columns].to_numpy(dtype=float).T, list(PAIRS), "Encoded standardized return inputs", cmap="coolwarm", symmetric=True)
    axes[0, 0].set_xlabel("sample index"); axes[0, 0].set_ylabel("instrument")
    pack.finish(fig, source)
    ideal = np.concatenate([frame[f"ideal_z_q{index}"].to_numpy() for index in range(N)])
    noisy = np.concatenate([frame[f"noisy_z_q{index}"].to_numpy() for index in range(N)])
    fig, axes = pack.figure("aer_ideal_noisy_scatter", "Aer ideal-versus-noisy Pauli-Z observables", source)
    axes[0, 0].scatter(ideal, noisy, c=np.repeat(np.arange(N), len(frame)), cmap="turbo", s=18, alpha=0.8)
    axes[0, 0].plot([-1, 1], [-1, 1], color=INK, lw=0.8, ls="--")
    polish(axes[0, 0], "Noisy output against ideal reference", xlabel="ideal ⟨Z⟩", ylabel="noisy ⟨Z⟩")
    pack.finish(fig, source)


def render_tomography(pack: Pack) -> None:
    source = DERIVED / "quantum_process_tomography.parquet"
    frame = pd.read_parquet(source).sort_values("timestamp")
    fig, axes = pack.figure("tomography_channel_errors", "Process-tomography channel checks", source)
    for column, label, color in (("trace_preservation_error", "trace preservation", CYAN), ("choi_hermitian_error", "Choi Hermiticity", CORAL), ("instrument_completeness_error", "instrument completeness", AMBER)):
        axes[0, 0].semilogy(frame["timestamp"], np.maximum(np.abs(frame[column]), 1e-18), label=label, color=color, lw=1.0)
    polish(axes[0, 0], "Physical-channel numerical invariants", xlabel="timestamp", ylabel="absolute error (log scale)")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=7)
    pack.finish(fig, source)
    fig, axes = pack.figure("tomography_transfer_radius", "Tomography traceless-transfer spectral radius", source)
    axes[0, 0].plot(frame["timestamp"], frame["traceless_transfer_spectral_radius"], color=VIOLET, lw=1.1)
    axes[0, 0].axhline(1.0, color=AMBER, lw=0.8, ls="--")
    polish(axes[0, 0], "Channel contraction diagnostic", xlabel="timestamp", ylabel="spectral radius")
    pack.finish(fig, source)
    columns = ["h_diag_eurusd", "h_diag_usdjpy", "h_diag_usdcnh"]
    fig, axes = pack.figure("tomography_hamiltonian_field", "Tomography Hamiltonian diagonal field", source, size=(13, 7.5))
    heatmap(axes[0, 0], frame[columns].to_numpy(dtype=float).T[:, ::4], ["EURUSD", "USDJPY", "USDCNH"], "Sampled Hamiltonian diagonal terms", cmap="coolwarm", symmetric=True)
    axes[0, 0].set_xlabel("every fourth tomography sample"); axes[0, 0].set_ylabel("qutrit basis channel")
    pack.finish(fig, source)


def render_dynamics(pack: Pack) -> None:
    sources = [DERIVED / f"dynamics_params_{pair}.parquet" for pair in ("EURUSD", "USDJPY", "USDCNH")]
    frames = {path.stem.removeprefix("dynamics_params_"): pd.read_parquet(path).sort_values("timestamp") for path in sources}
    primary = sources[0]
    fig, axes = pack.figure("dynamics_mass", "Causal dynamical mass estimates", primary)
    for index, (pair, frame) in enumerate(frames.items()):
        axes[0, 0].plot(frame["timestamp"], frame["m"], color=PALETTE[index], lw=1.0, label=pair)
    polish(axes[0, 0], "Mass = clipped inverse causal realized volatility", xlabel="timestamp", ylabel="dimensionless mass")
    axes[0, 0].legend(frameon=False, ncol=3)
    pack.finish(fig, primary)
    fig, axes = pack.figure("dynamics_restoring_damping", "Causal restoring and damping coefficients", primary, size=(13, 7.2), layout=(1, 3))
    for index, (pair, frame) in enumerate(frames.items()):
        ax = axes[0, index]
        ax.plot(frame["timestamp"], frame["kappa"], color=CYAN, lw=0.8, label="κ")
        ax.plot(frame["timestamp"], frame["gamma"], color=AMBER, lw=0.8, label="γ")
        polish(ax, pair, xlabel="timestamp", ylabel="coefficient [s⁻² / s⁻¹]")
        ax.legend(frameon=False, fontsize=7)
    pack.finish(fig, primary)


def write_index(pack: Pack) -> None:
    cards = "\n".join(
        f'<a class="card" href="{chart.path.name}"><img src="{chart.path.name}" loading="lazy"><span>{chart.number:02d}. {chart.title}</span><small>{chart.source}</small></a>'
        for chart in pack.charts
    )
    html = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Aura Quant Research Pack</title>
<style>body{{margin:0;background:#061321;color:#ecf5ff;font-family:Arial,sans-serif}}header{{padding:34px 5%;border-bottom:1px solid #26445f}}h1{{margin:0;color:#37d7ff;letter-spacing:.08em}}p{{color:#9eb5cc}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px;padding:28px 5%}}.card{{background:#0d2237;border:1px solid #26445f;border-radius:8px;padding:10px;text-decoration:none;color:#ecf5ff}}.card:hover{{border-color:#37d7ff}}img{{width:100%;display:block;border-radius:4px}}span{{display:block;margin-top:10px;font-weight:bold}}small{{display:block;color:#9eb5cc;margin-top:5px;font-size:11px}}</style></head>
<body><header><h1>AURA QUANT RESEARCH / 50 ARTIFACTS</h1><p>All visuals are computed from local causal research artifacts. Diagnostic only: no alpha, execution, quantum-advantage, or physical-market claim.</p></header><main class=\"grid\">{cards}</main></body></html>"""
    (pack.output / "index.html").write_text(html, encoding="utf-8")


def write_manifest(pack: Pack) -> None:
    manifest = {
        "version": "aura-quant-research-pack-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "renderer_sha256": sha256_file(Path(__file__).resolve()),
        "instrument_order": list(PAIRS),
        "claim_boundary": "computed local research visuals only; no alpha, execution, physical-market quantum, or promotion claim",
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sorted(pack.sources)},
        "charts": [{"number": chart.number, "title": chart.title, "source": chart.source,
                    "file": chart.path.name, "sha256": sha256_file(chart.path)} for chart in pack.charts],
    }
    (pack.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    pack = Pack(output)
    render_coupling(pack)
    render_lindblad(pack)
    render_trajectories(pack)
    render_mps(pack)
    render_reservoir(pack)
    render_kernel(pack)
    render_aer(pack)
    render_tomography(pack)
    render_dynamics(pack)
    if len(pack.charts) != TOTAL_CHARTS:
        raise RuntimeError(f"expected {TOTAL_CHARTS} charts, created {len(pack.charts)}")
    write_index(pack)
    write_manifest(pack)
    print(f"[AURA_PACK] charts={len(pack.charts)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
