from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np
import pandas as pd

from cpdp.oversampling_validation.parameter_plot import _configure_matplotlib


def _load_ranking(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Parameter ranking file not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    frame["mean_rank"] = pd.to_numeric(frame["mean_rank"], errors="raise")
    frame["selection_position"] = pd.to_numeric(
        frame["selection_position"], errors="raise"
    )
    if len(frame) < 2 or not np.isfinite(frame["mean_rank"]).all():
        raise ValueError(f"{path.name} does not contain valid candidate rankings")
    return frame


def _draw_panel(
    axis: Any,
    labels: list[str],
    ranks: np.ndarray,
    stage_best_index: int,
    final_index: int,
    title: str,
    xlabel: str,
) -> None:
    positions = np.arange(len(labels), dtype=float)
    line_color = "#2F6254"
    final_color = "#C83E4D"
    axis.plot(
        positions,
        ranks,
        color=line_color,
        linewidth=2.2,
        marker="o",
        markersize=6,
    )
    axis.scatter(
        [positions[stage_best_index]],
        [ranks[stage_best_index]],
        s=120,
        facecolors="white",
        edgecolors="#182433",
        linewidths=1.7,
        zorder=4,
    )
    axis.axvline(
        positions[final_index],
        color=final_color,
        linewidth=1.0,
        linestyle="--",
        alpha=0.65,
    )
    axis.scatter(
        [positions[final_index]],
        [ranks[final_index]],
        marker="*",
        s=190,
        color=final_color,
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
    )
    for position, rank in zip(positions, ranks):
        axis.annotate(
            f"{rank:.2f}",
            (position, rank),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#4B5563",
            fontsize=9,
        )
    axis.set_title(title, loc="left")
    axis.set_xlabel(xlabel)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.invert_yaxis()
    axis.grid(axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A949E")
    axis.spines["bottom"].set_color("#8A949E")
    axis.tick_params(colors="#374151")


def create_manifold_formula_figure(
    output_dir: Path,
    winner: Mapping[str, float],
    *,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if dpi < 72:
        raise ValueError("dpi must be at least 72")

    interval = _load_ranking(
        output_path / "stage_01_manifold_interval_ranking.csv",
        {
            "manifold_lower_quantile",
            "manifold_upper_quantile",
            "mean_rank",
            "selection_position",
        },
    ).sort_values(["manifold_lower_quantile", "manifold_upper_quantile"])
    neighbors = _load_ranking(
        output_path / "stage_02_manifold_neighbors_ranking.csv",
        {"manifold_neighbors", "mean_rank", "selection_position"},
    ).sort_values("manifold_neighbors")
    terms = _load_ranking(
        output_path / "stage_03_formula_max_terms_ranking.csv",
        {"formula_max_terms", "mean_rank", "selection_position"},
    ).sort_values("formula_max_terms")

    lower = float(winner["manifold_lower_quantile"])
    upper = float(winner["manifold_upper_quantile"])
    interval_mask = np.isclose(interval["manifold_lower_quantile"], lower) & np.isclose(
        interval["manifold_upper_quantile"], upper
    )
    neighbor_mask = np.isclose(
        neighbors["manifold_neighbors"], float(winner["manifold_neighbors"])
    )
    terms_mask = np.isclose(
        terms["formula_max_terms"], float(winner["formula_max_terms"])
    )
    if not interval_mask.any() or not neighbor_mask.any() or not terms_mask.any():
        raise ValueError("Final parameter combination is absent from stage rankings")

    plt = _configure_matplotlib()
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 5.6), constrained_layout=True)
    panels = (
        (
            interval,
            [
                f"{row.manifold_lower_quantile:g}-{row.manifold_upper_quantile:g}"
                for row in interval.itertuples()
            ],
            int(np.flatnonzero(interval_mask)[0]),
            "流形内部距离分位区间",
            "lower_quantile-upper_quantile",
        ),
        (
            neighbors,
            [f"{int(value)}" for value in neighbors["manifold_neighbors"]],
            int(np.flatnonzero(neighbor_mask)[0]),
            "流形邻居数",
            "manifold.neighbors",
        ),
        (
            terms,
            [f"{int(value)}" for value in terms["formula_max_terms"]],
            int(np.flatnonzero(terms_mask)[0]),
            "物理公式最大项数",
            "formula.max_terms",
        ),
    )
    for axis, (frame, labels, final_index, title, xlabel) in zip(axes, panels):
        stage_best_index = int(np.argmin(frame["selection_position"].to_numpy()))
        _draw_panel(
            axis,
            labels,
            frame["mean_rank"].to_numpy(dtype=float),
            stage_best_index,
            final_index,
            title,
            xlabel,
        )
    axes[0].set_ylabel("综合平均排名（越低越好）")

    from matplotlib.lines import Line2D

    figure.suptitle(
        "IPGDO流形与公式参数敏感性",
        x=0.02,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.legend(
        handles=(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="#182433",
                markeredgewidth=1.7,
                markersize=8,
                label="单参数阶段最优",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="none",
                markerfacecolor="#C83E4D",
                markeredgecolor="white",
                markersize=11,
                label="最终交互选择",
            ),
        ),
        loc="upper right",
        bbox_to_anchor=(0.98, 1.02),
        frameon=False,
        ncol=2,
    )
    figure.text(
        0.5,
        -0.015,
        "参数选择基于源项目内部Source-CV；最终冠军另行执行10种子完整CPDP。",
        ha="center",
        fontsize=9,
        color="#59636E",
    )
    png_path = output_path / "ipgdo_manifold_formula_sensitivity.png"
    pdf_path = output_path / "ipgdo_manifold_formula_sensitivity.pdf"
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


__all__ = ["create_manifold_formula_figure"]
