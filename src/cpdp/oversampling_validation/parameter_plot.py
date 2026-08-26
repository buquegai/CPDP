from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParameterPlotSpec:
    key: str
    title: str
    stage_file: str


PARAMETER_SPECS = (
    ParameterPlotSpec(
        "target_minority_ratio",
        "目标缺陷比例",
        "stage_01_target_minority_ratio_ranking.csv",
    ),
    ParameterPlotSpec(
        "physics_timestep_fraction",
        "物理损失时间步比例",
        "stage_02_physics_timestep_fraction_ranking.csv",
    ),
    ParameterPlotSpec(
        "condition_dropout",
        "条件丢弃率",
        "stage_03_condition_dropout_ranking.csv",
    ),
)


def _configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    )
    for candidate in font_candidates:
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            family = font_manager.FontProperties(fname=str(candidate)).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _load_stage(output_dir: Path, spec: ParameterPlotSpec) -> pd.DataFrame:
    path = output_dir / spec.stage_file
    if not path.is_file():
        raise FileNotFoundError(f"Parameter ranking file not found: {path}")
    frame = pd.read_csv(path)
    required = {spec.key, "mean_rank", "selection_position"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    frame[spec.key] = pd.to_numeric(frame[spec.key], errors="raise")
    frame["mean_rank"] = pd.to_numeric(frame["mean_rank"], errors="raise")
    frame["selection_position"] = pd.to_numeric(
        frame["selection_position"], errors="raise"
    )
    if len(frame) < 2 or not np.isfinite(frame["mean_rank"]).all():
        raise ValueError(f"{path.name} does not contain valid candidate rankings")
    return frame.sort_values(spec.key).reset_index(drop=True)


def create_parameter_comparison_figure(
    output_dir: Path,
    winner: Mapping[str, float],
    *,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(1, 3, figsize=(15.6, 5.6), constrained_layout=True)
    line_color = "#2F6254"
    final_color = "#C83E4D"

    for index, (axis, spec) in enumerate(zip(axes, PARAMETER_SPECS)):
        frame = _load_stage(output_path, spec)
        if spec.key not in winner:
            raise KeyError(f"Winner is missing parameter: {spec.key}")
        final_value = float(winner[spec.key])
        values = frame[spec.key].to_numpy(dtype=float)
        ranks = frame["mean_rank"].to_numpy(dtype=float)
        if not np.isclose(values, final_value).any():
            raise ValueError(
                f"Final value {final_value:g} is absent from {spec.stage_file}"
            )
        stage_best_row = frame.loc[frame["selection_position"].idxmin()]
        stage_best_value = float(stage_best_row[spec.key])
        stage_best_rank = float(stage_best_row["mean_rank"])
        final_rank = float(
            frame.loc[np.isclose(frame[spec.key], final_value), "mean_rank"].iloc[0]
        )

        axis.plot(
            values,
            ranks,
            color=line_color,
            linewidth=2.2,
            marker="o",
            markersize=6,
            markerfacecolor=line_color,
            markeredgecolor=line_color,
        )
        axis.scatter(
            [stage_best_value],
            [stage_best_rank],
            s=120,
            facecolors="white",
            edgecolors="#182433",
            linewidths=1.7,
            zorder=4,
        )
        axis.axvline(
            final_value,
            color=final_color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.65,
        )
        axis.scatter(
            [final_value],
            [final_rank],
            marker="*",
            s=190,
            color=final_color,
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        for value, rank in zip(values, ranks):
            axis.annotate(
                f"{rank:.2f}",
                (value, rank),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="#4B5563",
                fontsize=9,
            )
        axis.set_title(f"({chr(ord('a') + index)}) {spec.title}", loc="left")
        axis.set_xlabel(spec.key)
        axis.set_xticks(values)
        axis.set_xticklabels([f"{value:g}" for value in values])
        axis.invert_yaxis()
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.85)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#8A949E")
        axis.spines["bottom"].set_color("#8A949E")
        axis.tick_params(colors="#374151")

    axes[0].set_ylabel("综合平均排名（越低越好）")
    from matplotlib.lines import Line2D

    legend_items = (
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
            markerfacecolor=final_color,
            markeredgecolor="white",
            markersize=11,
            label="最终交互选择",
        ),
    )
    figure.suptitle(
        "IPGDO三参数敏感性：综合平均排名",
        x=0.02,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.legend(
        handles=legend_items,
        loc="upper right",
        bbox_to_anchor=(0.98, 1.02),
        frameon=False,
        ncol=2,
    )
    figure.text(
        0.5,
        -0.015,
        "参数选择基于源项目内部Source-CV；仅在各参数内部比较。",
        ha="center",
        fontsize=9,
        color="#59636E",
    )
    png_path = output_path / "ipgdo_three_parameter_sensitivity.png"
    pdf_path = output_path / "ipgdo_three_parameter_sensitivity.pdf"
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


__all__ = ["create_parameter_comparison_figure"]
