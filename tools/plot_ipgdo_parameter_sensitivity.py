from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(
    r"D:\StudyOfPostgraduate\Papers\CPDP\图片\参数实验\IPGDO五参数敏感性分析"
)
METRICS = ("auc", "auprc", "mcc", "f1", "gmean", "popt20")
METRIC_LABELS = {
    "auc": "AUC",
    "auprc": "AUPRC",
    "mcc": "MCC",
    "f1": "F1",
    "gmean": "G-mean",
    "popt20": "Popt@20%",
}
COLORS = {
    "auc": "#0072B2",
    "auprc": "#D55E00",
    "mcc": "#009E73",
    "f1": "#CC79A7",
    "gmean": "#E69F00",
    "popt20": "#56B4E9",
    "rank": "#2F5D50",
}


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    title: str
    source: Path
    protocol: str
    selected_value: float


PARAMETERS = (
    ParameterSpec(
        "candidate_multiplier",
        "候选冗余倍数",
        PROJECT_ROOT
        / "outputs/ipgdo_parameter_search/stage_02_candidate_multiplier_ranking.csv",
        "CPDP开发阶段",
        3.5,
    ),
    ParameterSpec(
        "physics_warmup_fraction",
        "物理损失预热比例",
        PROJECT_ROOT
        / "outputs/ipgdo_parameter_search/stage_03_physics_warmup_fraction_ranking.csv",
        "CPDP开发阶段",
        0.2,
    ),
    ParameterSpec(
        "formula_semantic_weight",
        "公式语义损失权重",
        PROJECT_ROOT
        / "outputs/ipgdo_semantic_anchor_guidance_search/"
        "stage_01_formula_semantic_weight_ranking.csv",
        "Source-CV",
        0.05,
    ),
    ParameterSpec(
        "defect_anchor_start_fraction",
        "缺陷锚点起始比例",
        PROJECT_ROOT
        / "outputs/ipgdo_semantic_anchor_guidance_search/"
        "stage_02_defect_anchor_start_fraction_ranking.csv",
        "Source-CV",
        0.5,
    ),
    ParameterSpec(
        "guidance_scale",
        "分类器自由引导强度",
        PROJECT_ROOT
        / "outputs/ipgdo_semantic_anchor_guidance_search/"
        "stage_03_guidance_scale_ranking.csv",
        "Source-CV",
        1.5,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制IPGDO五参数敏感性分析图。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_fonts() -> None:
    candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for candidate in candidates:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            family = font_manager.FontProperties(fname=str(candidate)).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_results() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    required = {"mean_rank", *(f"macro_{metric}" for metric in METRICS)}
    for spec in PARAMETERS:
        if not spec.source.exists():
            raise FileNotFoundError(f"缺少参数排名文件：{spec.source}")
        frame = pd.read_csv(spec.source).sort_values(spec.key).reset_index(drop=True)
        missing = {spec.key, *required} - set(frame.columns)
        if missing:
            raise ValueError(f"{spec.source.name} 缺少列：{sorted(missing)}")
        if len(frame) != 5:
            raise ValueError(f"{spec.source.name} 应包含5个候选值，实际为{len(frame)}")
        if not np.isclose(frame[spec.key].to_numpy(float), spec.selected_value).any():
            raise ValueError(f"{spec.key} 的最终选择值不在阶段结果中")
        frames[spec.key] = frame
    return frames


def panel_title(index: int, spec: ParameterSpec) -> str:
    letter = chr(ord("a") + index)
    return f"({letter}) {spec.title}"


def format_value(value: float) -> str:
    return f"{value:g}"


def style_axis(axis) -> None:
    axis.grid(axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A949E")
    axis.spines["bottom"].set_color("#8A949E")
    axis.tick_params(colors="#374151")


def add_selected_line(axis, selected_value: float) -> None:
    axis.axvline(selected_value, color="#7A1F1F", linewidth=1.0, linestyle="--", alpha=0.65)


def save_figure(figure, output_dir: Path, stem: str, dpi: int) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def create_five_panel_figure():
    figure = plt.figure(figsize=(15.6, 8.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 6)
    axes = (
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
        figure.add_subplot(grid[1, 1:3]),
        figure.add_subplot(grid[1, 3:5]),
    )
    return figure, axes


def plot_mean_rank(
    frames: dict[str, pd.DataFrame], output_dir: Path, dpi: int
) -> None:
    figure, axes = create_five_panel_figure()
    for index, (axis, spec) in enumerate(zip(axes, PARAMETERS)):
        frame = frames[spec.key]
        x = frame[spec.key].to_numpy(float)
        y = frame["mean_rank"].to_numpy(float)
        best_index = int(np.argmin(y))
        selected_index = int(np.flatnonzero(np.isclose(x, spec.selected_value))[0])

        axis.plot(x, y, color=COLORS["rank"], linewidth=2.2, marker="o", markersize=6)
        axis.scatter(
            x[best_index],
            y[best_index],
            s=95,
            facecolors="white",
            edgecolors="#111827",
            linewidths=1.8,
            zorder=4,
        )
        axis.scatter(
            x[selected_index],
            y[selected_index],
            s=150,
            marker="*",
            color="#C83E4D",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
        for value_x, value_y in zip(x, y):
            axis.annotate(
                f"{value_y:.2f}",
                (value_x, value_y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#374151",
            )
        add_selected_line(axis, spec.selected_value)
        axis.set_title(panel_title(index, spec), loc="left", fontweight="bold")
        axis.set_xlabel(spec.key)
        axis.set_xticks(x, [format_value(value) for value in x])
        axis.set_ylabel("综合平均排名（越低越好）" if index in (0, 3) else "")
        axis.invert_yaxis()
        style_axis(axis)

    handles = (
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#111827",
            markeredgewidth=1.8,
            markersize=8,
            label="单参数阶段最优",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#C83E4D",
            markeredgecolor="white",
            markersize=12,
            label="最终交互选择",
        ),
    )
    figure.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 1.05),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "IPGDO五参数敏感性：综合平均排名",
        x=0.01,
        y=1.05,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        "注：前两个参数来自CPDP开发阶段，后三个参数来自Source-CV；仅在各参数内部比较。",
        ha="center",
        color="#4B5563",
        fontsize=9,
    )
    save_figure(figure, output_dir, "01_mean_rank_sensitivity", dpi)


def metric_delta(frame: pd.DataFrame, spec: ParameterSpec, metric: str) -> np.ndarray:
    selected = frame.loc[
        np.isclose(frame[spec.key].to_numpy(float), spec.selected_value),
        f"macro_{metric}",
    ].iloc[0]
    return (frame[f"macro_{metric}"].to_numpy(float) - float(selected)) * 100.0


def plot_metric_group(
    frames: dict[str, pd.DataFrame],
    metrics: Sequence[str],
    title: str,
    stem: str,
    output_dir: Path,
    dpi: int,
) -> None:
    figure, axes = create_five_panel_figure()
    for index, (axis, spec) in enumerate(zip(axes, PARAMETERS)):
        frame = frames[spec.key]
        x = frame[spec.key].to_numpy(float)
        for metric in metrics:
            delta = metric_delta(frame, spec, metric)
            axis.plot(
                x,
                delta,
                color=COLORS[metric],
                linewidth=2.0,
                marker="o",
                markersize=5,
                label=METRIC_LABELS[metric],
            )
        axis.axhline(0.0, color="#6B7280", linewidth=0.9)
        add_selected_line(axis, spec.selected_value)
        axis.scatter(
            [spec.selected_value],
            [0.0],
            marker="*",
            s=120,
            color="#C83E4D",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        axis.set_title(panel_title(index, spec), loc="left", fontweight="bold")
        axis.set_xlabel(spec.key)
        axis.set_xticks(x, [format_value(value) for value in x])
        axis.set_ylabel("相对最终参数值的变化（百分点）" if index in (0, 3) else "")
        style_axis(axis)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[metric],
            marker="o",
            linewidth=2.0,
            label=METRIC_LABELS[metric],
        )
        for metric in metrics
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#C83E4D",
            markeredgecolor="white",
            markersize=12,
            label="最终选择",
        )
    )
    figure.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 1.05),
        ncol=len(handles),
        frameon=False,
    )
    figure.suptitle(
        title,
        x=0.01,
        y=1.05,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        "纵轴为相对最终参数值的宏平均变化；正值表示该指标提高，负值表示下降。",
        ha="center",
        color="#4B5563",
        fontsize=9,
    )
    save_figure(figure, output_dir, stem, dpi)


def plot_rank_heatmaps(
    frames: dict[str, pd.DataFrame], output_dir: Path, dpi: int
) -> None:
    figure, axes = create_five_panel_figure()
    image = None
    for index, (axis, spec) in enumerate(zip(axes, PARAMETERS)):
        frame = frames[spec.key]
        values = frame[[f"macro_{metric}" for metric in METRICS]]
        ranks = values.rank(axis=0, method="average", ascending=False).to_numpy(float)
        image = axis.imshow(ranks, cmap="RdYlGn_r", norm=Normalize(1, 5), aspect="auto")
        x = frame[spec.key].to_numpy(float)
        labels = [
            f"{format_value(value)}{' ★' if np.isclose(value, spec.selected_value) else ''}"
            for value in x
        ]
        axis.set_xticks(range(len(METRICS)), [METRIC_LABELS[metric] for metric in METRICS])
        axis.set_yticks(range(len(x)), labels)
        axis.set_title(panel_title(index, spec), loc="left", fontweight="bold")
        axis.set_ylabel(spec.key if index in (0, 3) else "")
        axis.tick_params(axis="x", rotation=25)
        for row in range(ranks.shape[0]):
            for column in range(ranks.shape[1]):
                rank = ranks[row, column]
                axis.text(
                    column,
                    row,
                    f"{rank:.0f}" if float(rank).is_integer() else f"{rank:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#111827",
                )
        for spine in axis.spines.values():
            spine.set_color("#8A949E")

    if image is None:
        raise RuntimeError("未生成热图")
    colorbar = figure.colorbar(image, ax=axes, location="right", shrink=0.82, pad=0.02)
    colorbar.set_label("单指标排名（1=最好，5=最差）")
    colorbar.set_ticks([1, 2, 3, 4, 5])
    figure.suptitle(
        "IPGDO五参数的多指标权衡热图",
        x=0.01,
        y=1.035,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        "每个子图只在同一参数的5个候选值之间排名；★表示最终交互选择值。",
        ha="center",
        color="#4B5563",
        fontsize=9,
    )
    save_figure(figure, output_dir, "05_metric_rank_heatmap", dpi)


def export_source_data(
    frames: dict[str, pd.DataFrame], output_dir: Path
) -> pd.DataFrame:
    exported = []
    for spec in PARAMETERS:
        frame = frames[spec.key].copy()
        frame.insert(0, "protocol_group", spec.protocol)
        frame.insert(1, "parameter", spec.key)
        frame.insert(2, "parameter_value", frame[spec.key])
        frame.insert(3, "final_selected", np.isclose(frame[spec.key], spec.selected_value))
        frame.insert(4, "one_factor_best", frame["mean_rank"] == frame["mean_rank"].min())
        exported.append(
            frame[
                [
                    "protocol_group",
                    "parameter",
                    "parameter_value",
                    "final_selected",
                    "one_factor_best",
                    "mean_rank",
                    *(f"macro_{metric}" for metric in METRICS),
                ]
            ]
        )
    result = pd.concat(exported, ignore_index=True)
    result.to_csv(output_dir / "ipgdo_parameter_sensitivity_data.csv", index=False)
    return result


def write_readme(output_dir: Path, source_data: pd.DataFrame) -> None:
    selected = source_data[source_data["final_selected"]][
        ["parameter", "parameter_value"]
    ]
    selected_lines = "\n".join(
        f"- `{row.parameter} = {format_value(float(row.parameter_value))}`"
        for row in selected.itertuples()
    )
    content = f"""# IPGDO五参数敏感性分析图

本目录包含5张组会汇报图，每张图同时覆盖5个参数：

1. `01_mean_rank_sensitivity`：综合平均排名，越低越好。
2. `02_auc_auprc_sensitivity`：AUC和AUPRC相对最终参数值的变化。
3. `03_classification_metrics_sensitivity`：MCC、F1和G-mean变化。
4. `04_popt20_sensitivity`：Popt@20%变化。
5. `05_metric_rank_heatmap`：每个参数内部的六指标排名热图。

最终参数：

{selected_lines}

`target_minority_ratio`暂不进入本轮图形，保留原始实验数据供后续讨论。
前两个参数来自第一轮CPDP开发阶段参数实验，后三个参数来自Source-CV参数实验。
两轮协议不同，因此图中只比较同一参数内部的候选值，不跨协议比较绝对数值。
`formula_semantic_weight`的单参数阶段最优值为0.1，但最终2x2x2交互检查选择0.05。

PNG文件适合直接插入组会PPT，PDF文件为矢量版本，适合论文排版。
原始绘图数据保存在`ipgdo_parameter_sensitivity_data.csv`。
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def write_manifest(output_dir: Path, dpi: int) -> None:
    manifest = {
        "dpi": dpi,
        "parameters": [spec.key for spec in PARAMETERS],
        "selected_values": {spec.key: spec.selected_value for spec in PARAMETERS},
        "sources": {spec.key: str(spec.source) for spec in PARAMETERS},
        "figures": [
            "01_mean_rank_sensitivity",
            "02_auc_auprc_sensitivity",
            "03_classification_metrics_sensitivity",
            "04_popt20_sensitivity",
            "05_metric_rank_heatmap",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.dpi < 150:
        raise ValueError("dpi不得低于150")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_fonts()
    frames = load_results()
    source_data = export_source_data(frames, args.output_dir)

    plot_mean_rank(frames, args.output_dir, args.dpi)
    plot_metric_group(
        frames,
        ("auc", "auprc"),
        "IPGDO五参数敏感性：判别指标",
        "02_auc_auprc_sensitivity",
        args.output_dir,
        args.dpi,
    )
    plot_metric_group(
        frames,
        ("mcc", "f1", "gmean"),
        "IPGDO五参数敏感性：分类指标",
        "03_classification_metrics_sensitivity",
        args.output_dir,
        args.dpi,
    )
    plot_metric_group(
        frames,
        ("popt20",),
        "IPGDO五参数敏感性：工作量感知指标",
        "04_popt20_sensitivity",
        args.output_dir,
        args.dpi,
    )
    plot_rank_heatmaps(frames, args.output_dir, args.dpi)
    write_readme(args.output_dir, source_data)
    write_manifest(args.output_dir, args.dpi)
    print(f"IPGDO_PARAMETER_FIGURES_READY output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
