from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(
    r"D:\StudyOfPostgraduate\Papers\CPDP\图片\参数实验\IPGDO九组参数敏感性分析"
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
class ParameterExperiment:
    experiment_id: str
    key: str
    title: str
    source: Path
    protocol: str
    round_name: str
    selected_value: float


EXPERIMENTS = (
    ParameterExperiment(
        "target_minority_ratio_round1",
        "target_minority_ratio",
        "目标缺陷比例（第一轮）",
        PROJECT_ROOT
        / "outputs/ipgdo_parameter_search/stage_01_target_minority_ratio_ranking.csv",
        "CPDP开发阶段",
        "第一轮",
        0.25,
    ),
    ParameterExperiment(
        "candidate_multiplier",
        "candidate_multiplier",
        "候选冗余倍数",
        PROJECT_ROOT
        / "outputs/ipgdo_parameter_search/stage_02_candidate_multiplier_ranking.csv",
        "CPDP开发阶段",
        "第一轮",
        3.5,
    ),
    ParameterExperiment(
        "physics_warmup_fraction",
        "physics_warmup_fraction",
        "物理损失预热比例",
        PROJECT_ROOT
        / "outputs/ipgdo_parameter_search/stage_03_physics_warmup_fraction_ranking.csv",
        "CPDP开发阶段",
        "第一轮",
        0.2,
    ),
    ParameterExperiment(
        "formula_semantic_weight",
        "formula_semantic_weight",
        "公式语义损失权重",
        PROJECT_ROOT
        / "outputs/ipgdo_semantic_anchor_guidance_search/"
        "stage_01_formula_semantic_weight_ranking.csv",
        "Source-CV",
        "第二轮",
        0.05,
    ),
    ParameterExperiment(
        "defect_anchor_start_fraction",
        "defect_anchor_start_fraction",
        "缺陷锚点起始比例",
        PROJECT_ROOT
        / "outputs/ipgdo_semantic_anchor_guidance_search/"
        "stage_02_defect_anchor_start_fraction_ranking.csv",
        "Source-CV",
        "第二轮",
        0.5,
    ),
    ParameterExperiment(
        "guidance_scale",
        "guidance_scale",
        "分类器自由引导强度",
        PROJECT_ROOT
        / "outputs/ipgdo_semantic_anchor_guidance_search/"
        "stage_03_guidance_scale_ranking.csv",
        "Source-CV",
        "第二轮",
        1.5,
    ),
    ParameterExperiment(
        "target_minority_ratio_round3",
        "target_minority_ratio",
        "目标缺陷比例（最新复检）",
        PROJECT_ROOT
        / "outputs/ipgdo_ratio_timestep_condition_search/"
        "stage_01_target_minority_ratio_ranking.csv",
        "Source-CV",
        "第三轮",
        0.6,
    ),
    ParameterExperiment(
        "physics_timestep_fraction",
        "physics_timestep_fraction",
        "物理损失时间步比例",
        PROJECT_ROOT
        / "outputs/ipgdo_ratio_timestep_condition_search/"
        "stage_02_physics_timestep_fraction_ranking.csv",
        "Source-CV",
        "第三轮",
        0.4,
    ),
    ParameterExperiment(
        "condition_dropout",
        "condition_dropout",
        "条件丢弃率",
        PROJECT_ROOT
        / "outputs/ipgdo_ratio_timestep_condition_search/"
        "stage_03_condition_dropout_ranking.csv",
        "Source-CV",
        "第三轮",
        0.1,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制IPGDO九组参数实验敏感性图。")
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
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_results() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    required = {"mean_rank", *(f"macro_{metric}" for metric in METRICS)}
    for experiment in EXPERIMENTS:
        if not experiment.source.exists():
            raise FileNotFoundError(f"缺少参数排名文件：{experiment.source}")
        frame = pd.read_csv(experiment.source).sort_values(experiment.key).reset_index(
            drop=True
        )
        missing = {experiment.key, *required} - set(frame.columns)
        if missing:
            raise ValueError(f"{experiment.source.name} 缺少列：{sorted(missing)}")
        if len(frame) != 5:
            raise ValueError(
                f"{experiment.source.name} 应包含5个候选值，实际为{len(frame)}"
            )
        values = frame[experiment.key].to_numpy(float)
        if not np.isclose(values, experiment.selected_value).any():
            raise ValueError(
                f"{experiment.experiment_id} 的最终选择值不在阶段结果中"
            )
        frames[experiment.experiment_id] = frame
    return frames


def create_nine_panel_figure():
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(18, 13.5),
        constrained_layout=True,
    )
    return figure, axes.ravel()


def panel_title(index: int, experiment: ParameterExperiment) -> str:
    letter = chr(ord("a") + index)
    return f"({letter}) {experiment.title}"


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
    axis.axvline(
        selected_value,
        color="#7A1F1F",
        linewidth=1.0,
        linestyle="--",
        alpha=0.65,
    )


def save_figure(figure, output_dir: Path, stem: str, dpi: int) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def add_figure_header(figure, title: str, handles: Sequence[Line2D]) -> None:
    figure.suptitle(
        title,
        x=0.01,
        y=1.025,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.03),
        ncol=len(handles),
        frameon=False,
    )


def add_protocol_note(figure, metric_note: str) -> None:
    figure.text(
        0.5,
        -0.012,
        (
            f"{metric_note}；各面板仅在同一参数的5个候选值内部比较。"
            "目标缺陷比例在第一轮CPDP开发实验和第三轮Source-CV实验中分别复检，"
            "两组结果不可按绝对值横向比较。"
        ),
        ha="center",
        color="#4B5563",
        fontsize=9,
    )


def plot_mean_rank(
    frames: dict[str, pd.DataFrame], output_dir: Path, dpi: int
) -> None:
    figure, axes = create_nine_panel_figure()
    for index, (axis, experiment) in enumerate(zip(axes, EXPERIMENTS)):
        frame = frames[experiment.experiment_id]
        x = frame[experiment.key].to_numpy(float)
        y = frame["mean_rank"].to_numpy(float)
        best_index = int(np.argmin(y))
        selected_index = int(
            np.flatnonzero(np.isclose(x, experiment.selected_value))[0]
        )
        axis.plot(
            x,
            y,
            color=COLORS["rank"],
            linewidth=2.1,
            marker="o",
            markersize=5.5,
        )
        axis.scatter(
            x[best_index],
            y[best_index],
            s=90,
            facecolors="white",
            edgecolors="#111827",
            linewidths=1.7,
            zorder=4,
        )
        axis.scatter(
            x[selected_index],
            y[selected_index],
            s=145,
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
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color="#374151",
            )
        add_selected_line(axis, experiment.selected_value)
        axis.set_title(panel_title(index, experiment), loc="left", fontweight="bold")
        axis.set_xlabel(experiment.key)
        axis.set_xticks(x, [format_value(value) for value in x])
        axis.set_ylabel("综合平均排名（越低越好）" if index % 3 == 0 else "")
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
            markeredgewidth=1.7,
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
    add_figure_header(figure, "IPGDO九组参数实验：综合平均排名", handles)
    add_protocol_note(figure, "纵轴越低越好")
    save_figure(figure, output_dir, "01_mean_rank_sensitivity", dpi)


def metric_delta(
    frame: pd.DataFrame, experiment: ParameterExperiment, metric: str
) -> np.ndarray:
    values = frame[experiment.key].to_numpy(float)
    selected = frame.loc[
        np.isclose(values, experiment.selected_value), f"macro_{metric}"
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
    figure, axes = create_nine_panel_figure()
    for index, (axis, experiment) in enumerate(zip(axes, EXPERIMENTS)):
        frame = frames[experiment.experiment_id]
        x = frame[experiment.key].to_numpy(float)
        for metric in metrics:
            axis.plot(
                x,
                metric_delta(frame, experiment, metric),
                color=COLORS[metric],
                linewidth=1.9,
                marker="o",
                markersize=4.8,
                label=METRIC_LABELS[metric],
            )
        axis.axhline(0.0, color="#6B7280", linewidth=0.9)
        add_selected_line(axis, experiment.selected_value)
        axis.scatter(
            [experiment.selected_value],
            [0.0],
            marker="*",
            s=115,
            color="#C83E4D",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        axis.set_title(panel_title(index, experiment), loc="left", fontweight="bold")
        axis.set_xlabel(experiment.key)
        axis.set_xticks(x, [format_value(value) for value in x])
        axis.set_ylabel("相对最终值变化（百分点）" if index % 3 == 0 else "")
        style_axis(axis)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[metric],
            marker="o",
            linewidth=1.9,
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
    add_figure_header(figure, title, handles)
    add_protocol_note(
        figure,
        "纵轴为相对各实验最终参数值的宏平均变化，正值表示提高，负值表示下降",
    )
    save_figure(figure, output_dir, stem, dpi)


def plot_rank_heatmaps(
    frames: dict[str, pd.DataFrame], output_dir: Path, dpi: int
) -> None:
    figure, axes = create_nine_panel_figure()
    image = None
    for index, (axis, experiment) in enumerate(zip(axes, EXPERIMENTS)):
        frame = frames[experiment.experiment_id]
        metric_values = frame[[f"macro_{metric}" for metric in METRICS]]
        ranks = metric_values.rank(
            axis=0, method="average", ascending=False
        ).to_numpy(float)
        image = axis.imshow(
            ranks,
            cmap="RdYlGn_r",
            norm=Normalize(1, 5),
            aspect="auto",
        )
        x = frame[experiment.key].to_numpy(float)
        labels = [
            f"{format_value(value)}{' ★' if np.isclose(value, experiment.selected_value) else ''}"
            for value in x
        ]
        axis.set_xticks(
            range(len(METRICS)), [METRIC_LABELS[metric] for metric in METRICS]
        )
        axis.set_yticks(range(len(x)), labels)
        axis.set_title(panel_title(index, experiment), loc="left", fontweight="bold")
        axis.set_ylabel(experiment.key if index % 3 == 0 else "")
        axis.tick_params(axis="x", rotation=25)
        for row in range(ranks.shape[0]):
            for column in range(ranks.shape[1]):
                rank = ranks[row, column]
                label = f"{rank:.0f}" if float(rank).is_integer() else f"{rank:.1f}"
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="#111827",
                )
        for spine in axis.spines.values():
            spine.set_color("#8A949E")

    if image is None:
        raise RuntimeError("未生成热图")
    colorbar = figure.colorbar(image, ax=axes, location="right", shrink=0.82, pad=0.015)
    colorbar.set_label("单指标排名（1=最好，5=最差）")
    colorbar.set_ticks([1, 2, 3, 4, 5])
    figure.suptitle(
        "IPGDO九组参数实验：六指标排名热图",
        x=0.01,
        y=1.025,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    add_protocol_note(figure, "星号表示最终交互选择值")
    save_figure(figure, output_dir, "05_metric_rank_heatmap", dpi)


def export_source_data(
    frames: dict[str, pd.DataFrame], output_dir: Path
) -> pd.DataFrame:
    exported = []
    for experiment in EXPERIMENTS:
        frame = frames[experiment.experiment_id].copy()
        values = frame[experiment.key].to_numpy(float)
        frame.insert(0, "experiment_id", experiment.experiment_id)
        frame.insert(1, "round", experiment.round_name)
        frame.insert(2, "protocol_group", experiment.protocol)
        frame.insert(3, "parameter", experiment.key)
        frame.insert(4, "parameter_value", values)
        frame.insert(5, "final_selected", np.isclose(values, experiment.selected_value))
        frame.insert(6, "one_factor_best", frame["mean_rank"] == frame["mean_rank"].min())
        exported.append(
            frame[
                [
                    "experiment_id",
                    "round",
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
    result.to_csv(output_dir / "ipgdo_nine_experiment_sensitivity_data.csv", index=False)
    return result


def write_readme(output_dir: Path) -> None:
    selected_lines = "\n".join(
        f"- `{experiment.experiment_id}`: `{experiment.key} = "
        f"{format_value(experiment.selected_value)}`"
        for experiment in EXPERIMENTS
    )
    content = f"""# IPGDO九组参数实验敏感性图

本目录包含5组图，每组图均采用3×3布局展示9组单参数实验：

1. `01_mean_rank_sensitivity`：综合平均排名，越低越好。
2. `02_auc_auprc_sensitivity`：AUC和AUPRC相对最终参数值的变化。
3. `03_classification_metrics_sensitivity`：MCC、F1和G-mean变化。
4. `04_popt20_sensitivity`：Popt@20%变化。
5. `05_metric_rank_heatmap`：每组实验内部六项指标的候选值排名。

最终选择值：

{selected_lines}

9组实验对应8个不同参数。`target_minority_ratio`在第一轮CPDP开发阶段和第三轮Source-CV阶段分别测试，固定参数背景和评价协议不同，因此作为两个独立实验面板展示，不将其解释为两个不同超参数，也不跨面板比较绝对数值。

PNG适合组会汇报，PDF为矢量版本；原始绘图数据保存在`ipgdo_nine_experiment_sensitivity_data.csv`。
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def write_manifest(output_dir: Path, dpi: int) -> None:
    manifest = {
        "dpi": dpi,
        "experiment_count": len(EXPERIMENTS),
        "unique_parameter_count": len({experiment.key for experiment in EXPERIMENTS}),
        "experiments": [
            {
                "experiment_id": experiment.experiment_id,
                "round": experiment.round_name,
                "protocol": experiment.protocol,
                "parameter": experiment.key,
                "selected_value": experiment.selected_value,
                "source": str(experiment.source),
            }
            for experiment in EXPERIMENTS
        ],
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
    export_source_data(frames, args.output_dir)
    plot_mean_rank(frames, args.output_dir, args.dpi)
    plot_metric_group(
        frames,
        ("auc", "auprc"),
        "IPGDO九组参数实验：判别指标",
        "02_auc_auprc_sensitivity",
        args.output_dir,
        args.dpi,
    )
    plot_metric_group(
        frames,
        ("mcc", "f1", "gmean"),
        "IPGDO九组参数实验：分类指标",
        "03_classification_metrics_sensitivity",
        args.output_dir,
        args.dpi,
    )
    plot_metric_group(
        frames,
        ("popt20",),
        "IPGDO九组参数实验：工作量感知指标",
        "04_popt20_sensitivity",
        args.output_dir,
        args.dpi,
    )
    plot_rank_heatmaps(frames, args.output_dir, args.dpi)
    write_readme(args.output_dir)
    write_manifest(args.output_dir, args.dpi)
    print(f"IPGDO_NINE_EXPERIMENT_FIGURES_READY output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
