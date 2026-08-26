from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
FIGURE_DIR = Path(
    r"D:\StudyOfPostgraduate\Papers\CPDP\图片\参数实验\IPGDO本周参数实验总览"
)
PNG_PATH = FIGURE_DIR / "IPGDO本周18项参数实验总览.png"
PDF_PATH = FIGURE_DIR / "IPGDO本周18项参数实验总览.pdf"


@dataclass(frozen=True)
class PanelSpec:
    group: int
    csv_path: Path
    columns: tuple[str, ...]
    chinese_name: str
    english_name: str
    role: str
    selected: object
    value_formatter: Callable[[object], str]


def _number(value: object) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    if abs(number) < 0.01:
        return f"{number:.5f}".rstrip("0").rstrip(".")
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _interval(value: object) -> str:
    lower, upper = value
    return f"{_number(lower)}-{_number(upper)}"


def _csv(directory: str, filename: str) -> Path:
    return OUTPUTS_ROOT / directory / filename


PANELS: tuple[PanelSpec, ...] = (
    PanelSpec(
        1,
        _csv("ipgdo_parameter_search", "stage_01_target_minority_ratio_ranking.csv"),
        ("target_minority_ratio",),
        "目标缺陷比例（初筛）",
        "Target minority ratio (initial)",
        "规定增强后缺陷样本数与正常样本数之比，直接控制合成样本规模。",
        0.25,
        _number,
    ),
    PanelSpec(
        1,
        _csv("ipgdo_parameter_search", "stage_02_candidate_multiplier_ranking.csv"),
        ("candidate_multiplier",),
        "候选冗余倍数",
        "Candidate multiplier",
        "先生成超额候选供有效性、流形和多样性筛选，平衡质量与计算成本。",
        3.5,
        _number,
    ),
    PanelSpec(
        1,
        _csv("ipgdo_parameter_search", "stage_03_physics_warmup_fraction_ranking.csv"),
        ("physics_warmup_fraction",),
        "物理损失预热比例",
        "Physics warm-up fraction",
        "训练前期仅学习扩散目标，达到该比例后再引入物理语义损失。",
        0.2,
        _number,
    ),
    PanelSpec(
        2,
        _csv(
            "ipgdo_semantic_anchor_guidance_search",
            "stage_01_formula_semantic_weight_ranking.csv",
        ),
        ("formula_semantic_weight",),
        "公式语义损失权重",
        "Formula-semantic loss weight",
        "控制隐式物理公式损失对扩散训练总目标的贡献强度。",
        0.05,
        _number,
    ),
    PanelSpec(
        2,
        _csv(
            "ipgdo_semantic_anchor_guidance_search",
            "stage_02_defect_anchor_start_fraction_ranking.csv",
        ),
        ("defect_anchor_start_fraction",),
        "缺陷锚点起始比例",
        "Defect-anchor start fraction",
        "决定反向扩散何时启用缺陷锚点约束，协调生成自由度与缺陷贴合度。",
        0.5,
        _number,
    ),
    PanelSpec(
        2,
        _csv(
            "ipgdo_semantic_anchor_guidance_search",
            "stage_03_guidance_scale_ranking.csv",
        ),
        ("guidance_scale",),
        "无分类器引导强度",
        "Classifier-free guidance scale",
        "放大条件与无条件噪声预测之差，增强缺陷及项目条件的引导作用。",
        1.5,
        _number,
    ),
    PanelSpec(
        3,
        _csv(
            "ipgdo_ratio_timestep_condition_search",
            "stage_01_target_minority_ratio_ranking.csv",
        ),
        ("target_minority_ratio",),
        "目标缺陷比例（复核）",
        "Target minority ratio (recheck)",
        "在统一 Source-CV 协议下复核增强规模，并确定正式目标比例。",
        0.6,
        _number,
    ),
    PanelSpec(
        3,
        _csv(
            "ipgdo_ratio_timestep_condition_search",
            "stage_02_physics_timestep_fraction_ranking.csv",
        ),
        ("physics_timestep_fraction",),
        "物理损失时间步比例",
        "Physics timestep fraction",
        "限定物理损失参与训练的扩散时间步范围，避免在高噪声阶段错误约束。",
        0.4,
        _number,
    ),
    PanelSpec(
        3,
        _csv(
            "ipgdo_ratio_timestep_condition_search",
            "stage_03_condition_dropout_ranking.csv",
        ),
        ("condition_dropout",),
        "条件丢弃率",
        "Condition dropout",
        "训练时随机移除条件以学习无条件分支，支撑无分类器引导并增强鲁棒性。",
        0.1,
        _number,
    ),
    PanelSpec(
        4,
        _csv(
            "ipgdo_manifold_formula_search", "stage_01_manifold_interval_ranking.csv"
        ),
        ("manifold_lower_quantile", "manifold_upper_quantile"),
        "流形内部距离分位区间",
        "Manifold-distance quantile interval",
        "定义真实缺陷样本内部 KNN 距离的正常区间，剔除过近或脱离流形的候选。",
        (0.025, 0.975),
        _interval,
    ),
    PanelSpec(
        4,
        _csv(
            "ipgdo_manifold_formula_search", "stage_02_manifold_neighbors_ranking.csv"
        ),
        ("manifold_neighbors",),
        "流形邻居数",
        "Manifold neighbors",
        "规定 KNN 流形距离使用的邻居数量，控制局部结构估计的平滑程度。",
        5,
        _number,
    ),
    PanelSpec(
        4,
        _csv(
            "ipgdo_manifold_formula_search", "stage_03_formula_max_terms_ranking.csv"
        ),
        ("formula_max_terms",),
        "物理公式最大项数",
        "Maximum formula terms",
        "限制自适应公式包含的特征项数量，平衡表达能力、稳定性与可解释性。",
        10,
        _number,
    ),
    PanelSpec(
        5,
        _csv(
            "ipgdo_formula_optimization_search", "stage_01_formula_l1_c_ranking.csv"
        ),
        ("formula_l1_c",),
        "公式 L1 稀疏系数",
        "Formula L1 sparsity coefficient",
        "控制公式挖掘中的 L1 正则化强度，从而调节公式稀疏性与复杂度。",
        0.1,
        _number,
    ),
    PanelSpec(
        5,
        _csv(
            "ipgdo_formula_optimization_search",
            "stage_02_diffusion_learning_rate_ranking.csv",
        ),
        ("diffusion_learning_rate",),
        "扩散模型学习率",
        "Diffusion learning rate",
        "控制扩散网络每次梯度更新的步长，影响收敛速度与训练稳定性。",
        0.001,
        _number,
    ),
    PanelSpec(
        5,
        _csv(
            "ipgdo_formula_optimization_search", "stage_03_diffusion_epochs_ranking.csv"
        ),
        ("diffusion_epochs",),
        "扩散模型训练轮数",
        "Diffusion training epochs",
        "决定扩散模型的训练预算，影响拟合充分性、过拟合风险与运行时间。",
        300,
        _number,
    ),
    PanelSpec(
        6,
        _csv(
            "ipgdo_network_capacity_search", "stage_01_diffusion_hidden_dim_ranking.csv"
        ),
        ("diffusion_hidden_dim",),
        "扩散网络隐藏维度",
        "Diffusion hidden dimension",
        "决定残差网络中间表征宽度，控制模型容量、显存占用与计算量。",
        256,
        _number,
    ),
    PanelSpec(
        6,
        _csv(
            "ipgdo_network_capacity_search",
            "stage_02_diffusion_residual_blocks_ranking.csv",
        ),
        ("diffusion_residual_blocks",),
        "扩散网络残差块数",
        "Diffusion residual blocks",
        "决定去噪网络深度，影响非线性建模能力、梯度传播与训练成本。",
        4,
        _number,
    ),
    PanelSpec(
        6,
        _csv(
            "ipgdo_network_capacity_search", "stage_03_diffusion_timesteps_ranking.csv"
        ),
        ("diffusion_timesteps",),
        "扩散时间步数",
        "Diffusion timesteps",
        "规定前向加噪与反向去噪离散步数，权衡生成精度与采样时间。",
        200,
        _number,
    ),
)


GROUP_TITLES = {
    1: "第 1 轮｜生成规模与物理预热",
    2: "第 2 轮｜物理语义与条件引导",
    3: "第 3 轮｜比例复核与物理调度",
    4: "第 4 轮｜KNN 流形与公式复杂度",
    5: "第 5 轮｜公式稀疏与训练优化",
    6: "第 6 轮｜扩散网络容量",
}


GROUP_COLORS = {
    1: "#1F6F78",
    2: "#B84A3A",
    3: "#34699A",
    4: "#6A7B3F",
    5: "#9A6A28",
    6: "#67507D",
}


def _font() -> FontProperties:
    candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("未找到可用于中文绘图的字体")
    return FontProperties(fname=str(path))


def _panel_values(frame: pd.DataFrame, spec: PanelSpec) -> list[object]:
    if len(spec.columns) == 1:
        return frame[spec.columns[0]].tolist()
    return [tuple(row) for row in frame[list(spec.columns)].itertuples(index=False, name=None)]


def _selected_index(values: Sequence[object], selected: object) -> int:
    for index, value in enumerate(values):
        if isinstance(selected, tuple):
            if np.allclose(np.asarray(value, dtype=float), np.asarray(selected, dtype=float)):
                return index
        elif np.isclose(float(value), float(selected)):
            return index
    raise ValueError(f"最终选择值 {selected!r} 不在单参数候选中")


def _draw_panel(
    ax: plt.Axes,
    spec: PanelSpec,
    panel_index: int,
    font: FontProperties,
) -> None:
    if not spec.csv_path.is_file():
        raise FileNotFoundError(f"缺少参数排名文件：{spec.csv_path}")
    frame = pd.read_csv(spec.csv_path)
    values = _panel_values(frame, spec)
    mean_ranks = frame["mean_rank"].astype(float).to_numpy()

    order = sorted(
        range(len(values)),
        key=lambda index: tuple(values[index])
        if isinstance(values[index], tuple)
        else float(values[index]),
    )
    values = [values[index] for index in order]
    mean_ranks = mean_ranks[order]
    x = np.arange(len(values), dtype=float)
    selected_index = _selected_index(values, spec.selected)
    stage_best_index = int(np.argmin(mean_ranks))
    accent = GROUP_COLORS[spec.group]

    ax.plot(
        x,
        mean_ranks,
        color="#215B4A",
        linewidth=3.2,
        marker="o",
        markersize=8,
        markerfacecolor="#215B4A",
        zorder=3,
    )
    ax.scatter(
        [x[stage_best_index]],
        [mean_ranks[stage_best_index]],
        s=220,
        facecolors="white",
        edgecolors="#172A3A",
        linewidths=2.5,
        zorder=5,
    )
    ax.axvline(
        x[selected_index],
        color="#D1495B",
        linewidth=1.8,
        linestyle="--",
        alpha=0.75,
        zorder=2,
    )
    ax.scatter(
        [x[selected_index]],
        [mean_ranks[selected_index]],
        s=280,
        marker="*",
        color="#D1495B",
        edgecolors="white",
        linewidths=1.0,
        zorder=6,
    )

    spread = max(0.08, float(np.ptp(mean_ranks)))
    y_min = float(np.min(mean_ranks) - 0.15 * spread - 0.035)
    y_max = float(np.max(mean_ranks) + 0.16 * spread + 0.035)
    ax.set_ylim(y_max, y_min)
    ax.set_xlim(-0.35, len(values) - 0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [spec.value_formatter(value) for value in values],
        fontproperties=font,
        fontsize=15,
    )
    ax.tick_params(axis="y", labelsize=14, colors="#405064")
    ax.tick_params(axis="x", colors="#405064", pad=7)
    ax.grid(axis="y", color="#DCE3E8", linewidth=1.1)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#93A1AD")
    ax.spines["bottom"].set_color("#93A1AD")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.set_facecolor("#FBFCFD")

    for point_x, rank in zip(x, mean_ranks):
        ax.annotate(
            f"{rank:.2f}",
            (point_x, rank),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=13,
            color="#526273",
            fontproperties=font,
        )

    letter = chr(ord("a") + panel_index)
    selected_text = spec.value_formatter(spec.selected)
    ax.text(
        0.0,
        1.22,
        f"({letter}) {spec.chinese_name} ｜最终值：{selected_text}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=19,
        fontweight="bold",
        color="#13212B",
        fontproperties=font,
    )
    ax.text(
        0.0,
        1.155,
        spec.english_name,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        color=accent,
        fontproperties=font,
    )
    role_text = "\n".join(wrap(f"作用：{spec.role}", width=27))
    ax.text(
        0.0,
        1.025,
        role_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13.5,
        color="#526273",
        fontproperties=font,
        linespacing=1.25,
    )
    ax.set_ylabel("综合平均排名 ↓", fontsize=15, color="#405064", fontproperties=font)


def create_figure() -> tuple[Path, Path]:
    font = _font()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(3, 6, figsize=(48, 27), dpi=160)
    figure.patch.set_facecolor("white")
    figure.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.845,
        bottom=0.075,
        wspace=0.28,
        hspace=0.72,
    )

    for index, (ax, spec) in enumerate(zip(axes.flat, PANELS)):
        _draw_panel(ax, spec, index, font)

    figure.canvas.draw()
    for row, (left_group, right_group) in enumerate(((1, 2), (3, 4), (5, 6))):
        for start_column, group in ((0, left_group), (3, right_group)):
            left_box = axes[row, start_column].get_position()
            right_box = axes[row, start_column + 2].get_position()
            x0 = left_box.x0
            x1 = right_box.x1
            y = left_box.y1 + 0.072
            color = GROUP_COLORS[group]
            figure.add_artist(
                Rectangle(
                    (x0, y - 0.008),
                    x1 - x0,
                    0.003,
                    transform=figure.transFigure,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.9,
                )
            )
            figure.text(
                (x0 + x1) / 2,
                y,
                GROUP_TITLES[group],
                ha="center",
                va="bottom",
                fontsize=24,
                fontweight="bold",
                color=color,
                fontproperties=font,
            )

    figure.text(
        0.035,
        0.975,
        "IPGDO 本周超参数实验总览",
        ha="left",
        va="top",
        fontsize=40,
        fontweight="bold",
        color="#10212D",
        fontproperties=font,
    )
    figure.text(
        0.035,
        0.947,
        "18 项单参数敏感性分析｜纵轴为综合平均排名（越低越好）｜最终参数由顺序搜索与 2×2×2 局部交互检查共同确定",
        ha="left",
        va="top",
        fontsize=22,
        color="#526273",
        fontproperties=font,
    )

    legend_handles = (
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#172A3A",
            markeredgewidth=2.0,
            markersize=12,
            label="单参数阶段最优",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#D1495B",
            markeredgecolor="white",
            markersize=17,
            label="最终交互组合选择",
        ),
    )
    figure.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.978),
        frameon=False,
        ncol=2,
        fontsize=19,
        prop=font,
        handletextpad=0.6,
        columnspacing=1.8,
    )

    figure.text(
        0.035,
        0.025,
        "说明：各面板仅用于同一参数候选值之间的内部比较，不应按纵轴绝对值跨参数比较；目标缺陷比例在初筛与统一 Source-CV 协议下分别测试，因此保留两幅结果。",
        ha="left",
        va="bottom",
        fontsize=19,
        color="#526273",
        fontproperties=font,
    )
    figure.text(
        0.965,
        0.025,
        "Source-CV：3 折 × 5 个筛选种子；最终组合另以 10 个种子进行完整 CPDP 验证",
        ha="right",
        va="bottom",
        fontsize=18,
        color="#526273",
        fontproperties=font,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_PATH, dpi=160, bbox_inches=None)
    figure.savefig(PDF_PATH, bbox_inches=None)
    plt.close(figure)
    return PNG_PATH, PDF_PATH


if __name__ == "__main__":
    png_path, pdf_path = create_figure()
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
