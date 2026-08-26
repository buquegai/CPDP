from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


OUTPUT_DIR = Path(
    r"D:\StudyOfPostgraduate\Papers\CPDP\图片\组会汇报\IPGDO过采样性能对比"
)
FIGURE_1 = OUTPUT_DIR / "01_五种过采样方法前五项指标宏平均.png"
FIGURE_2 = OUTPUT_DIR / "02_IPGDO相对最强对手前五项指标提升.png"

METRICS = ("AUC", "AUPRC", "MCC", "F1", "G-mean")
METHODS = ("不使用过采样", "ROS", "SMOTE", "TabDDPM", "IPGDO")
MEANS = {
    "不使用过采样": (0.7214833181, 0.4710092491, 0.2482622178, 0.4028707581, 0.5878089810),
    "ROS": (0.7229381166, 0.4693534411, 0.2576602459, 0.4094436992, 0.5886128449),
    "SMOTE": (0.7174938118, 0.4651314657, 0.2538916062, 0.4092036417, 0.5853693386),
    "TabDDPM": (0.7238664640, 0.4693304410, 0.2502262243, 0.4044083615, 0.5856744776),
    "IPGDO": (0.7276311011, 0.4719762749, 0.2636124712, 0.4152269023, 0.5928420419),
}
BEST_COMPETITORS = {
    "AUC": ("TabDDPM", 0.7238664640),
    "AUPRC": ("不使用过采样", 0.4710092491),
    "MCC": ("ROS", 0.2576602459),
    "F1": ("ROS", 0.4094436992),
    "G-mean": ("ROS", 0.5886128449),
}

NAVY = "#183446"
TEAL = "#0F766E"
TEAL_LIGHT = "#D9EEEA"
GOLD = "#D29A2E"
GOLD_LIGHT = "#FFF0CF"
INK = "#17232C"
MUTED = "#586773"
GRID = "#D8E0E5"
ROW_ALT = "#F5F7F8"
WHITE = "#FFFFFF"


def _font() -> FontProperties:
    candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    )
    font_path = next((path for path in candidates if path.is_file()), None)
    if font_path is None:
        raise FileNotFoundError("未找到中文字体")
    return FontProperties(fname=str(font_path))


def _draw_cell(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    font: FontProperties,
    *,
    fill: str = WHITE,
    color: str = INK,
    size: float = 24,
    weight: str = "normal",
    align: str = "center",
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=fill,
            edgecolor=GRID,
            linewidth=1.4,
        )
    )
    text_x = x + width / 2 if align == "center" else x + width * 0.08
    ax.text(
        text_x,
        y + height / 2,
        text,
        ha=align,
        va="center",
        fontsize=size,
        fontweight=weight,
        color=color,
        fontproperties=font,
    )


def _base_figure(title: str, subtitle: str, font: FontProperties) -> tuple[plt.Figure, plt.Axes]:
    figure, ax = plt.subplots(figsize=(16, 9), dpi=240)
    figure.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.055,
        0.925,
        title,
        ha="left",
        va="top",
        fontsize=35,
        fontweight="bold",
        color=NAVY,
        fontproperties=font,
    )
    ax.text(
        0.055,
        0.842,
        subtitle,
        ha="left",
        va="top",
        fontsize=20,
        color=MUTED,
        fontproperties=font,
    )
    ax.add_patch(Rectangle((0.055, 0.802), 0.89, 0.006, facecolor=TEAL, edgecolor="none"))
    return figure, ax


def create_method_average_figure(font: FontProperties) -> Path:
    figure, ax = _base_figure(
        "五种过采样策略的项目宏平均性能",
        "25 个目标项目宏平均，每个项目权重相同",
        font,
    )
    left = 0.055
    bottom = 0.205
    table_width = 0.89
    table_height = 0.56
    row_count = len(METRICS) + 1
    column_widths = (0.16, 0.20, 0.14, 0.14, 0.16, 0.20)
    row_height = table_height / row_count

    headers = ("指标", *METHODS)
    x = left
    for index, (header, fraction) in enumerate(zip(headers, column_widths)):
        width = table_width * fraction
        fill = TEAL if header == "IPGDO" else NAVY
        _draw_cell(
            ax,
            x,
            bottom + table_height - row_height,
            width,
            row_height,
            header,
            font,
            fill=fill,
            color=WHITE,
            size=23,
            weight="bold",
        )
        x += width

    for row_index, metric in enumerate(METRICS):
        y = bottom + table_height - row_height * (row_index + 2)
        base_fill = ROW_ALT if row_index % 2 else WHITE
        _draw_cell(
            ax,
            left,
            y,
            table_width * column_widths[0],
            row_height,
            metric,
            font,
            fill=base_fill,
            size=24,
            weight="bold",
            align="left",
        )
        values = [MEANS[method][row_index] for method in METHODS]
        ordered = sorted(values, reverse=True)
        x = left + table_width * column_widths[0]
        for method_index, (method, fraction, value) in enumerate(
            zip(METHODS, column_widths[1:], values)
        ):
            width = table_width * fraction
            if value == ordered[0]:
                fill, color, weight = TEAL_LIGHT, TEAL, "bold"
            elif value == ordered[1]:
                fill, color, weight = GOLD_LIGHT, INK, "bold"
            else:
                fill, color, weight = base_fill, INK, "normal"
            _draw_cell(
                ax,
                x,
                y,
                width,
                row_height,
                f"{value:.4f}",
                font,
                fill=fill,
                color=color,
                size=25,
                weight=weight,
            )
            x += width

    ax.text(
        0.055,
        0.135,
        "结论：IPGDO 在 AUC、AUPRC、MCC、F1 和 G-mean 五项指标上均取得最高项目宏平均。",
        ha="left",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=TEAL,
        fontproperties=font,
    )
    ax.text(
        0.945,
        0.07,
        "深绿色：最优　浅金色：次优",
        ha="right",
        va="center",
        fontsize=16,
        color=MUTED,
        fontproperties=font,
    )
    figure.savefig(FIGURE_1, dpi=240, facecolor=WHITE)
    plt.close(figure)
    return FIGURE_1


def create_improvement_figure(font: FontProperties) -> Path:
    figure, ax = _base_figure(
        "IPGDO 相对最强对手的性能提升",
        "最强对手按每项指标的项目宏平均分别确定",
        font,
    )
    left = 0.055
    bottom = 0.205
    table_width = 0.89
    table_height = 0.56
    row_count = len(METRICS) + 1
    column_widths = (0.14, 0.18, 0.17, 0.17, 0.17, 0.17)
    row_height = table_height / row_count
    headers = ("指标", "IPGDO", "最强对手", "对手均值", "绝对提升", "相对提升")

    x = left
    for index, (header, fraction) in enumerate(zip(headers, column_widths)):
        width = table_width * fraction
        fill = TEAL if header == "IPGDO" else NAVY
        _draw_cell(
            ax,
            x,
            bottom + table_height - row_height,
            width,
            row_height,
            header,
            font,
            fill=fill,
            color=WHITE,
            size=23,
            weight="bold",
        )
        x += width

    for row_index, metric in enumerate(METRICS):
        y = bottom + table_height - row_height * (row_index + 2)
        base_fill = ROW_ALT if row_index % 2 else WHITE
        ipgdo_value = MEANS["IPGDO"][row_index]
        competitor, competitor_value = BEST_COMPETITORS[metric]
        absolute = ipgdo_value - competitor_value
        relative = absolute / competitor_value * 100
        row = (
            metric,
            f"{ipgdo_value:.4f}",
            competitor,
            f"{competitor_value:.4f}",
            f"+{absolute:.4f}",
            f"+{relative:.2f}%",
        )
        x = left
        for column_index, (text, fraction) in enumerate(zip(row, column_widths)):
            width = table_width * fraction
            if column_index == 1:
                fill, color, weight = TEAL_LIGHT, TEAL, "bold"
            elif column_index in (4, 5):
                fill, color, weight = "#E8F4EC", "#176B3A", "bold"
            else:
                fill, color, weight = base_fill, INK, "bold" if column_index == 0 else "normal"
            _draw_cell(
                ax,
                x,
                y,
                width,
                row_height,
                text,
                font,
                fill=fill,
                color=color,
                size=24,
                weight=weight,
                align="left" if column_index == 0 else "center",
            )
            x += width

    ax.text(
        0.055,
        0.135,
        "优势重点：MCC 相对提升 2.31%，F1 相对提升 1.41%；其余三项指标也保持领先。",
        ha="left",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=TEAL,
        fontproperties=font,
    )
    ax.text(
        0.945,
        0.07,
        "绝对提升 = IPGDO均值 - 最强对手均值",
        ha="right",
        va="center",
        fontsize=16,
        color=MUTED,
        fontproperties=font,
    )
    figure.savefig(FIGURE_2, dpi=240, facecolor=WHITE)
    plt.close(figure)
    return FIGURE_2


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    font = _font()
    plt.rcParams["axes.unicode_minus"] = False
    first = create_method_average_figure(font)
    second = create_improvement_figure(font)
    print(f"Figure 1: {first}")
    print(f"Figure 2: {second}")


if __name__ == "__main__":
    main()
