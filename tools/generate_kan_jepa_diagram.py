from __future__ import annotations

import argparse
import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


WIDTH = 3800
HEIGHT = 3300
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")


@dataclass(frozen=True)
class Box:
    key: str
    x: int
    y: int
    width: int
    height: int
    title: str
    body: str
    fill: str
    stroke: str
    title_fill: str

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2

    @property
    def top(self) -> Tuple[int, int]:
        return self.center_x, self.y

    @property
    def bottom(self) -> Tuple[int, int]:
        return self.center_x, self.y + self.height

    @property
    def left(self) -> Tuple[int, int]:
        return self.x, self.center_y

    @property
    def right(self) -> Tuple[int, int]:
        return self.x + self.width, self.center_y


@dataclass(frozen=True)
class Group:
    x: int
    y: int
    width: int
    height: int
    title: str
    fill: str
    stroke: str


@dataclass(frozen=True)
class Edge:
    points: Tuple[Tuple[int, int], ...]
    label: str
    label_x: int
    label_y: int
    color: str = "#425466"
    dashed: bool = False


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


TITLE_FONT = font(58)
SUBTITLE_FONT = font(29)
GROUP_FONT = font(33)
BOX_TITLE_FONT = font(31)
BOX_BODY_FONT = font(25)
EDGE_FONT = font(22)
FOOT_FONT = font(24)


def wrap_text(text: str, selected_font, max_width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for character in paragraph:
            candidate = current + character
            if current and selected_font.getlength(candidate) > max_width:
                lines.append(current)
                current = character
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    selected_font,
    center_x: int,
    start_y: int,
    fill: str,
    spacing: int,
) -> None:
    y = start_y
    for line in lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=selected_font)
        draw.text(
            (center_x - (right - left) / 2, y),
            line,
            font=selected_font,
            fill=fill,
        )
        y += bottom - top + spacing


def draw_group(draw: ImageDraw.ImageDraw, group: Group) -> None:
    draw.rounded_rectangle(
        (group.x, group.y, group.x + group.width, group.y + group.height),
        radius=30,
        fill=group.fill,
        outline=group.stroke,
        width=4,
    )
    draw.rounded_rectangle(
        (group.x + 28, group.y - 27, group.x + 430, group.y + 45),
        radius=22,
        fill=group.stroke,
    )
    draw.text(
        (group.x + 52, group.y - 18),
        group.title,
        font=GROUP_FONT,
        fill="#FFFFFF",
    )


def draw_box(draw: ImageDraw.ImageDraw, box: Box) -> None:
    draw.rounded_rectangle(
        (box.x, box.y, box.x + box.width, box.y + box.height),
        radius=24,
        fill=box.fill,
        outline=box.stroke,
        width=4,
    )
    draw.rounded_rectangle(
        (box.x, box.y, box.x + box.width, box.y + 62),
        radius=22,
        fill=box.title_fill,
    )
    draw.rectangle(
        (box.x, box.y + 38, box.x + box.width, box.y + 64),
        fill=box.title_fill,
    )
    title_lines = wrap_text(box.title, BOX_TITLE_FONT, box.width - 32)
    draw_centered_lines(
        draw,
        title_lines,
        BOX_TITLE_FONT,
        box.center_x,
        box.y + 10,
        "#FFFFFF",
        4,
    )
    body_lines = wrap_text(box.body, BOX_BODY_FONT, box.width - 40)
    total_height = len(body_lines) * 33
    body_start = max(box.y + 80, box.center_y - total_height // 2 + 24)
    draw_centered_lines(
        draw,
        body_lines,
        BOX_BODY_FONT,
        box.center_x,
        body_start,
        "#17202A",
        7,
    )


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[int, int]],
    color: str,
    width: int,
    dashed: bool,
) -> None:
    for start, end in zip(points, points[1:]):
        if not dashed:
            draw.line((start, end), fill=color, width=width)
            continue
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        unit_x, unit_y = dx / length, dy / length
        distance = 0.0
        while distance < length:
            segment_end = min(distance + 16, length)
            draw.line(
                (
                    start[0] + unit_x * distance,
                    start[1] + unit_y * distance,
                    start[0] + unit_x * segment_end,
                    start[1] + unit_y * segment_end,
                ),
                fill=color,
                width=width,
            )
            distance += 28


def draw_edge(draw: ImageDraw.ImageDraw, edge: Edge) -> None:
    draw_polyline(draw, edge.points, edge.color, 5, edge.dashed)
    start = edge.points[-2]
    end = edge.points[-1]
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_length = 24
    spread = 0.58
    arrow = [
        end,
        (
            end[0] - arrow_length * math.cos(angle - spread),
            end[1] - arrow_length * math.sin(angle - spread),
        ),
        (
            end[0] - arrow_length * math.cos(angle + spread),
            end[1] - arrow_length * math.sin(angle + spread),
        ),
    ]
    draw.polygon(arrow, fill=edge.color)
    if edge.label:
        lines = wrap_text(edge.label, EDGE_FONT, 470)
        widths = [EDGE_FONT.getlength(line) for line in lines]
        label_width = int(max(widths, default=0) + 28)
        label_height = len(lines) * 29 + 18
        x1 = edge.label_x - label_width // 2
        y1 = edge.label_y - label_height // 2
        draw.rounded_rectangle(
            (x1, y1, x1 + label_width, y1 + label_height),
            radius=12,
            fill="#FFFFFF",
            outline="#C5CED8",
            width=2,
        )
        draw_centered_lines(
            draw,
            lines,
            EDGE_FONT,
            edge.label_x,
            y1 + 7,
            "#2E4053",
            4,
        )


def svg_text_lines(
    lines: Sequence[str],
    x: float,
    y: float,
    size: int,
    fill: str,
    weight: int = 400,
    line_height: int = 34,
) -> str:
    spans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else line_height
        spans.append(
            f'<tspan x="{x}" dy="{dy}">{html.escape(line)}</tspan>'
        )
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" '
        f'font-family="Microsoft YaHei, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{"".join(spans)}</text>'
    )


def svg_box(box: Box) -> str:
    title_lines = wrap_text(box.title, BOX_TITLE_FONT, box.width - 32)
    body_lines = wrap_text(box.body, BOX_BODY_FONT, box.width - 40)
    total_height = len(body_lines) * 33
    body_start = max(box.y + 105, box.center_y - total_height // 2 + 47)
    return "".join(
        (
            f'<rect x="{box.x}" y="{box.y}" width="{box.width}" '
            f'height="{box.height}" rx="24" fill="{box.fill}" '
            f'stroke="{box.stroke}" stroke-width="4"/>',
            f'<path d="M {box.x + 24} {box.y} H {box.x + box.width - 24} '
            f'Q {box.x + box.width} {box.y} {box.x + box.width} {box.y + 24} '
            f'V {box.y + 64} H {box.x} V {box.y + 24} '
            f'Q {box.x} {box.y} {box.x + 24} {box.y} Z" fill="{box.title_fill}"/>',
            svg_text_lines(
                title_lines,
                box.center_x,
                box.y + 39,
                31,
                "#FFFFFF",
                500,
                35,
            ),
            svg_text_lines(
                body_lines,
                box.center_x,
                body_start,
                25,
                "#17202A",
                400,
                34,
            ),
        )
    )


def svg_group(group: Group) -> str:
    return "".join(
        (
            f'<rect x="{group.x}" y="{group.y}" width="{group.width}" '
            f'height="{group.height}" rx="30" fill="{group.fill}" '
            f'stroke="{group.stroke}" stroke-width="4"/>',
            f'<rect x="{group.x + 28}" y="{group.y - 27}" width="402" '
            f'height="72" rx="22" fill="{group.stroke}"/>',
            f'<text x="{group.x + 52}" y="{group.y + 22}" '
            f'font-family="Microsoft YaHei, sans-serif" font-size="33" '
            f'font-weight="500" fill="#FFFFFF">{html.escape(group.title)}</text>',
        )
    )


def svg_edge(edge: Edge) -> str:
    path = " ".join(
        ("M" if index == 0 else "L") + f" {point[0]} {point[1]}"
        for index, point in enumerate(edge.points)
    )
    dash = ' stroke-dasharray="16 12"' if edge.dashed else ""
    result = [
        f'<path d="{path}" fill="none" stroke="{edge.color}" '
        f'stroke-width="5" marker-end="url(#arrow)"{dash}/>'
    ]
    if edge.label:
        lines = wrap_text(edge.label, EDGE_FONT, 470)
        max_width = max((EDGE_FONT.getlength(line) for line in lines), default=0)
        label_width = int(max_width + 28)
        label_height = len(lines) * 29 + 18
        x1 = edge.label_x - label_width // 2
        y1 = edge.label_y - label_height // 2
        result.append(
            f'<rect x="{x1}" y="{y1}" width="{label_width}" '
            f'height="{label_height}" rx="12" fill="#FFFFFF" '
            f'stroke="#C5CED8" stroke-width="2"/>'
        )
        result.append(
            svg_text_lines(
                lines,
                edge.label_x,
                y1 + 25,
                22,
                "#2E4053",
                400,
                29,
            )
        )
    return "".join(result)


def diagram() -> Tuple[List[Group], List[Box], List[Edge]]:
    groups = [
        Group(70, 840, 1680, 1380, "KAN-JEPA 自监督表征学习", "#F7F4FC", "#7157A5"),
        Group(1840, 840, 1890, 570, "跨项目域适配", "#F2FAF7", "#27866F"),
        Group(1840, 1490, 1890, 520, "增强源项目监督分类", "#FFF8EF", "#C97722"),
        Group(1840, 2090, 1890, 350, "无标签目标一致性", "#F2F7FC", "#3778A8"),
    ]

    boxes = [
        Box("real", 80, 160, 800, 250, "真实源项目", "(X_s^p, y_s^p)\nX_s^p ∈ R^(B_p×d)\n仅真实样本", "#EDF5FC", "#3E78A8", "#3E78A8"),
        Box("aug", 1000, 160, 800, 250, "增强源项目", "(X_aug^p, y_aug^p)\n真实样本 + 扩散缺陷样本\n用于监督分类", "#FFF5E9", "#C97722", "#C97722"),
        Box("target", 1920, 160, 800, 250, "无标签目标项目", "X_t ∈ R^(B_t×d)\n允许使用特征\n禁止读取缺陷标签", "#EDF8F4", "#27866F", "#27866F"),
        Box("eval", 2840, 160, 840, 250, "目标标签隔离区", "y_t 仅由统一 evaluator 持有\n训练、调参、阈值选择均不可访问", "#FBEFF0", "#A84450", "#A84450"),
        Box("norm", 720, 500, 2380, 230, "源项目稳健标准化", "拟合：c_j=median(X_s[:,j])，r_j=IQR(X_s[:,j])+ε\n变换：x̃_j=(x_j-c_j)/r_j；同一组(c,r)应用于真实源、增强源和目标特征", "#F4F7F9", "#596A78", "#596A78"),
        Box("pool", 150, 960, 650, 170, "自监督样本池", "X_ssl={X̃_s(real), X̃_t}\n不使用扩散样本标签规律", "#FFFFFF", "#7157A5", "#7157A5"),
        Box("mask", 940, 960, 650, 170, "特征掩码生成器", "m∈{0,1}^d\n随机掩码 + 相关特征组掩码", "#FFFFFF", "#7157A5", "#7157A5"),
        Box("context", 150, 1230, 650, 200, "上下文视图", "x_c=m⊙x̃+(1-m)⊙e_mask\nv_c=concat(x_c,m)∈R^(B×2d)", "#FFFFFF", "#7157A5", "#7157A5"),
        Box("target_view", 940, 1230, 650, 200, "目标表征视图", "x_τ=m_τ⊙x̃+(1-m_τ)⊙e_mask\nm_τ 使用较低遮挡率", "#FFFFFF", "#7157A5", "#7157A5"),
        Box("context_encoder", 150, 1525, 650, 200, "KAN Context Encoder f_θ", "h_j^(l+1)=Σ_i φ_ij^(l)(h_i^l)\nz_c=f_θ(v_c)∈R^(B×h)", "#FFFFFF", "#7157A5", "#7157A5"),
        Box("target_encoder", 940, 1525, 650, 200, "EMA KAN Target Encoder f_ξ", "z_τ=stopgrad(f_ξ(x_τ))\nξ←ρξ+(1-ρ)θ；不接收梯度", "#FFFFFF", "#7157A5", "#7157A5"),
        Box("predictor", 320, 1815, 700, 170, "KAN JEPA Predictor g_φ", "e_m=MLP(m)\nz_hat=g_φ(concat(z_c,e_m))", "#FFFFFF", "#7157A5", "#7157A5"),
        Box("jepa_loss", 300, 2040, 720, 130, "JEPA 潜在预测损失", "L_JEPA=||norm(z_hat)-norm(z_τ)||²", "#FFFDF6", "#A7832E", "#A7832E"),
        Box("anti", 1100, 1815, 580, 355, "防坍塌约束", "L_var=Σ_j max(0,γ-std(Z_j))\nL_cov=Σ_(i≠j) Cov(Z)_(ij)^2\n保持潜在维度方差并降低冗余", "#FFFDF6", "#A7832E", "#A7832E"),
        Box("source_z", 1900, 960, 470, 210, "各源项目完整表征", "Z_s^p=f_θ([X̃_s^p,1_d])\n计算 μ_p 与 C_p", "#FFFFFF", "#27866F", "#27866F"),
        Box("target_z", 2470, 960, 470, 210, "目标项目完整表征", "Z_t=f_θ([X̃_t,1_d])\n计算 μ_t 与 C_t", "#FFFFFF", "#27866F", "#27866F"),
        Box("distance", 3040, 930, 620, 270, "源目标距离与权重", "D_p=||μ_p-μ_t||²+||C_p-C_t||_F²/(4h²)\nw_p=softmax(-D_p/T)", "#FFFFFF", "#27866F", "#27866F"),
        Box("coral", 2410, 1240, 780, 120, "加权 CORAL 损失", "L_align=Σ_p w_p·D_p", "#FFFDF6", "#27866F", "#27866F"),
        Box("aug_view", 1910, 1600, 500, 220, "增强样本完整视图", "v_aug=[X̃_aug,1_d]\n输入维度 B_a×2d", "#FFFFFF", "#C97722", "#C97722"),
        Box("shared", 2520, 1600, 500, 220, "共享 Context Encoder", "z_aug=f_θ(v_aug)\n与JEPA共用参数 θ", "#FFFFFF", "#C97722", "#C97722"),
        Box("classifier", 3130, 1600, 500, 220, "缺陷分类头 h_ψ", "logit=h_ψ(z_aug)\np=sigmoid(logit)", "#FFFFFF", "#C97722", "#C97722"),
        Box("bce", 2470, 1870, 900, 105, "源项目加权分类损失", "L_cls=-Σ_i w_project(i)[y_i log p_i+(1-y_i)log(1-p_i)]", "#FFFDF6", "#C97722", "#C97722"),
        Box("views", 1910, 2170, 520, 190, "目标样本双视图", "x_t^(1), x_t^(2)\n不同掩码或轻微扰动", "#FFFFFF", "#3778A8", "#3778A8"),
        Box("shared_predict", 2570, 2170, 600, 190, "共享编码与分类", "p_1=H(x_t^(1))\np_2=H(x_t^(2))", "#FFFFFF", "#3778A8", "#3778A8"),
        Box("consistency", 3310, 2170, 350, 190, "一致性损失", "L_cons=mean((p_1-p_2)^2)", "#FFFDF6", "#3778A8", "#3778A8"),
        Box("total", 320, 2580, 1550, 300, "联合训练目标", "L_total=λ_j L_JEPA + λ_c L_cls + λ_a L_align\n+ λ_k L_cons + λ_v(L_var+L_cov)\n梯度更新 θ、φ、ψ；Target Encoder 仅EMA更新", "#FBEFF0", "#A84450", "#A84450"),
        Box("inference", 2070, 2540, 1630, 260, "目标项目推理路径", "X_t → 稳健标准化 → [X̃_t,1_d] → f_θ → h_ψ → p_t\n阈值 τ* 由源项目内层验证确定；ŷ_t=1[p_t≥τ*]", "#EEF2F5", "#34495E", "#34495E"),
        Box("evaluator", 2250, 2940, 1280, 190, "统一评估器", "输入：p_t、ŷ_t、隔离的 y_t、effort_t\n输出：AUC、AUPRC、MCC、F1、G-mean、Popt@20%", "#FBEFF0", "#A84450", "#A84450"),
    ]

    by_key = {box.key: box for box in boxes}
    edges = [
        Edge((by_key["real"].bottom, (480, 460), (1280, 460), (1280, 500)), "X_s(real)：拟合 c、r", 820, 454),
        Edge((by_key["aug"].bottom, (1400, 470), (1720, 470), (1720, 500)), "X_aug,y_aug：仅复用 c、r", 1570, 454),
        Edge((by_key["target"].bottom, (2320, 470), (2380, 470), (2380, 500)), "X_t：无标签", 2370, 444),
        Edge((by_key["norm"].bottom, (1910, 790), (475, 790), by_key["pool"].top), "X̃_ssl={X̃_s(real),X̃_t}", 1120, 790),
        Edge((by_key["pool"].right, by_key["mask"].left), "x̃∈R^(B×d)", 870, 1045),
        Edge((by_key["mask"].bottom, (1265, 1180), (475, 1180), by_key["context"].top), "m：上下文掩码", 855, 1180),
        Edge((by_key["mask"].bottom, by_key["target_view"].top), "m_τ：低遮挡率", 1260, 1185),
        Edge((by_key["context"].bottom, by_key["context_encoder"].top), "v_c∈R^(B×2d)", 480, 1480),
        Edge((by_key["target_view"].bottom, by_key["target_encoder"].top), "x_τ∈R^(B×d)", 1265, 1480),
        Edge((by_key["context_encoder"].bottom, (475, 1775), (670, 1775), by_key["predictor"].top), "z_c∈R^(B×h)", 530, 1770),
        Edge((by_key["mask"].right, (1660, 1045), (1660, 1770), (850, 1770), (850, 1815)), "掩码嵌入 e_m", 1530, 1680),
        Edge((by_key["predictor"].bottom, by_key["jepa_loss"].top), "z_hat∈R^(B×h)", 670, 2010),
        Edge((by_key["target_encoder"].bottom, (1265, 1995), (900, 1995), (900, 2040)), "stopgrad(z_τ)", 1110, 1995),
        Edge((by_key["context_encoder"].right, (1060, 1625), (1060, 1770), (1390, 1770), by_key["anti"].top), "批次表征 Z", 1170, 1760),
        Edge((by_key["context_encoder"].right, (870, 1625), by_key["target_encoder"].left), "EMA参数：ξ←ρξ+(1-ρ)θ", 870, 1590, color="#7157A5", dashed=True),
        Edge(((3100, 615), (3240, 615), (3240, 900), by_key["source_z"].top), "X̃_s^p；完整掩码 1_d", 2950, 790),
        Edge(((3100, 645), (3470, 645), (3470, 880), (2705, 880), by_key["target_z"].top), "X̃_t；完整掩码 1_d", 3360, 700),
        Edge((by_key["source_z"].right, (2800, 1065), by_key["distance"].left), "μ_p,C_p", 2760, 1040),
        Edge((by_key["target_z"].right, by_key["distance"].left), "μ_t,C_t", 2990, 1145),
        Edge((by_key["distance"].bottom, (3350, 1300), (3190, 1300)), "D_p,w_p", 3360, 1270),
        Edge(((3100, 680), (3680, 680), (3680, 1550), (2160, 1550), by_key["aug_view"].top), "X̃_aug,y_aug", 3610, 1470),
        Edge((by_key["aug_view"].right, by_key["shared"].left), "v_aug=[X̃_aug,1_d]", 2465, 1710),
        Edge((by_key["shared"].right, by_key["classifier"].left), "z_aug∈R^(B_a×h)", 3075, 1710),
        Edge((by_key["classifier"].bottom, (3380, 1850), (3130, 1850), by_key["bce"].top), "p=sigmoid(logit)", 3350, 1850),
        Edge((by_key["distance"].bottom, (3560, 1450), (3560, 1930), by_key["bce"].right), "源项目权重 w_p", 3590, 1730),
        Edge(((2720, 410), (2780, 450), (3700, 450), (3700, 2120), by_key["views"].top), "同一 X̃_t 的两种视图", 3620, 2050),
        Edge((by_key["views"].right, by_key["shared_predict"].left), "x_t^(1),x_t^(2)", 2500, 2265),
        Edge((by_key["shared_predict"].right, by_key["consistency"].left), "p_1,p_2", 3240, 2265),
        Edge((by_key["jepa_loss"].bottom, (660, 2460), (650, 2460), by_key["total"].top), "L_JEPA", 570, 2460),
        Edge((by_key["anti"].bottom, (1390, 2480), by_key["total"].top), "L_var+L_cov", 1420, 2470),
        Edge((by_key["coral"].bottom, (2800, 1450), (1800, 1450), (1800, 2530), (1650, 2530), by_key["total"].top), "L_align", 1770, 2390),
        Edge((by_key["bce"].bottom, (2920, 2050), (1900, 2050), (1900, 2510), (1550, 2510), by_key["total"].top), "L_cls", 1920, 2310),
        Edge((by_key["consistency"].bottom, (3485, 2490), (1800, 2490), by_key["total"].top), "L_cons", 2580, 2485),
        Edge((by_key["total"].right, by_key["inference"].left), "训练完成的 θ、ψ", 1970, 2685, color="#A84450", dashed=True),
        Edge(((2720, 350), (2760, 440), (3740, 440), (3740, 2500), by_key["inference"].top), "X_t：完整特征", 3680, 2430),
        Edge((by_key["inference"].bottom, (2885, 2890), by_key["evaluator"].top), "p_t、ŷ_t、effort_t", 2920, 2885),
        Edge((by_key["eval"].bottom, (3260, 460), (3760, 460), (3760, 3040), by_key["evaluator"].right), "隔离标签 y_t；仅评估时传入", 3670, 2960, color="#A84450", dashed=True),
    ]
    return groups, boxes, edges


def render_png(path: Path, groups: Sequence[Group], boxes: Sequence[Box], edges: Sequence[Edge]) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    draw.text((WIDTH // 2, 55), "KAN-JEPA 跨项目软件缺陷预测模型详细框架", font=TITLE_FONT, fill="#17202A", anchor="ma")
    draw.text(
        (WIDTH // 2, 122),
        "箭头标注传输数据；模块内部给出主要数学变换；目标项目标签在训练阶段完全隔离",
        font=SUBTITLE_FONT,
        fill="#52616B",
        anchor="ma",
    )
    for group in groups:
        draw_group(draw, group)
    for edge in edges:
        draw_edge(draw, edge)
    for box in boxes:
        draw_box(draw, box)
    draw.text(
        (80, 3230),
        "符号：d=输入特征数，h=潜在维度，p=源项目编号，m=二进制特征掩码，⊙=逐元素乘法，C=协方差矩阵。",
        font=FOOT_FONT,
        fill="#52616B",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True, dpi=(300, 300))


def render_svg(path: Path, groups: Sequence[Group], boxes: Sequence[Box], edges: Sequence[Edge]) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M 0 0 L 12 6 L 0 12 Z" fill="#425466"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        svg_text_lines(["KAN-JEPA 跨项目软件缺陷预测模型详细框架"], WIDTH / 2, 100, 58, "#17202A", 500, 64),
        svg_text_lines(["箭头标注传输数据；模块内部给出主要数学变换；目标项目标签在训练阶段完全隔离"], WIDTH / 2, 153, 29, "#52616B", 400, 34),
    ]
    parts.extend(svg_group(group) for group in groups)
    parts.extend(svg_edge(edge) for edge in edges)
    parts.extend(svg_box(box) for box in boxes)
    parts.append(
        '<text x="80" y="3260" font-family="Microsoft YaHei, sans-serif" '
        'font-size="24" fill="#52616B">符号：d=输入特征数，h=潜在维度，p=源项目编号，m=二进制特征掩码，⊙=逐元素乘法，C=协方差矩阵。</text>'
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    groups, boxes, edges = diagram()
    png_path = args.output_directory / "kan-jepa-predictor-detailed-architecture.png"
    svg_path = args.output_directory / "kan-jepa-predictor-detailed-architecture.svg"
    render_png(png_path, groups, boxes, edges)
    render_svg(svg_path, groups, boxes, edges)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
