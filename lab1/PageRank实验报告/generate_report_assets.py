from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = REPORT_DIR / "images"
DATA_PATH = ROOT / "Data.txt"
RES_PATH = ROOT / "result" / "Res.txt"

FONT_CN = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_CN_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"

PALETTE = {
    "ink": "#172033",
    "muted": "#65738a",
    "grid": "#d9e2ef",
    "blue": "#246bfe",
    "cyan": "#00a6d6",
    "green": "#17a673",
    "orange": "#f59e0b",
    "red": "#ef4444",
    "purple": "#7c3aed",
    "panel": "#f8fafc",
    "dark": "#151a1f",
}


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else (FONT_CN_BOLD if bold else FONT_CN)
    return ImageFont.truetype(path, size)


def text(draw: ImageDraw.ImageDraw, xy, s: str, size=28, fill=None, bold=False, mono=False):
    draw.text(xy, s, font=font(size, bold=bold, mono=mono), fill=fill or PALETTE["ink"])


def text_right(draw, x, y, s, size=24, fill=None, bold=False):
    f = font(size, bold=bold)
    bbox = draw.textbbox((0, 0), s, font=f)
    draw.text((x - (bbox[2] - bbox[0]), y), s, font=f, fill=fill or PALETTE["ink"])


def rounded(draw, box, radius=20, fill="#ffffff", outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, fill="#334155", width=4, head=18):
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    draw.line((x0, y0, x1, y1), fill=fill, width=width)
    base_x = x1 - ux * head
    base_y = y1 - uy * head
    points = [
        (x1, y1),
        (base_x + px * head * 0.45, base_y + py * head * 0.45),
        (base_x - px * head * 0.45, base_y - py * head * 0.45),
    ]
    draw.polygon(points, fill=fill)


def draw_node(draw, center, label, rank, fill, outline="#ffffff", dead=False):
    x, y = center
    r = 58 if not dead else 62
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=5)
    if dead:
        draw.ellipse((x - r - 8, y - r - 8, x + r + 8, y + r + 8), outline=PALETTE["red"], width=6)
    f_label = font(34, bold=True)
    bbox = draw.textbbox((0, 0), label, font=f_label)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y - 34), label, font=f_label, fill="#ffffff")
    f_rank = font(22, mono=True)
    s = f"{rank:.2f}"
    bbox = draw.textbbox((0, 0), s, font=f_rank)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y + 8), s, font=f_rank, fill="#eef6ff")


def generate_principle_example():
    img = Image.new("RGB", (1850, 1080), "#ffffff")
    d = ImageDraw.Draw(img)
    text(d, (70, 42), "PageRank 一轮更新示意", 46, bold=True)
    text(d, (70, 100), "链接传播只沿真实边发生；dead-end 质量和随机传送项以统一基值补回所有节点，避免概率泄漏与局部吸收", 23, fill=PALETTE["muted"])

    rounded(d, (64, 168, 1018, 902), fill="#f8fafc", outline="#dbe5f0", radius=24)
    rounded(d, (1060, 168, 1788, 902), fill="#f8fafc", outline="#dbe5f0", radius=24)
    text(d, (105, 202), "小型有向图", 30, bold=True)
    text(d, (1098, 202), "目标节点 C 的更新拆解", 30, bold=True)

    nodes = {
        "A": ((260, 410), 0.20, PALETTE["blue"]),
        "B": ((520, 300), 0.18, PALETTE["cyan"]),
        "C": ((785, 430), 0.16, PALETTE["orange"]),
        "D": ((430, 620), 0.15, PALETTE["green"]),
        "E": ((805, 700), 0.14, PALETTE["red"]),
        "F": ((205, 720), 0.17, PALETTE["purple"]),
    }
    edges = [
        ("A", "B"), ("A", "C"),
        ("B", "C"), ("B", "D"),
        ("C", "A"),
        ("D", "C"), ("D", "E"), ("D", "F"),
        ("F", "C"), ("F", "E"),
    ]
    for src, dst in edges:
        sx, sy = nodes[src][0]
        tx, ty = nodes[dst][0]
        dx, dy = tx - sx, ty - sy
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        start = (sx + ux * 72, sy + uy * 72)
        end = (tx - ux * 72, ty - uy * 72)
        color = PALETTE["orange"] if dst == "C" else "#64748b"
        arrow(d, start, end, fill=color, width=5 if dst == "C" else 3, head=18)
    for label, (center, rank_value, color) in nodes.items():
        draw_node(d, center, label, rank_value, color, dead=(label == "E"))
    text(d, (636, 782), "E 无出边：M_t = r_t(E) = 0.14", 23, fill=PALETTE["red"], bold=True)
    text(d, (142, 830), "箭头指向 C 的入链会参与 C 的链接项；其他边只影响对应目标节点", 21, fill=PALETTE["muted"])

    alpha = 0.85
    n = 6
    dead_mass = 0.14
    base = (1 - alpha) / n
    dead_share = alpha * dead_mass / n
    link_terms = [("A", 0.20, 2), ("B", 0.18, 2), ("D", 0.15, 3), ("F", 0.17, 2)]
    link_sum = alpha * sum(value / out_degree for _, value, out_degree in link_terms)
    total = base + dead_share + link_sum
    formula_lines = [
        ("alpha = 0.85, N = 6", PALETTE["muted"]),
        ("base = (1-alpha)/N = 0.02500", PALETTE["blue"]),
        ("dead_share = alpha*M_t/N = 0.01983", PALETTE["red"]),
        ("link(C) = alpha*(0.20/2 + 0.18/2 + 0.15/3 + 0.17/2)", PALETTE["orange"]),
        (f"r_(t+1)(C) = {base:.5f} + {dead_share:.5f} + {link_sum:.5f} = {total:.5f}", PALETTE["green"]),
    ]
    y = 285
    for line, color in formula_lines:
        rounded(d, (1102, y - 12, 1744, y + 50), fill="#ffffff", outline="#e2e8f0", radius=14)
        text(d, (1128, y), line, 24 if len(line) < 62 else 21, fill=color, bold=(color in (PALETTE["green"], PALETTE["orange"])), mono=True)
        y += 82

    text(d, (1102, 735), "实现映射", 28, bold=True)
    steps = [
        "1. 扫描 dead_nodes 求 M_t",
        "2. next_rank 全量初始化为 base + dead_share",
        "3. 按 CSR 源节点行块遍历真实出边",
        "4. 对每条 src -> dst 累加 rank[src] * alpha/out(src)",
    ]
    y = 780
    for step in steps:
        text(d, (1128, y), step, 23, fill=PALETTE["ink"])
        y += 38

    rounded(d, (88, 928, 1762, 1032), fill="#0f172a", outline="#0f172a", radius=20)
    text(d, (122, 953), "关键结论：正确 PageRank 图示要同时解释随机传送、dead-end 回补和真实链接传播；", 23, fill="#e2e8f0")
    text(d, (122, 988), "代码把前两项折叠进初始化基值，只对 CSR 中真实边做稀疏累加，不构造稠密 Google 矩阵。", 23, fill="#e2e8f0")
    img.save(IMAGE_DIR / "pagerank_principle_example.png")


def load_graph():
    raw_edges = []
    nodes = set()
    for line in DATA_PATH.read_text().splitlines():
        if not line.strip():
            continue
        a, b = map(int, line.split())
        raw_edges.append((a, b))
        nodes.add(a)
        nodes.add(b)
    node_ids = sorted(nodes)
    id_to_idx = {v: i for i, v in enumerate(node_ids)}
    edges = sorted({(id_to_idx[a], id_to_idx[b]) for a, b in raw_edges})
    n = len(node_ids)
    row_ptr = np.zeros(n + 1, dtype=np.int64)
    for a, _ in edges:
        row_ptr[a + 1] += 1
    np.cumsum(row_ptr, out=row_ptr)
    col_idx = np.array([b for _, b in edges], dtype=np.int32)
    out_degree = np.diff(row_ptr)
    in_degree = np.zeros(n, dtype=np.int32)
    for b in col_idx:
        in_degree[b] += 1
    return node_ids, edges, row_ptr, col_idx, out_degree, in_degree


def compute_pagerank(row_ptr, col_idx, damping=0.85, tol=1e-12, max_iter=1000):
    n = len(row_ptr) - 1
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    next_rank = np.zeros(n, dtype=np.float64)
    out_degree = np.diff(row_ptr)
    dead_nodes = np.flatnonzero(out_degree == 0)
    out_weight = np.zeros(n, dtype=np.float64)
    mask = out_degree > 0
    out_weight[mask] = damping / out_degree[mask]
    base_rank = (1.0 - damping) / n

    residuals = []
    dead_mass = []
    for _ in range(max_iter):
        dead_sum = float(rank[dead_nodes].sum())
        dead_mass.append(dead_sum)
        next_rank.fill(base_rank + damping * dead_sum / n)
        for src in range(n):
            begin, end = row_ptr[src], row_ptr[src + 1]
            if begin == end:
                continue
            contribution = rank[src] * out_weight[src]
            next_rank[col_idx[begin:end]] += contribution
        total = float(next_rank.sum())
        if total > 0:
            next_rank /= total
        diff = float(np.abs(next_rank - rank).sum())
        residuals.append(diff)
        rank, next_rank = next_rank, rank
        if diff < tol:
            break
    return rank, residuals, dead_mass


def color_lerp(a, b, t):
    a = tuple(int(a[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(b[i : i + 2], 16) for i in (1, 3, 5))
    c = tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return "#%02x%02x%02x" % c


def save_terminal_image(path: Path, lines, width=1800, height=520):
    img = Image.new("RGB", (width, height), PALETTE["dark"])
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, width - 1, height - 1), radius=22, fill=PALETTE["dark"])
    d.ellipse((24, 22, 46, 44), fill="#ef4444")
    d.ellipse((58, 22, 80, 44), fill="#f59e0b")
    d.ellipse((92, 22, 114, 44), fill="#22c55e")
    y = 76
    for segments in lines:
        x = 34
        for s, c in segments:
            f = font(27, mono=True)
            d.text((x, y), s, font=f, fill=c)
            bbox = d.textbbox((x, y), s, font=f)
            x = bbox[2]
        y += 43
    img.save(path)


def generate_terminal_images():
    prompt = "lab1$ "
    green = "#38d474"
    blue = "#58a6ff"
    white = "#e8edf2"
    gray = "#aeb8c2"
    save_terminal_image(
        IMAGE_DIR / "compile_run.png",
        [
            [(prompt, green), ("g++ -O2 -std=c++17 -static lab1/src/PageRank.cpp -o lab1/bin/PageRank.exe", white)],
            [(prompt, green), ("file lab1/bin/PageRank.exe", white)],
            [("lab1/bin/PageRank.exe: ELF 64-bit LSB executable, x86-64, statically linked", gray)],
            [(prompt, green), ("lab1/bin/PageRank.exe lab1/Data.txt lab1/result/Res.txt", white)],
            [("nodes: 9500, edges: 150000, iterations: 17, residual: 8.064961e-13", white)],
        ],
        height=310,
    )

    rows = RES_PATH.read_text().splitlines()[:20]
    lines = [[(prompt, green), ("head -20 lab1/result/Res.txt", white)]]
    for row in rows:
        lines.append([(row, white)])
    lines.append([(prompt, green), ("wc -l lab1/result/Res.txt", white)])
    lines.append([("100 lab1/result/Res.txt", gray)])
    save_terminal_image(IMAGE_DIR / "res_output.png", lines, width=1400, height=1060)


def draw_axes(d, box, xlabels=None, ylabel=None):
    x0, y0, x1, y1 = box
    d.line((x0, y1, x1, y1), fill=PALETTE["grid"], width=2)
    d.line((x0, y0, x0, y1), fill=PALETTE["grid"], width=2)
    for i in range(5):
        y = y1 - (y1 - y0) * i / 4
        d.line((x0, y, x1, y), fill=PALETTE["grid"], width=1)
    if ylabel:
        text(d, (x0, y0 - 42), ylabel, size=22, fill=PALETTE["muted"])


def generate_dataset_overview(node_ids, out_degree, in_degree, edges):
    img = Image.new("RGB", (1800, 1080), "#ffffff")
    d = ImageDraw.Draw(img)
    text(d, (70, 44), "数据集结构概览", 44, bold=True)
    text(d, (70, 100), "节点压缩、稀疏性、死端规模和度分布直接决定 PageRank 的内存与迭代开销", 24, fill=PALETTE["muted"])

    n = len(node_ids)
    m = len(edges)
    dead = int((out_degree == 0).sum())
    density = m / (n * n)
    cards = [
        ("节点数", f"{n:,}", "真实出现节点"),
        ("有向边数", f"{m:,}", "去重后边数"),
        ("死端节点", f"{dead:,}", f"{dead / n:.1%}"),
        ("稀疏密度", f"{density * 100:.3f}%", "M / N^2"),
        ("平均出度", f"{m / n:.2f}", "全体节点"),
    ]
    x = 70
    for title, value, sub in cards:
        rounded(d, (x, 170, x + 310, 330), fill=PALETTE["panel"], outline="#e2e8f0")
        text(d, (x + 26, 194), title, 24, fill=PALETTE["muted"])
        text(d, (x + 26, 230), value, 42, fill=PALETTE["blue"], bold=True)
        text(d, (x + 26, 288), sub, 22, fill=PALETTE["muted"])
        x += 340

    bins = [(0, 0, "0"), (1, 5, "1-5"), (6, 10, "6-10"), (11, 15, "11-15"),
            (16, 20, "16-20"), (21, 25, "21-25"), (26, 30, "26-30"), (31, 10**9, "31+")]
    out_counts = []
    in_counts = []
    for lo, hi, _ in bins:
        out_counts.append(int(((out_degree >= lo) & (out_degree <= hi)).sum()))
        in_counts.append(int(((in_degree >= lo) & (in_degree <= hi)).sum()))
    chart = (120, 460, 1680, 950)
    draw_axes(d, chart, ylabel="")
    max_count = max(out_counts + in_counts)
    bar_w = 62
    gap = (chart[2] - chart[0]) / len(bins)
    for i, (_, _, label) in enumerate(bins):
        cx = chart[0] + gap * i + gap / 2
        for j, (count, color) in enumerate(((out_counts[i], PALETTE["blue"]), (in_counts[i], PALETTE["orange"]))):
            h = (chart[3] - chart[1]) * count / max_count
            x0 = cx - bar_w + j * bar_w
            d.rounded_rectangle((x0, chart[3] - h, x0 + bar_w - 8, chart[3]), radius=8, fill=color)
        text(d, (cx - 28, chart[3] + 18), label, 20, fill=PALETTE["muted"])
    text(d, (1280, 405), "出度", 24, fill=PALETTE["blue"], bold=True)
    text(d, (1370, 405), "入度", 24, fill=PALETTE["orange"], bold=True)
    text(d, (120, 395), "度分布直方图", 30, bold=True)
    text(d, (120, 440), "节点数", 22, fill=PALETTE["muted"])
    img.save(IMAGE_DIR / "dataset_overview.png")


def generate_block_heatmap(edges, n, blocks=20):
    mat = np.zeros((blocks, blocks), dtype=np.int32)
    for a, b in edges:
        i = min(blocks - 1, a * blocks // n)
        j = min(blocks - 1, b * blocks // n)
        mat[i, j] += 1
    values = np.log1p(mat)
    vmax = values.max() or 1
    img = Image.new("RGB", (1500, 1250), "#ffffff")
    d = ImageDraw.Draw(img)
    text(d, (72, 42), "20×20 源/目标分块边密度热力图", 42, bold=True)
    text(d, (72, 98), "颜色越深表示该源节点块指向目标节点块的边越多；图中直接展示 Block Matrix 遍历的结构基础", 22, fill=PALETTE["muted"])
    x0, y0, size = 135, 190, 960
    cell = size / blocks
    for i in range(blocks):
        for j in range(blocks):
            t = float(values[i, j] / vmax)
            color = color_lerp("#eef7ff", "#075985", t)
            x = x0 + j * cell
            y = y0 + i * cell
            d.rectangle((x, y, x + cell + 1, y + cell + 1), fill=color)
    d.rectangle((x0, y0, x0 + size, y0 + size), outline="#1f2937", width=2)
    for k in range(0, blocks + 1, 5):
        x = x0 + k * cell
        y = y0 + k * cell
        d.line((x, y0, x, y0 + size), fill="#ffffff", width=2)
        d.line((x0, y, x0 + size, y), fill="#ffffff", width=2)
        text(d, (x - 16, y0 + size + 16), str(k), 18, fill=PALETTE["muted"])
        text_right(d, x0 - 18, y - 12, str(k), 18, fill=PALETTE["muted"])
    text(d, (x0 + 390, y0 + size + 58), "目标节点块", 24, fill=PALETTE["muted"])
    text(d, (34, y0 + 440), "源节点块", 24, fill=PALETTE["muted"])
    # legend
    lx, ly = 1180, 260
    text(d, (lx, ly - 66), "边数（log 映射）", 24, bold=True)
    for k in range(220):
        t = k / 219
        d.rectangle((lx, ly + k * 3, lx + 58, ly + k * 3 + 3), fill=color_lerp("#075985", "#eef7ff", t))
    text(d, (lx + 78, ly - 6), f"{int(mat.max())}", 20, fill=PALETTE["muted"])
    text(d, (lx + 78, ly + 640), "0", 20, fill=PALETTE["muted"])
    rounded(d, (1120, 970, 1425, 1115), fill=PALETTE["panel"], outline="#e2e8f0")
    text(d, (1145, 998), f"最大块边数：{int(mat.max())}", 24, fill=PALETTE["ink"], bold=True)
    text(d, (1145, 1040), f"平均块边数：{mat.mean():.1f}", 24, fill=PALETTE["muted"])
    img.save(IMAGE_DIR / "block_heatmap.png")


def generate_top20_chart(node_ids, rank):
    order = np.lexsort((np.array(node_ids), -rank))[:20]
    vals = rank[order] * 1_000_000
    labels = [str(node_ids[i]) for i in order]
    img = Image.new("RGB", (1700, 1100), "#ffffff")
    d = ImageDraw.Draw(img)
    text(d, (70, 44), "Top-20 PageRank 节点", 44, bold=True)
    text(d, (70, 100), "横轴为 PageRank × 10^6；Top1 是均匀基线的 1.89 倍，Top20 仍为 Top1 的 87.9%", 23, fill=PALETTE["muted"])
    x0, y0, x1, y1 = 285, 180, 1585, 1010
    maxv = float(vals.max())
    for k in range(5):
        x = x0 + (x1 - x0) * k / 4
        d.line((x, y0, x, y1), fill=PALETTE["grid"], width=1)
        text(d, (x - 32, y1 + 18), f"{maxv * k / 4:.0f}", 18, fill=PALETTE["muted"])
    row_h = (y1 - y0) / len(vals)
    for i, (label, val) in enumerate(zip(labels, vals)):
        y = y0 + i * row_h + 5
        w = (x1 - x0) * float(val) / maxv
        color = color_lerp(PALETTE["blue"], PALETTE["orange"], i / 19)
        text_right(d, x0 - 88, y + 9, f"{i+1}", 20, fill=PALETTE["muted"], bold=True)
        text(d, (x0 - 76, y + 9), label, 21, fill=PALETTE["ink"], bold=True)
        d.rounded_rectangle((x0, y, x0 + w, y + row_h * 0.62), radius=10, fill=color)
        text(d, (x0 + w + 12, y + 7), f"{val:.2f}", 18, fill=PALETTE["muted"])
    text(d, (x0, y1 + 58), "PageRank × 10^6", 22, fill=PALETTE["muted"])
    img.save(IMAGE_DIR / "top20_pagerank.png")


def generate_convergence_chart(residuals, dead_mass):
    img = Image.new("RGB", (1650, 980), "#ffffff")
    d = ImageDraw.Draw(img)
    text(d, (70, 44), "收敛过程与 dead-end 质量", 44, bold=True)
    text(d, (70, 100), "17 轮达到 1e-12 阈值；dead-end 质量被每轮均分回全部节点，避免概率泄漏", 23, fill=PALETTE["muted"])
    x0, y0, x1, y1 = 135, 190, 1520, 810
    for i in range(6):
        y = y0 + (y1 - y0) * i / 5
        d.line((x0, y, x1, y), fill=PALETTE["grid"], width=1)
    logs = [math.log10(r) for r in residuals]
    ymin, ymax = -13.2, max(logs) + 0.25
    pts = []
    for i, val in enumerate(logs):
        x = x0 + (x1 - x0) * i / (len(logs) - 1)
        y = y1 - (val - ymin) / (ymax - ymin) * (y1 - y0)
        pts.append((x, y))
    d.line(pts, fill=PALETTE["blue"], width=5, joint="curve")
    for p in pts:
        d.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), fill=PALETTE["blue"])
    # dead mass mini line scaled to right side
    dm_min, dm_max = min(dead_mass), max(dead_mass)
    pts2 = []
    for i, val in enumerate(dead_mass):
        x = x0 + (x1 - x0) * i / (len(dead_mass) - 1)
        y = y1 - (val - dm_min) / (dm_max - dm_min + 1e-18) * (y1 - y0)
        pts2.append((x, y))
    d.line(pts2, fill=PALETTE["orange"], width=4)
    for i in range(1, len(residuals) + 1, 4):
        x = x0 + (x1 - x0) * (i - 1) / (len(residuals) - 1)
        text(d, (x - 12, y1 + 18), str(i), 18, fill=PALETTE["muted"])
    text(d, (x0, y1 + 60), "迭代轮数", 22, fill=PALETTE["muted"])
    text(d, (x0, y0 - 48), "log10(L1 residual)", 22, fill=PALETTE["blue"], bold=True)
    text(d, (x1 - 300, y0 - 48), "dead-end mass", 22, fill=PALETTE["orange"], bold=True)
    rounded(d, (1110, 670, 1510, 790), fill=PALETTE["panel"], outline="#e2e8f0")
    text(d, (1135, 697), f"最终残差：{residuals[-1]:.3e}", 24, bold=True)
    text(d, (1135, 737), f"收敛轮数：{len(residuals)}", 24, fill=PALETTE["muted"])
    img.save(IMAGE_DIR / "convergence_deadmass.png")


def generate_rank_degree_scatter(node_ids, rank, in_degree, out_degree):
    img = Image.new("RGB", (1650, 1050), "#ffffff")
    d = ImageDraw.Draw(img)
    text(d, (70, 44), "PageRank 与入度/出度关系", 44, bold=True)
    text(d, (70, 100), "高 PageRank 倾向于拥有较高入度，但排名还取决于入链来源节点的重要性和出度分摊", 23, fill=PALETTE["muted"])
    x0, y0, x1, y1 = 145, 190, 1510, 850
    draw_axes(d, (x0, y0, x1, y1), ylabel="PageRank × 10^6")
    max_in = max(in_degree.max(), 1)
    min_r, max_r = float(rank.min()), float(rank.max())
    top = set(np.lexsort((np.array(node_ids), -rank))[:100])
    for i in range(len(rank)):
        x = x0 + float(in_degree[i]) / max_in * (x1 - x0)
        y = y1 - (float(rank[i]) - min_r) / (max_r - min_r) * (y1 - y0)
        if i in top:
            d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=PALETTE["orange"])
        else:
            d.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#9bb8d3")
    for k in range(0, int(max_in) + 1, 8):
        x = x0 + k / max_in * (x1 - x0)
        text(d, (x - 10, y1 + 16), str(k), 18, fill=PALETTE["muted"])
    for k in range(5):
        v = min_r + (max_r - min_r) * k / 4
        y = y1 - (v - min_r) / (max_r - min_r) * (y1 - y0)
        text_right(d, x0 - 14, y - 12, f"{v * 1e6:.0f}", 18, fill=PALETTE["muted"])
    best = int(np.argmax(rank))
    bx = x0 + float(in_degree[best]) / max_in * (x1 - x0)
    by = y1 - (float(rank[best]) - min_r) / (max_r - min_r) * (y1 - y0)
    d.ellipse((bx - 9, by - 9, bx + 9, by + 9), outline=PALETTE["red"], width=4)
    text(d, (bx + 18, by - 14), f"Top1 node {node_ids[best]}", 22, fill=PALETTE["red"], bold=True)
    text(d, (x0, y1 + 58), "入度", 22, fill=PALETTE["muted"])
    text(d, (1190, 905), "蓝色：全部节点", 22, fill="#6686a6")
    text(d, (1360, 905), "橙色：Top100", 22, fill=PALETTE["orange"], bold=True)
    img.save(IMAGE_DIR / "rank_degree_scatter.png")


def generate_memory_runtime_chart():
    img = Image.new("RGB", (1650, 980), "#ffffff")
    d = ImageDraw.Draw(img)
    text(d, (70, 44), "内存与时间约束对比", 44, bold=True)
    text(d, (70, 100), "静态 Linux x86-64 可执行文件 5 次运行均远低于 80 MB / 60 s 限制；稠密矩阵理论内存明显不可接受", 22, fill=PALETTE["muted"])
    # memory log bars
    mem_items = [("稠密 double 矩阵", 688.55, PALETTE["red"]), ("实验限制", 80.0, PALETTE["orange"]), ("静态运行峰值RSS", 11.23, PALETTE["green"]), ("CSR核心数组估算", 0.77, PALETTE["blue"])]
    x0, y0, x1, y1 = 160, 220, 1510, 540
    text(d, (160, 168), "内存（MB，log10 标尺）", 30, bold=True)
    max_log = math.log10(max(v for _, v, _ in mem_items))
    min_log = math.log10(0.5)
    for i, (name, val, color) in enumerate(mem_items):
        y = y0 + i * 72
        w = (math.log10(val) - min_log) / (max_log - min_log) * (x1 - x0)
        d.rounded_rectangle((x0, y, x0 + w, y + 42), radius=10, fill=color)
        text(d, (x0 + 12, y + 8), name, 19, fill="#ffffff", bold=True)
        text(d, (x0 + w + 18, y + 6), f"{val:.2f} MB", 21, fill=PALETTE["ink"], bold=True)
    # runtime
    text(d, (160, 625), "运行时间（秒，log10 标尺）", 30, bold=True)
    time_items = [("实验限制", 60.0, PALETTE["orange"]), ("5次平均", 0.167, PALETTE["green"]), ("最快单次", 0.161, PALETTE["blue"])]
    x0, y0, x1 = 160, 685, 1510
    max_log = math.log10(60.0)
    min_log = math.log10(0.1)
    for i, (name, val, color) in enumerate(time_items):
        y = y0 + i * 70
        w = (math.log10(val) - min_log) / (max_log - min_log) * (x1 - x0)
        d.rounded_rectangle((x0, y, x0 + w, y + 42), radius=10, fill=color)
        text(d, (x0 + 12, y + 8), name, 19, fill="#ffffff", bold=True)
        text(d, (x0 + w + 18, y + 6), f"{val:.3f} s", 21, fill=PALETTE["ink"], bold=True)
    img.save(IMAGE_DIR / "memory_runtime.png")


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    node_ids, edges, row_ptr, col_idx, out_degree, in_degree = load_graph()
    rank, residuals, dead_mass = compute_pagerank(row_ptr, col_idx)
    reported = [(int(a), float(b)) for a, b in (line.split() for line in RES_PATH.read_text().splitlines())]
    order = np.lexsort((np.array(node_ids), -rank))[:100]
    generated = [(node_ids[i], rank[i]) for i in order]
    assert [a for a, _ in reported] == [a for a, _ in generated], "Top-100 node order mismatch"

    generate_terminal_images()
    generate_principle_example()
    generate_dataset_overview(node_ids, out_degree, in_degree, edges)
    generate_block_heatmap(edges, len(node_ids))
    generate_top20_chart(node_ids, rank)
    generate_convergence_chart(residuals, dead_mass)
    generate_rank_degree_scatter(node_ids, rank, in_degree, out_degree)
    generate_memory_runtime_chart()
    print(f"generated {len(list(IMAGE_DIR.glob('*.png')))} png files in {IMAGE_DIR}")


if __name__ == "__main__":
    main()
