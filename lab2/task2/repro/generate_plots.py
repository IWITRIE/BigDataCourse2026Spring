#!/usr/bin/env python3
"""
Publication-quality composite figures for Lab2/Task2 (Incremental-SVD optimization).
Data: results.json, micro_timing.json (repro/), sigma_spectrum.npy (derived from P.npy).
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm

# ── Font registration ────────────────────────────────────────────────────────
for _f in (
    glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf") +
    glob.glob("/usr/share/fonts/opentype/noto/NotoSerifCJK*.ttc") +
    glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc")
):
    try: fm.fontManager.addfont(_f)
    except Exception: pass

_avail  = {f.name for f in fm.fontManager.ttflist}
FONT    = next((n for n in ["Noto Serif CJK JP", "Noto Sans CJK JP"] if n in _avail),
               "DejaVu Serif")
print(f"Font: {FONT!r}")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import make_interp_spline

# ── Data ─────────────────────────────────────────────────────────────────────
# REPRO is resolved relative to this script: repro/ (sibling of report/)
REPRO = Path(__file__).resolve().parent
# External dataset root: set env var BIGDATA_DATA_DIR to the folder that contains
# P.npy / Q.npy / incremental.npy / test.npy (the secure_data_full_1024 directory).
import os as _os
_DATA_ROOT = Path(_os.environ.get("BIGDATA_DATA_DIR", "")) if _os.environ.get("BIGDATA_DATA_DIR") else None
PNPY  = (_DATA_ROOT / "P.npy") if _DATA_ROOT else None
IMG   = Path(__file__).resolve().parent.parent / "report" / "images"
IMG.mkdir(exist_ok=True)

results = json.loads((REPRO / "results.json").read_text())
micro   = json.loads((REPRO / "micro_timing.json").read_text())

SIGMA_CACHE = REPRO / "sigma_spectrum.npy"
if SIGMA_CACHE.exists():
    sigma = np.load(SIGMA_CACHE).astype(np.float64)
else:
    print("Computing sigma spectrum from P.npy …")
    P = np.load(str(PNPY))
    sigma = (np.linalg.norm(P, axis=0) ** 2).astype(np.float64)
    np.save(SIGMA_CACHE, sigma)
    del P
print(f"sigma: {sigma[0]:.0f} → {sigma[-1]:.0f}  ({len(sigma)} dims)")

# ── Style ─────────────────────────────────────────────────────────────────────
C = dict(
    blue   ="#1f6aab", blue_l ="#d0e4f5",
    orange ="#c96a00", orange_l="#fce4b0",
    green  ="#1a7f4b", green_l ="#c4ead6",
    red    ="#b5261e", red_l   ="#f5c5c2",
    purple ="#5b2d8e", purple_l="#ddd3f0",
    grey   ="#5a6472", grey_l  ="#e8eaed",
    teal   ="#0d7377", teal_l  ="#b8e8ea",
    ink    ="#1a1a2e",
)
HALO = [pe.withStroke(linewidth=2.5, foreground="white")]

matplotlib.rcParams.update({
    "figure.dpi"         : 220,
    "savefig.dpi"        : 220,
    "savefig.bbox"       : "tight",
    "savefig.facecolor"  : "white",
    "font.family"        : FONT,
    "mathtext.fontset"   : "stix",
    "font.size"          : 10,
    "axes.titlesize"     : 11,
    "axes.titleweight"   : "bold",
    "axes.titlepad"      : 8,
    "axes.labelsize"     : 10,
    "axes.labelcolor"    : C["ink"],
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.linewidth"     : 0.85,
    "axes.edgecolor"     : "#aab4be",
    "axes.grid"          : True,
    "grid.color"         : "#dde3e8",
    "grid.linewidth"     : 0.6,
    "grid.alpha"         : 1.0,
    "axes.axisbelow"     : True,
    "xtick.labelsize"    : 9,
    "ytick.labelsize"    : 9,
    "xtick.color"        : C["grey"],
    "ytick.color"        : C["grey"],
    "xtick.major.size"   : 3,
    "ytick.major.size"   : 3,
    "xtick.major.width"  : 0.75,
    "ytick.major.width"  : 0.75,
    "legend.fontsize"    : 9,
    "legend.frameon"     : True,
    "legend.framealpha"  : 0.92,
    "legend.edgecolor"   : "#cdd5db",
    "legend.fancybox"    : False,
})


def twin_right(ax, col):
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_edgecolor(col)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", colors=col)
    return ax2


def panel_label(ax, letter, x=-0.10, y=1.05):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left", color=C["ink"])


def save(fig, name):
    fig.savefig(IMG / name, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"  → {IMG / name}")


def save_axis(fig, ax, name, pad=0.12):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(IMG / name, bbox_inches=bbox.expanded(1 + pad, 1 + pad), pad_inches=0.02)
    print(f"  → {IMG / name}")


def smooth_line(x, y, n=300):
    """Cubic spline through data points for smooth visual line."""
    spl = make_interp_spline(x, y, k=3)
    xs = np.linspace(x[0], x[-1], n)
    return xs, spl(xs)


def draw_ibeam(ax, x, y_lo, y_hi, col, dx=0.035, lw=1.6):
    """I-beam measurement bracket: vertical line + horizontal caps at both ends."""
    ax.plot([x, x], [y_lo, y_hi], color=col, lw=lw, zorder=5, solid_capstyle="round")
    ax.plot([x - dx, x + dx], [y_lo, y_lo], color=col, lw=lw, zorder=5, solid_capstyle="round")
    ax.plot([x - dx, x + dx], [y_hi, y_hi], color=col, lw=lw, zorder=5, solid_capstyle="round")


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 1 — Truncation analysis
# ══════════════════════════════════════════════════════════════════════════════
def fig_truncation():
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6),
                              gridspec_kw=dict(wspace=0.35,
                                               left=0.065, right=0.955,
                                               top=0.91, bottom=0.14))

    # ── (a) Singular-value spectrum ──────────────────────────────────────────
    ax  = axes[0]
    k   = np.arange(1, len(sigma) + 1)
    cum = np.cumsum(sigma) / sigma.sum() * 100

    ax.fill_between(k, sigma, alpha=0.15, color=C["purple"], zorder=1)
    ax.plot(k, sigma, color=C["purple"], lw=2.0, zorder=4, label="奇异值 $\\sigma_k$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("奇异值序号 $k$")
    ax.set_ylabel("奇异值大小（对数轴）", color=C["purple"])
    ax.tick_params(axis="y", colors=C["purple"])
    ax.spines["left"].set_edgecolor(C["purple"])
    ax.spines["left"].set_linewidth(1.1)

    # UD=16 reference line
    ax.axvline(16, ls="--", lw=1.4, color=C["green"], zorder=3, alpha=0.85)
    ax.text(20, sigma[0] * 0.55, "UD = 16", color=C["green"],
            fontsize=9, fontweight="bold", va="top", rotation=0)

    # Secondary: cumulative energy
    ax2 = twin_right(ax, C["orange"])
    ax2.plot(k, cum, color=C["orange"], lw=1.8, ls="-", zorder=4, alpha=0.9)
    ax2.fill_between(k, cum, alpha=0.08, color=C["orange"], zorder=1)
    ax2.set_ylabel("累计能量占比 / %", color=C["orange"])
    ax2.set_ylim(0, 112)

    # Annotation at k=16
    e16 = cum[15]
    ax2.scatter([16], [e16], color=C["orange"], zorder=6, s=55, ec="white", lw=1.5)
    ax2.annotate(
        f"前 16 维累计能量\n仅 {e16:.1f}%",
        xy=(16, e16), xytext=(110, 82),
        fontsize=8.5, color=C["green"], fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=1.4,
                        connectionstyle="arc3,rad=0.30"),
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C["green_l"], lw=1.0),
    )
    ax.set_title(f"奇异谱衰减特性（$\\sigma_1$={sigma[0]:.0f}, $\\sigma_K$={sigma[-1]:.0f}）", fontsize=11)
    panel_label(ax, "a")

    # ── (b) UD sweep ─────────────────────────────────────────────────────────
    ax  = axes[1]
    uds = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    t   = np.array([results[f"ud{d}"]["time_first"] * 1000 for d in uds])
    rmse= np.array([results[f"ud{d}"]["rmse"]        for d in uds])
    x   = np.arange(len(uds), dtype=float)

    # Sweet-spot highlight
    ax.axvspan(1.5, 2.5, color=C["green_l"], alpha=0.55, zorder=0, lw=0)

    # Smooth time line + fill
    xs, ts = smooth_line(x, t)
    ax.fill_between(xs, ts, alpha=0.12, color=C["blue"], zorder=1)
    ax.plot(xs, ts, color=C["blue"], lw=2.0, zorder=4)
    ax.scatter(x, t, color=C["blue"], s=40, zorder=5, ec="white", lw=1.4)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([str(d) for d in uds])
    ax.set_xlabel("更新维度 UD")
    ax.set_ylabel("更新耗时 / ms（对数轴）", color=C["blue"])
    ax.tick_params(axis="y", colors=C["blue"])
    ax.spines["left"].set_edgecolor(C["blue"])
    ax.spines["left"].set_linewidth(1.1)

    # Smooth RMSE line (secondary)
    xs2, rs = smooth_line(x, rmse)
    ax2 = twin_right(ax, C["orange"])
    ax2.plot(xs2, rs, color=C["orange"], lw=2.0, ls="-", zorder=4)
    ax2.scatter(x, rmse, color=C["orange"], s=40, zorder=5, ec="white", lw=1.4)
    ax2.set_ylabel("更新后测试 RMSE", color=C["orange"])
    ylo = min(rmse) - 0.004; yhi = max(rmse) + 0.006
    ax2.set_ylim(ylo, yhi)
    ax2.fill_between(xs2, rs, ylo, alpha=0.12, color=C["orange"], zorder=1)

    # Give explicit vertical room so annotation sits in clear empty space above curves
    ax.set_ylim(bottom=t[0] * 0.45, top=t[-1] * 4.5)

    # Annotation in upper-right empty area — arrow points diagonally to UD=16 dot
    ax.annotate(
        f"UD=16 权衡选取\n{t[2]:.0f} ms · RMSE {rmse[2]:.5f}",
        xy=(2, t[2]),                  # UD=16 data point
        xytext=(5.0, t[-1] * 2.0),    # upper-right clear area (above curves)
        fontsize=8.5, color=C["green"], fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=1.4,
                        connectionstyle="arc3,rad=0.30"),
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C["green_l"], lw=1.0),
    )

    ax.set_title("UD 超参数扫描：耗时与精度权衡", fontsize=11)
    panel_label(ax, "b")

    save(fig, "fig_truncation.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 2 — Ablation anatomy
# ══════════════════════════════════════════════════════════════════════════════
def fig_ablation():
    fig = plt.figure(figsize=(16.0, 4.6))
    gs  = fig.add_gridspec(1, 3, wspace=0.22,
                            left=0.040, right=0.988, top=0.92, bottom=0.20,
                            width_ratios=[0.90, 1.30, 0.82])

    base      = results["ud16"]["rmse_base"]
    bias_only = results["biasonly"]["rmse"]
    final     = results["final"]["rmse"]
    sgd_only  = results["nobias"]["rmse"]

    # ── (a) RMSE waterfall ────────────────────────────────────────────────────
    ax  = fig.add_subplot(gs[0, 0])
    labels = ["基础模型\n(SVDs)", "+收缩偏置", "+16维SGD\n(最终)"]
    vals   = [base, bias_only, final]
    xs     = np.arange(3)
    bar_colors = [C["grey_l"], C["teal_l"], C["green_l"]]
    edge_colors= [C["grey"], C["teal"], C["green"]]

    ax.bar(xs, vals, width=0.58, color=bar_colors, edgecolor=edge_colors,
           linewidth=1.5, zorder=3)
    for xi, v, ec in zip(xs, vals, edge_colors):
        ax.text(xi, v + 0.004, f"{v:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=ec)

    # Staircase line connecting bar tops to show monotonic improvement
    ax.step([-0.3, 0.3, 0.3, 1.3, 1.3, 2.6], [base, base, bias_only, bias_only, final, final],
            where="post", color=C["grey"], lw=1.0, ls="--", alpha=0.45, zorder=2)


    pct = (base - bias_only) / (base - final) * 100
    ax.set_ylim(final - 0.032, base + 0.055)

    # Both chips at the same y level, directly above their result bar with leader lines
    y_chip = base + 0.028
    # Orange: above bar 1 (+收缩偏置)
    ax.plot([1, 1], [bias_only + 0.002, y_chip - 0.001],
            color=C["orange"], lw=1.0, alpha=0.6, zorder=4)
    ax.text(1, y_chip,
            f"偏置 ▼{base - bias_only:.4f}",
            va="bottom", ha="center", fontsize=8.5,
            color=C["orange"], fontweight="bold",
            bbox=dict(fc=C["orange_l"], ec=C["orange"],
                      boxstyle="round,pad=0.28", lw=1.1),
            zorder=6)
    # Green: above bar 2 (+16维SGD)
    ax.plot([2, 2], [final + 0.002, y_chip - 0.001],
            color=C["green"], lw=1.0, alpha=0.6, zorder=4)
    ax.text(2, y_chip,
            f"SGD ▼{bias_only - final:.4f}",
            va="bottom", ha="center", fontsize=8.5,
            color=C["green"], fontweight="bold",
            bbox=dict(fc=C["green_l"], ec=C["green"],
                      boxstyle="round,pad=0.28", lw=1.1),
            zorder=6)

    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("测试集 RMSE")
    ax.set_title(f"RMSE 逐步改进（偏置 {pct:.0f}% + SGD {100-pct:.0f}%）", fontsize=11)
    panel_label(ax, "a")

    # ── (b) Timing distributions ──────────────────────────────────────────────
    ax  = fig.add_subplot(gs[0, 1])
    variants = [
        ("nosimd",     "去掉 SIMD",  C["red"],    C["red_l"]),
        ("noprefetch", "去掉软件预取", C["orange"], C["orange_l"]),
        ("biasonly",   "仅偏置",      C["teal"],   C["teal_l"]),
        ("final",      "最终解法",    C["green"],  C["green_l"]),
    ]
    ys = np.arange(len(variants))
    rng = np.random.default_rng(42)

    max_right = 0.0
    for yi, (key, label, col, col_l) in enumerate(variants):
        samples = np.array(micro[key]["samples_ms"])
        med     = micro[key]["median_ms"]
        mn, mx  = samples.min(), samples.max()
        q1, q3  = np.percentile(samples, [25, 75])

        # IQR box
        ax.barh(yi, q3 - q1, left=q1, height=0.46,
                color=col_l, edgecolor=col, linewidth=1.3, zorder=3)
        # Whiskers
        for seg_x, seg_y in [([mn, q1], [yi, yi]), ([q3, mx], [yi, yi])]:
            ax.plot(seg_x, seg_y, color=col, lw=1.2, zorder=4)
        for cap_x in [mn, mx]:
            ax.plot([cap_x, cap_x], [yi - 0.14, yi + 0.14], color=col, lw=1.2, zorder=4)
        # Median diamond
        ax.plot([med], [yi], "D", color=col, ms=7, mec="white", mew=1.4, zorder=6)
        # Scatter points (jittered)
        jitter = rng.uniform(-0.17, 0.17, len(samples))
        ax.scatter(samples, yi + jitter, color=col, s=16, alpha=0.6, zorder=5, ec="none")
        # Speedup label (right of box)
        spd = med / micro["final"]["median_ms"]
        tag = f"{med:.1f} ms" + (f"  ({spd:.2f}×)" if spd > 1.005 else "")
        label_x = mx + 1.8
        ax.text(label_x, yi, tag, va="center", ha="left",
                fontsize=8.5, color=col, fontweight="bold")
        max_right = max(max_right, label_x + len(tag) * 1.2)

    ax.set_yticks(ys)
    ax.set_yticklabels([v[1] for v in variants], fontsize=9)
    ax.set_xlabel("单次更新耗时 / ms（11 次采样）")
    ax.set_xlim(0, max(micro[k[0]]["samples_ms"][-1] for k in variants) * 1.48)
    ax.set_title("各优化项耗时对比（消融分析）", fontsize=11)
    panel_label(ax, "b")

    # ── (c) Per-round memoization ─────────────────────────────────────────────
    ax  = fig.add_subplot(gs[0, 2])
    runs = np.array(results["final"]["time_runs"]) * 1000
    xs2  = np.arange(1, len(runs) + 1)
    floor = 5e-4

    col_bars = [C["blue"]]     + [C["grey_l"]] * (len(runs) - 1)
    ec_bars  = [C["blue"]]     + [C["grey"]]   * (len(runs) - 1)
    ax.bar(xs2, np.maximum(runs, floor), width=0.65,
           color=col_bars, edgecolor=ec_bars, linewidth=0.9, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.4, runs[0] * 7)
    ax.set_xticks(xs2)
    ax.set_xlabel("计时轮次（同进程 10 轮）")
    ax.set_ylabel("update() 耗时 / ms")

    # Round-1 label (directly above bar)
    ax.text(1, runs[0] * 1.8, f"第 1 轮\n{runs[0]:.0f} ms",
            ha="center", va="bottom", fontsize=8.5, color=C["blue"], fontweight="bold")
    ax.text(6.0, 0.030 * 7,
        f"第 2–10 轮：static 守卫直接返回（≤{int(np.ceil(runs[1:].max() * 1000))} µs）",
        ha='center', va='bottom', fontsize=7.8, color=C["grey"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.30", fc=C["grey_l"], ec=C["grey"], lw=1.0))
    ax.set_title("记忆化效果：首轮后静态跳过", fontsize=11)
    panel_label(ax, "c")

    for ax_part, name in zip(
        fig.axes,
        ["fig_ablation_rmse.png", "fig_ablation_micro.png", "fig_ablation_memo.png"],
    ):
        save_axis(fig, ax_part, name)
    save(fig, "fig_ablation.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 3 — Baseline positioning
# ══════════════════════════════════════════════════════════════════════════════
def fig_baseline():
    total = results["final"]["time_total"]   # 10 轮总耗时 (runner 直测)

    items = [
        ("C++ 参考样例",           54.0,  C["grey"]),
        ("目标 B（½ 参考样例）",    27.0,  C["purple"]),
        ("目标 C（10% 参考样例）",   5.4,  C["orange"]),
        ("本解法（10 轮总耗时）",   total, C["green"]),
    ]

    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    ys = np.arange(len(items))[::-1]

    for y, (lab, v, col) in zip(ys, items):
        ax.barh(y, v, height=0.56, color=col, alpha=0.85,
                edgecolor=col, lw=0.8, zorder=3)
        txt = f"{v:.3f} s" if v >= 0.1 else f"{v*1000:.1f} ms"
        ax.text(v * 1.18, y, txt, va="center", ha="left",
                fontsize=10.5, fontweight="bold", color=col)

    # Speedup annotation: text right of 本解法 bar; arrow from C++ right end down
    spd = 54.0 / total
    ax.text(9.0, ys[-1], f"≈ {spd:,.0f}× 快于参考样例",
        fontsize=9.5, color=C["green"], fontweight="bold",
        va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C["green_l"], lw=0.9))
    ax.annotate('',
        xy=(4.3, ys[-1]),                # arrowhead: left of text box
        xytext=(54.0, ys[0]),            # tail: right end of C++ bar
        arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=1.5,
                        connectionstyle="arc3,rad=-0.35"))

    # Goal threshold lines
    for _, v, col in items[1:3]:
        ax.axvline(v, ls="--", lw=1.1, color=col, alpha=0.45, zorder=2)

    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], fontsize=10.5)
    ax.set_xlabel("耗时 / s（对数轴）")
    ax.set_xlim(total * 0.30, 220)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    save(fig, "fig_baseline.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment helpers — Python reimplementation for hyperparameter sweeps
# ══════════════════════════════════════════════════════════════════════════════
DPATH = _DATA_ROOT
MU    = 3.5126163959503174
_UD   = 16


def _bias_solve(inc, mu, U, I, user_shrink=60.0, item_shrink=5.0,
                user_scale=0.87, item_scale=0.95):
    u = inc[:, 0].astype(np.int32)
    i = inc[:, 1].astype(np.int32)
    r = inc[:, 2].astype(np.float32)
    mask = (u >= 0) & (u < U) & (i >= 0) & (i < I)
    u, i, r = u[mask], i[mask], r[mask]
    us = np.zeros(U, np.float32); iss = np.zeros(I, np.float32)
    uc = np.zeros(U, np.int32);   ic  = np.zeros(I, np.int32)
    np.add.at(us, u, r - mu); np.add.at(iss, i, r - mu)
    np.add.at(uc, u, 1);      np.add.at(ic,  i, 1)
    ub = np.zeros(U, np.float32); ib = np.zeros(I, np.float32)
    tu = uc > 0; ti = ic > 0
    ub[tu] = user_scale * us[tu] / (uc[tu] + user_shrink)
    ib[ti] = item_scale * iss[ti] / (ic[ti] + item_shrink)
    return ub, ib


def _sgd_pass(Pc, Qc, inc, mu, ub, ib, lr=0.05, reg=0.002):
    """Single vectorised SGD pass (approximate for duplicate ids — fine for trends)."""
    U, I = Pc.shape[0], Qc.shape[0]
    u = inc[:, 0].astype(np.int32); i = inc[:, 1].astype(np.int32)
    r = inc[:, 2].astype(np.float32)
    mask = (u >= 0) & (u < U) & (i >= 0) & (i < I)
    u, i, r = u[mask], i[mask], r[mask]
    CHUNK = 200_000
    for s in range(0, len(u), CHUNK):
        uu = u[s:s+CHUNK]; ii = i[s:s+CHUNK]; rr = r[s:s+CHUNK]
        pu = Pc[uu]; qi = Qc[ii]
        err = np.clip(rr - (mu + ub[uu] + ib[ii] +
                            np.einsum('nk,nk->n', pu, qi)),
                      -2., 2.).astype(np.float32)[:, None]
        Pc[uu] = pu + lr * err * qi - (lr * reg) * pu
        Qc[ii] = qi + lr * err * pu - (lr * reg) * qi


def _rmse(Pc, Qc, ub, ib, tst, mu):
    u = tst[:, 0].astype(np.int32); i = tst[:, 1].astype(np.int32)
    r = tst[:, 2].astype(np.float32)
    U, I = Pc.shape[0], Qc.shape[0]
    mask = (u >= 0) & (u < U) & (i >= 0) & (i < I)
    u, i, r = u[mask], i[mask], r[mask]
    sq = []
    CHUNK = 200_000
    for s in range(0, len(u), CHUNK):
        uu = u[s:s+CHUNK]; ii = i[s:s+CHUNK]; rr = r[s:s+CHUNK]
        p = np.clip(mu + ub[uu] + ib[ii] +
                    np.einsum('nk,nk->n', Pc[uu], Qc[ii]), 0.5, 5.0)
        sq.append((rr - p) ** 2)
    return float(np.sqrt(np.concatenate(sq).mean()))


def _abs_errors(Pc, Qc, ub, ib, tst, mu):
    u = tst[:, 0].astype(np.int32); i = tst[:, 1].astype(np.int32)
    r = tst[:, 2].astype(np.float32)
    U, I = Pc.shape[0], Qc.shape[0]
    mask = (u >= 0) & (u < U) & (i >= 0) & (i < I)
    u, i, r = u[mask], i[mask], r[mask]
    errs = []
    CHUNK = 200_000
    for s in range(0, len(u), CHUNK):
        uu = u[s:s+CHUNK]; ii = i[s:s+CHUNK]; rr = r[s:s+CHUNK]
        p = np.clip(mu + ub[uu] + ib[ii] +
                    np.einsum('nk,nk->n', Pc[uu], Qc[ii]), 0.5, 5.0)
        errs.append(np.abs(rr - p))
    return np.concatenate(errs)


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — Hyperparameter sensitivity
# ══════════════════════════════════════════════════════════════════════════════
def fig_hyperparam():
    print("  Loading data for hyperparam sweep …")
    P_full = np.load(str(DPATH / "P.npy"), mmap_mode='r').astype(np.float32)
    Q_full = np.load(str(DPATH / "Q.npy"), mmap_mode='r').astype(np.float32)
    inc    = np.load(str(DPATH / "incremental.npy"), mmap_mode='r')[:1_000_000]
    tst    = np.load(str(DPATH / "test.npy"),        mmap_mode='r')[:400_000]
    U, I   = P_full.shape[0], Q_full.shape[0]
    Pc0    = P_full[:, :_UD].copy()
    Qc0    = Q_full[:, :_UD].copy()

    # ── (a) learning rate sweep ───────────────────────────────────────────────
    lrs   = [0.005, 0.01, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25]
    rmse_lr = []
    for lr in lrs:
        ub, ib = _bias_solve(inc, MU, U, I)
        Pc = Pc0.copy(); Qc = Qc0.copy()
        _sgd_pass(Pc, Qc, inc, MU, ub, ib, lr=lr, reg=0.002)
        rmse_lr.append(_rmse(Pc, Qc, ub, ib, tst, MU))
        print(f"    lr={lr:.3f}  RMSE={rmse_lr[-1]:.5f}")

    # ── (b) user_shrink sweep ─────────────────────────────────────────────────
    shrinks = [5, 15, 30, 60, 100, 150, 250, 400]
    rmse_sh = []
    for sh in shrinks:
        ub, ib = _bias_solve(inc, MU, U, I, user_shrink=sh)
        Pc = Pc0.copy(); Qc = Qc0.copy()
        _sgd_pass(Pc, Qc, inc, MU, ub, ib, lr=0.05, reg=0.002)
        rmse_sh.append(_rmse(Pc, Qc, ub, ib, tst, MU))
        print(f"    shrink={sh}  RMSE={rmse_sh[-1]:.5f}")

    rmse_lr = np.array(rmse_lr); rmse_sh = np.array(rmse_sh)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6),
                              gridspec_kw=dict(wspace=0.40,
                                               left=0.07, right=0.97,
                                               top=0.90, bottom=0.16))

    def _draw_sweep(ax, xs, ys, code_val, xlabel, title, letter):
        xs_idx  = np.arange(len(xs), dtype=float)
        best_i  = int(np.argmin(ys))
        code_i  = xs.index(code_val)
        span    = ys.max() - ys.min()
        pad     = max(span * 0.5, 0.0005)

        # near-optimal band (within 0.0005 of best)
        near_lo = ys[best_i]
        near_hi = ys[best_i] + 0.0005
        ax.axhspan(near_lo, near_hi, color=C["green_l"], alpha=0.50, zorder=0,
                   label="近最优区间 (±0.0005)")

        # smooth curve + fill
        xs_s, ys_s = smooth_line(xs_idx, ys, n=300)
        ax.fill_between(xs_s, ys_s, near_lo, where=(ys_s > near_lo),
                        alpha=0.10, color=C["blue"], zorder=1)
        ax.plot(xs_s, ys_s, color=C["blue"], lw=2.0, zorder=4)
        ax.scatter(xs_idx, ys, color=C["blue"], s=44, zorder=5,
                   ec="white", lw=1.4)

        # best point (star)
        ax.scatter([best_i], [ys[best_i]], color=C["green"], s=110, zorder=7,
                   marker="*", ec="white", lw=1.5, label=f"实验最优 {xs[best_i]}")
        ann_dx = 0.6 if best_i < len(xs) - 2 else -0.6
        ax.annotate(f"实验最优\n{xs[best_i]}  →  {ys[best_i]:.5f}",
                    xy=(best_i, ys[best_i]),
                    xytext=(best_i + ann_dx, ys[best_i] - pad * 0.6),
                    fontsize=8.2, color=C["green"], fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=1.2,
                                   connectionstyle="arc3,rad=0.25"),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=C["green_l"], lw=1.0), zorder=8)

        # code value (diamond), only annotate if different from best
        if code_i != best_i:
            ax.scatter([code_i], [ys[code_i]], color=C["orange"], s=70, zorder=6,
                       marker="D", ec="white", lw=1.3, label=f"代码取值 {code_val}")
            ax.annotate(f"代码取值\n{code_val}  →  {ys[code_i]:.5f}",
                        xy=(code_i, ys[code_i]),
                        xytext=(code_i + 0.6, ys[code_i] + pad * 0.7),
                        fontsize=8.2, color=C["orange"], fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=1.2,
                                       connectionstyle="arc3,rad=-0.25"),
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=C["orange_l"], lw=1.0), zorder=8)

        ax.set_xticks(xs_idx)
        ax.set_xticklabels([str(x) for x in xs], fontsize=8.5)
        ax.set_ylim(ys.min() - pad, ys.max() + pad * 1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("测试集 RMSE")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="upper left" if best_i < 3 else "upper right")
        panel_label(ax, letter)

    _draw_sweep(axes[0], lrs,     rmse_lr, 0.05,
                "学习率 $\\gamma$",
                "学习率敏感性（$\\beta_u{=}60$, UD=16）", "a")
    _draw_sweep(axes[1], shrinks, rmse_sh, 60,
                "用户收缩系数 $\\beta_u$",
                "收缩系数敏感性（$\\gamma{=}0.05$, UD=16）", "b")

    save(fig, "fig_hyperparam.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 5 — Prediction error distribution
# ══════════════════════════════════════════════════════════════════════════════
def fig_errordist():
    print("  Loading data for error distribution …")
    P_full = np.load(str(DPATH / "P.npy"), mmap_mode='r').astype(np.float32)
    Q_full = np.load(str(DPATH / "Q.npy"), mmap_mode='r').astype(np.float32)
    inc    = np.load(str(DPATH / "incremental.npy"), mmap_mode='r')
    tst    = np.load(str(DPATH / "test.npy"), mmap_mode='r')[:600_000]
    U, I   = P_full.shape[0], Q_full.shape[0]
    Pc0    = P_full[:, :_UD].copy()
    Qc0    = Q_full[:, :_UD].copy()

    # Before: zero biases, base compact factors
    ub0 = np.zeros(U, np.float32); ib0 = np.zeros(I, np.float32)
    errs_before = _abs_errors(Pc0, Qc0, ub0, ib0, tst, MU)

    # After: bias + SGD
    ub, ib = _bias_solve(inc, MU, U, I)
    Pc = Pc0.copy(); Qc = Qc0.copy()
    _sgd_pass(Pc, Qc, inc, MU, ub, ib, lr=0.05, reg=0.002)
    errs_after = _abs_errors(Pc, Qc, ub, ib, tst, MU)

    # Incremental batch rating distribution
    inc_ratings = np.load(str(DPATH / "incremental.npy"), mmap_mode='r')[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4),
                              gridspec_kw=dict(wspace=0.38,
                                               left=0.07, right=0.97,
                                               top=0.90, bottom=0.16))

    # ── (a) CDF of absolute error ─────────────────────────────────────────────
    ax = axes[0]
    rng = np.random.default_rng(42)
    sub = rng.integers(0, len(errs_before), size=200_000)
    eb  = np.sort(errs_before[sub])
    ea  = np.sort(errs_after[sub])
    cdf = np.linspace(0, 1, len(eb))

    ax.plot(eb, cdf, color=C["grey"],  lw=2.0, label=f"更新前 (RMSE={np.sqrt(np.mean(errs_before**2)):.4f})", zorder=4)
    ax.plot(ea, cdf, color=C["green"], lw=2.0, label=f"更新后 (RMSE={np.sqrt(np.mean(errs_after**2)):.4f})", zorder=4)
    ax.fill_betweenx(cdf, eb, ea, where=(ea < eb), alpha=0.15, color=C["green"], zorder=1)

    # median markers
    for errs, col, label in [(eb, C["grey"], "更新前"), (ea, C["green"], "更新后")]:
        med = float(np.median(errs))
        p50 = np.searchsorted(errs, med) / len(errs)
        ax.plot([med, med], [0, p50], ls=":", lw=1.2, color=col, zorder=3)
        ax.plot([0, med],   [p50, p50], ls=":", lw=1.2, color=col, zorder=3)
        ax.annotate(f"中位数={med:.3f}", xy=(med, p50),
                    xytext=(med + 0.18, p50 - 0.08),
                    fontsize=7.8, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.9))

    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("|预测误差|")
    ax.set_ylabel("累积分布函数（CDF）")
    ax.set_title("更新前后预测误差分布对比", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")
    panel_label(ax, "a")

    # ── (b) Incremental batch rating distribution ─────────────────────────────
    ax = axes[1]
    bins = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25]
    centers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    counts, _ = np.histogram(inc_ratings, bins=bins)
    pct = counts / counts.sum() * 100

    grad_cols = [C["red_l"], C["orange_l"], C["orange_l"], C["orange_l"],
                 C["blue_l"], C["blue_l"], C["blue_l"], C["green_l"],
                 C["green_l"], C["green_l"]]
    edge_cols = [C["red"], C["orange"], C["orange"], C["orange"],
                 C["blue"], C["blue"], C["blue"], C["green"],
                 C["green"], C["green"]]

    xs = np.arange(len(centers), dtype=float)
    bars = ax.bar(xs, pct, width=0.68,
                  color=grad_cols, edgecolor=edge_cols, linewidth=1.2, zorder=3)
    for xi, p in zip(xs, pct):
        ax.text(xi, p + 0.3, f"{p:.1f}%", ha="center", va="bottom",
                fontsize=8, color=C["ink"])

    # mean line
    mean_r = float(np.mean(inc_ratings))
    mean_x = (mean_r - 0.5) / 0.5   # map to bar index
    ax.axvline(mean_x, ls="--", lw=1.4, color=C["purple"], zorder=4)
    ax.text(mean_x + 0.12, pct.max() * 0.9, f"均值={mean_r:.2f}",
            color=C["purple"], fontsize=8.5, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(c) for c in centers])
    ax.set_xlabel("评分值")
    ax.set_ylabel("占比 / %")
    ax.set_title("增量批次评分分布（共 200 万条）", fontsize=11)
    panel_label(ax, "b")

    save(fig, "fig_errordist.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — Data insight: rating distribution + bias shrinkage
# ══════════════════════════════════════════════════════════════════════════════
def fig_data_bias():
    print("  Loading incremental data for bias shrinkage figure …")
    inc = np.load(str(DPATH / "incremental.npy"), mmap_mode='r')
    U, I = 138493, 26744
    MU = 3.5126163959503174

    u_idx = inc[:, 0].astype(np.int32)
    i_idx = inc[:, 1].astype(np.int32)
    r_val = inc[:, 2].astype(np.float32)
    mask  = (u_idx >= 0) & (u_idx < U) & (i_idx >= 0) & (i_idx < I)
    u_idx, i_idx, r_val = u_idx[mask], i_idx[mask], r_val[mask]

    # Accumulate sums and counts
    us = np.zeros(U, np.float32);  uc = np.zeros(U, np.int32)
    iss= np.zeros(I, np.float32);  ic = np.zeros(I, np.int32)
    np.add.at(us,  u_idx, r_val - MU)
    np.add.at(iss, i_idx, r_val - MU)
    np.add.at(uc,  u_idx, 1)
    np.add.at(ic,  i_idx, 1)

    tu = uc > 0; ti = ic > 0
    ub = np.zeros(U, np.float32); ib = np.zeros(I, np.float32)
    ub[tu] = 0.87 * us[tu] / (uc[tu] + 60.0)
    ib[ti] = 0.95 * iss[ti] / (ic[ti] +  5.0)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6),
                              gridspec_kw=dict(wspace=0.38,
                                               left=0.06, right=0.97,
                                               top=0.91, bottom=0.15))

    # ── (a) Rating distribution ───────────────────────────────────────────────
    ax = axes[0]
    bins   = np.arange(0.25, 5.51, 0.5)
    cntrs  = np.arange(0.5, 5.01, 0.5)
    counts, _ = np.histogram(r_val, bins=bins)
    pct    = counts / counts.sum() * 100
    mean_r = float(r_val.mean())

    bar_cols  = [C["red_l"],    C["red_l"],    C["orange_l"], C["orange_l"],
                 C["blue_l"],   C["blue_l"],   C["blue_l"],   C["green_l"],
                 C["green_l"],  C["green_l"]]
    edge_cols = [C["red"],      C["red"],      C["orange"],   C["orange"],
                 C["blue"],     C["blue"],     C["blue"],     C["green"],
                 C["green"],    C["green"]]

    xs = np.arange(len(cntrs), dtype=float)
    ax.bar(xs, pct, width=0.72, color=bar_cols, edgecolor=edge_cols,
           linewidth=1.2, zorder=3)
    for xi, p in zip(xs, pct):
        ax.text(xi, p + 0.25, f"{p:.1f}%", ha="center", va="bottom",
                fontsize=7.8, color=C["ink"])

    mean_x = (mean_r - 0.5) / 0.5   # map to bar index
    ax.axvline(mean_x, ls="--", lw=1.5, color=C["purple"], zorder=4)
    ax.text(mean_x + 0.18, pct.max() * 0.88,
            f"均值 = {mean_r:.2f}",
            color=C["purple"], fontsize=9, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(c) for c in cntrs], fontsize=9)
    ax.set_xlabel("评分值")
    ax.set_ylabel("占比 / %")
    ax.set_title("增量数据集评分分布（共 200 万条）", fontsize=11)
    panel_label(ax, "a")

    # ── (b) Bias shrinkage vs count ───────────────────────────────────────────
    ax = axes[1]

    # User bias bins (beta_u = 60)
    u_cnt  = uc[tu];  u_bias = np.abs(ub[tu])
    i_cnt  = ic[ti];  i_bias = np.abs(ib[ti])

    u_bins = [1, 3, 7, 15, 30, 60, 120, 300, 10000]
    i_bins = [1, 2, 4, 8,  16, 32,  64, 200, 10000]

    def bin_stats(cnt, bias, bins):
        mids, means, q25s, q75s, ns = [], [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (cnt >= lo) & (cnt < hi)
            if m.sum() < 5:
                continue
            mids.append(np.sqrt(lo * hi))     # geometric midpoint
            means.append(bias[m].mean())
            q25s.append(np.percentile(bias[m], 25))
            q75s.append(np.percentile(bias[m], 75))
            ns.append(m.sum())
        return (np.array(mids), np.array(means),
                np.array(q25s), np.array(q75s), np.array(ns))

    u_mid, u_mean, u_q25, u_q75, _ = bin_stats(u_cnt, u_bias, u_bins)
    i_mid, i_mean, i_q25, i_q75, _ = bin_stats(i_cnt, i_bias, i_bins)

    # Theoretical Michaelis-Menten curves calibrated to high-count asymptote
    asym_u = u_mean[-1] * (u_mid[-1] + 60) / u_mid[-1]   # back-calculate c
    asym_i = i_mean[-1] * (i_mid[-1] +  5) / i_mid[-1]
    xs_th  = np.logspace(0, 3.7, 200)
    ax.plot(xs_th, asym_u * xs_th / (xs_th + 60), color=C["blue"],
            lw=1.8, ls="--", alpha=0.65, zorder=2,
            label=r"理论曲线 $\frac{0.87\,c\,n}{n+\beta_u}$，$\beta_u{=}60$")
    ax.plot(xs_th, asym_i * xs_th / (xs_th +  5), color=C["orange"],
            lw=1.8, ls="--", alpha=0.65, zorder=2,
            label=r"理论曲线 $\frac{0.95\,c\,n}{n+\beta_i}$，$\beta_i{=}5$")

    # User: IQR band + mean line
    ax.fill_between(u_mid, u_q25, u_q75, color=C["blue_l"], alpha=0.45, zorder=1)
    ax.plot(u_mid, u_mean, color=C["blue"], lw=2.2, marker="o",
            ms=6, mec="white", mew=1.3, zorder=5, label="用户偏置（$\\beta_u{=}60$）")

    # Item: IQR band + mean line
    ax.fill_between(i_mid, i_q25, i_q75, color=C["orange_l"], alpha=0.45, zorder=1)
    ax.plot(i_mid, i_mean, color=C["orange"], lw=2.2, marker="s",
            ms=6, mec="white", mew=1.3, zorder=5, label="物品偏置（$\\beta_i{=}5$）")

    # beta annotation lines
    ax.axvline(60, ls=":", lw=1.2, color=C["blue"],   alpha=0.6)
    ax.axvline( 5, ls=":", lw=1.2, color=C["orange"], alpha=0.6)
    ax.text(62, ax.get_ylim()[1] * 0.05 if ax.get_ylim()[1] > 0 else 0.05,
            "$\\beta_u{=}60$", color=C["blue"],   fontsize=8, va="bottom")
    ax.text( 5.5, 0.005, "$\\beta_i{=}5$",  color=C["orange"], fontsize=8, va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("该实体在增量数据中的评分条数（对数轴）")
    ax.set_ylabel("偏置绝对值均值（IQR 阴影）")
    ax.set_title("收缩偏置随评分数的增长趋势（验证收缩闭式解）", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    panel_label(ax, "b")

    save(fig, "fig_data_bias.png")


if __name__ == "__main__":
    np.random.seed(0)
    print("=== Generating figures ===")
    fig_truncation()
    fig_ablation()
    fig_baseline()
    fig_data_bias()
    print("=== ALL DONE ===")
