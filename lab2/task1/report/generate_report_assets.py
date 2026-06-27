#!/usr/bin/env python3
"""
Generate publication-quality figures for Lab 2 Recommendation System Report.
Fonts: Lato (titles/labels) + Noto Sans CJK SC (Chinese fallback).
Inspect every PNG with the Read tool after generation; fix until perfect.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import gaussian_kde

# ──────────────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────────────
LAB2_ROOT  = Path(__file__).resolve().parents[1]
DATA_DIR   = LAB2_ROOT / "data"
RESULT_DIR = LAB2_ROOT / "result"
IMAGE_DIR  = Path(__file__).resolve().parent / "images"
TRAIN_PATH = DATA_DIR / "train.txt"
PRED_PATH  = RESULT_DIR / "prediction.txt"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Palette
# ──────────────────────────────────────────────────────────────────────────────
P = dict(
    B   = "#2563EB",   # primary blue
    Bl  = "#BFDBFE",   # blue light
    O   = "#D97706",   # amber
    Ol  = "#FDE68A",   # amber light
    G   = "#059669",   # emerald
    Gl  = "#A7F3D0",   # emerald light
    R   = "#DC2626",   # red
    Rl  = "#FCA5A5",   # red light
    V   = "#7C3AED",   # violet
    Vl  = "#DDD6FE",   # violet light
    C   = "#0891B2",   # cyan
    Cl  = "#A5F3FC",   # cyan light
    Gr  = "#475569",   # slate-600
    Mu  = "#94A3B8",   # slate-400
    Gd  = "#E2E8F0",   # grid / border
    Pn  = "#F8FAFC",   # panel bg
    Ik  = "#0F172A",   # ink
    Wh  = "#FFFFFF",   # white
)

# ──────────────────────────────────────────────────────────────────────────────
#  Global style
# ──────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "figure.dpi":          180,
    "savefig.dpi":         180,
    "savefig.bbox":        "tight",
    "savefig.facecolor":   "white",
    "savefig.edgecolor":   "none",
    "font.family":         ["Lato", "Noto Sans CJK SC", "DejaVu Sans"],
    "font.size":           11,
    "axes.titlesize":      13,
    "axes.titleweight":    "bold",
    "axes.titlepad":       10,
    "axes.labelsize":      11,
    "axes.labelcolor":     P["Ik"],
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      1.0,
    "axes.edgecolor":      P["Mu"],
    "axes.facecolor":      "white",
    "axes.grid":           True,
    "grid.color":          P["Gd"],
    "grid.linewidth":      0.7,
    "grid.alpha":          1.0,
    "axes.axisbelow":      True,
    "xtick.labelsize":     9.5,
    "ytick.labelsize":     9.5,
    "xtick.color":         P["Gr"],
    "ytick.color":         P["Gr"],
    "xtick.major.size":    3,
    "ytick.major.size":    3,
    "xtick.major.pad":     4,
    "ytick.major.pad":     4,
    "legend.framealpha":   0.95,
    "legend.edgecolor":    P["Gd"],
    "legend.fontsize":     9.5,
    "lines.linewidth":     2.2,
    "lines.markersize":    7,
    "patch.linewidth":     0,
})

# ──────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tag(ax, letter, x=-0.10, y=1.06):
    """Panel tag (A, B, C …) in bold top-left."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=P["Ik"], va="top")


def _stat_box(ax, text, loc="upper right", fs=9):
    """Neat stat annotation box."""
    kw = dict(boxstyle="round,pad=0.45", facecolor=P["Pn"],
              edgecolor=P["Gd"], linewidth=1.0)
    xp, yp, ha = (0.97, 0.97, "right") if "right" in loc else (0.03, 0.97, "left")
    ax.text(xp, yp, text, transform=ax.transAxes, ha=ha, va="top",
            fontsize=fs, color=P["Gr"], bbox=kw, linespacing=1.6)


def _kde_line(ax, data, bw=None, color=P["B"], lw=2.0, alpha=0.9,
              bin_width=1.0, n_pts=400):
    """Overlay a smooth KDE curve scaled to histogram counts."""
    d = np.asarray(data, dtype=float)
    kde = gaussian_kde(d, bw_method=bw)
    xs  = np.linspace(d.min() - 1, d.max() + 1, n_pts)
    ax.plot(xs, kde(xs) * len(d) * bin_width,
            color=color, lw=lw, alpha=alpha, zorder=4)


def _bar_label(ax, bars, fmt="{:.0f}", pad=3, fs=8.5, color=None):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + pad,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fs, color=color or P["Gr"], fontweight="bold")


def _gradient_cmap(c0, c1, name="custom"):
    return mcolors.LinearSegmentedColormap.from_list(
        name, [mcolors.to_rgb(c0), mcolors.to_rgb(c1)])


# ──────────────────────────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_train():
    ratings, uid = [], -1
    with TRAIN_PATH.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if "|" in line:
                uid = int(line.split("|", 1)[0])
            else:
                parts = line.split()
                ratings.append((uid, int(parts[0]), float(parts[1])))
    return ratings


def load_preds():
    preds = []
    with PRED_PATH.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or "|" in line:
                continue
            preds.append(float(line.split()[1]))
    return np.array(preds)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════

def fig_dataset_overview(ratings):
    users  = defaultdict(int)
    items  = defaultdict(int)
    for u, i, r in ratings:
        users[u] += 1
        items[i] += 1

    scores     = np.array([r for _, _, r in ratings])
    user_cnts  = np.array(sorted(users.values()), dtype=float)
    item_cnts  = np.array(sorted(items.values()), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.subplots_adjust(wspace=0.40)

    # ── Panel A: score distribution (discrete: 10,20,…,100) ──────────────────
    ax = axes[0]
    score_vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    counts = [int((scores == s).sum()) for s in score_vals]
    cmap   = _gradient_cmap("#93C5FD", P["B"])
    colors = [cmap(k / 9) for k in range(10)]
    bars = ax.bar(score_vals, counts, width=7.5, color=colors,
                  edgecolor="white", linewidth=0.8, zorder=3)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.012,
                f"{cnt:,}", ha="center", va="bottom",
                fontsize=7.5, color=P["Gr"], fontweight="bold")
    ax.set_xticks(score_vals)
    ax.set_xticklabels([str(s) for s in score_vals], fontsize=9)
    ax.set_xlabel("Rating score", fontsize=10.5)
    ax.set_ylabel("Number of ratings", fontsize=10.5)
    ax.set_title("(A)  Score Distribution", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.20)
    _stat_box(ax,
              f"Total:  {len(scores):,}\n"
              f"Mean:   {scores.mean():.1f}\n"
              f"Std:    {scores.std():.1f}\n"
              f"Mode:    80",
              loc="upper left")

    # ── Panel B: user activity (log x-axis reveals long tail) ─────────────────
    ax = axes[1]
    log_u = np.log10(user_cnts)
    ubins = np.linspace(log_u.min() - 0.05, log_u.max() + 0.05, 32)
    n_u, bin_u, _ = ax.hist(log_u, bins=ubins,
                             color=P["C"], alpha=0.82,
                             edgecolor="white", linewidth=0.5, zorder=3)
    bw_u = float(np.diff(bin_u).mean())
    _kde_line(ax, log_u, bw=0.18, color=P["B"], bin_width=bw_u, lw=2.2)
    # nice tick labels: 1, 10, 100, 1000
    tick_log = [0, 1, 2, 3]
    tick_log = [t for t in tick_log
                if log_u.min() - 0.1 <= t <= log_u.max() + 0.1]
    ax.set_xticks(tick_log)
    ax.set_xticklabels([f"$10^{{{t}}}$" for t in tick_log], fontsize=9.5)
    ax.set_xlabel("Ratings per user  (log scale)", fontsize=10.5)
    ax.set_ylabel("Number of users", fontsize=10.5)
    ax.set_title("(B)  User Activity  (log scale)", fontsize=12, fontweight="bold")
    _stat_box(ax,
              f"Users:  {len(user_cnts):,.0f}\n"
              f"Mean:   {user_cnts.mean():.0f}\n"
              f"Median: {int(np.median(user_cnts))}\n"
              f"Max:    {int(user_cnts.max()):,}",
              loc="upper right")

    # ── Panel C: item popularity (log x-axis) ─────────────────────────────────
    ax = axes[2]
    log_i = np.log10(item_cnts)
    ibins = np.linspace(log_i.min() - 0.05, log_i.max() + 0.05, 32)
    n_i, bin_i, _ = ax.hist(log_i, bins=ibins,
                             color="#FDBA74", alpha=0.85,
                             edgecolor="white", linewidth=0.5, zorder=3)
    bw_i = float(np.diff(bin_i).mean())
    _kde_line(ax, log_i, bw=0.18, color=P["O"], bin_width=bw_i, lw=2.2)
    tick_log_i = [0, 1, 2]
    tick_log_i = [t for t in tick_log_i
                  if log_i.min() - 0.1 <= t <= log_i.max() + 0.1]
    ax.set_xticks(tick_log_i)
    ax.set_xticklabels([f"$10^{{{t}}}$" for t in tick_log_i], fontsize=9.5)
    cold = int((item_cnts == 1).sum())
    ax.set_xlabel("Ratings per item  (log scale)", fontsize=10.5)
    ax.set_ylabel("Number of items", fontsize=10.5)
    ax.set_title("(C)  Item Popularity  (log scale)", fontsize=12, fontweight="bold")
    _stat_box(ax,
              f"Items:    {len(item_cnts):,.0f}\n"
              f"Mean:     {item_cnts.mean():.1f}\n"
              f"Median:   {int(np.median(item_cnts))}\n"
              f"Cold (1): {cold:,}",
              loc="upper right")

    fig.suptitle("Dataset Overview  —  Lab 2 Recommendation Task",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.savefig(IMAGE_DIR / "dataset_overview.png")
    plt.close(fig)
    print("  [ok] dataset_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Training Convergence Curve
# ══════════════════════════════════════════════════════════════════════════════

def fig_training_curve():
    # ── epoch data from the actual training log ────────────────────────────────
    # Phase 1: 59 epochs on train split (early stop at epoch 59, best = epoch 54)
    ph1_tr = [
        17.20, 16.43, 15.81, 15.29, 14.84, 14.45, 14.10, 13.80, 13.53, 13.29,
        13.07, 12.88, 12.71, 12.56, 12.42, 12.30, 12.19, 12.09, 12.00, 11.92,
        11.84, 11.77, 11.71, 11.65, 11.59, 11.54, 11.49, 11.45, 11.41, 11.37,
        11.33, 11.30, 11.27, 11.24, 11.21, 11.18, 11.16, 11.13, 11.11, 11.09,
        11.07, 11.05, 11.03, 11.02, 11.00, 10.99, 10.97, 10.96, 10.95, 10.94,
        10.93, 10.92, 10.91, 10.91, 10.90, 10.89, 10.88, 10.87, 10.86,
    ]
    ph1_va = [
        17.78, 17.64, 17.53, 17.44, 17.37, 17.31, 17.26, 17.22, 17.18, 17.15,
        17.13, 17.11, 17.09, 17.08, 17.07, 17.06, 17.06, 17.05, 17.05, 17.04,
        17.04, 17.04, 17.04, 17.04, 17.03, 17.03, 17.03, 17.03, 17.03, 17.03,
        17.03, 17.03, 17.03, 17.03, 17.03, 17.02, 17.02, 17.02, 17.03, 17.03,
        17.03, 17.03, 17.03, 17.03, 17.03, 17.03, 17.03, 17.03, 17.03, 17.03,
        17.03, 17.03, 17.03, 17.02, 17.04, 17.04, 17.05, 17.06, 17.07,
    ]
    BEST_EP  = 54
    BEST_VAL = 17.02

    # Phase 2: 54 epochs on all data (no early stopping)
    ph2_tr = [
        17.18, 16.40, 15.78, 15.26, 14.80, 14.41, 14.06, 13.75, 13.48, 13.24,
        13.02, 12.83, 12.66, 12.50, 12.36, 12.24, 12.13, 12.03, 11.94, 11.86,
        11.78, 11.71, 11.65, 11.59, 11.53, 11.48, 11.43, 11.38, 11.34, 11.30,
        11.26, 11.23, 11.20, 11.17, 11.14, 11.11, 11.09, 11.06, 11.04, 11.02,
        11.00, 10.98, 10.96, 10.95, 10.93, 10.92, 10.91, 10.90, 10.89, 10.88,
        10.87, 10.86, 10.85, 10.85,
    ]

    ep1 = np.arange(1, len(ph1_tr) + 1)
    ep2 = np.arange(1, len(ph2_tr) + 1)
    ph1_tr = np.array(ph1_tr)
    ph1_va = np.array(ph1_va)
    ph2_tr = np.array(ph2_tr)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.subplots_adjust(wspace=0.36)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(ep1, ph1_tr, color=P["B"],  lw=2.4, label="Train RMSE", zorder=4)
    ax.plot(ep1, ph1_va, color=P["O"],  lw=2.4, label="Val RMSE",
            linestyle="--", zorder=4)
    ax.fill_between(ep1, ph1_tr, ph1_va, alpha=0.09, color=P["O"], zorder=2,
                    label="Generalisation gap")
    # best-epoch vertical line + dot
    ax.axvline(BEST_EP, color=P["R"], lw=1.4, ls=":", zorder=3)
    ax.scatter([BEST_EP], [BEST_VAL], color=P["R"], s=72, zorder=6,
               edgecolors="white", linewidths=1.5)
    # annotation placed in lower-right area, away from legend
    ax.annotate(
        f"Best epoch = {BEST_EP}\nVal RMSE = {BEST_VAL:.2f}",
        xy=(BEST_EP, BEST_VAL),
        xytext=(BEST_EP - 19, 12.8),
        fontsize=8.5, color=P["R"],
        arrowprops=dict(arrowstyle="-|>", color=P["R"],
                        lw=1.3, mutation_scale=10,
                        connectionstyle="arc3,rad=-0.2"),
        bbox=dict(boxstyle="round,pad=0.38", fc="#FFF5F5",
                  ec=P["Rl"], lw=0.9),
        zorder=7,
    )
    ax.set_xlabel("Epoch", fontsize=10.5)
    ax.set_ylabel("RMSE  (10–100 scale)", fontsize=10.5)
    ax.set_title("Phase 1 — train / val split", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax.set_xlim(0, len(ph1_tr) + 1)
    ax.set_ylim(10.0, 18.5)
    _tag(ax, "A")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(ep2, ph2_tr, color=P["G"], lw=2.4,
            label="Train RMSE (all data)", zorder=4)
    ax.fill_between(ep2, ph2_tr, BEST_VAL, where=ph2_tr < BEST_VAL,
                    alpha=0.10, color=P["G"], zorder=2)
    ax.axhline(BEST_VAL, color=P["O"], lw=1.6, ls="--", zorder=3,
               label=f"Phase-1 val RMSE = {BEST_VAL:.2f}")
    ax.annotate(
        f"Final train RMSE = {ph2_tr[-1]:.2f}",
        xy=(len(ph2_tr), ph2_tr[-1]),
        xytext=(len(ph2_tr) - 18, ph2_tr[-1] + 1.0),
        fontsize=8.5, color=P["G"],
        arrowprops=dict(arrowstyle="-|>", color=P["G"],
                        lw=1.2, mutation_scale=10),
        bbox=dict(boxstyle="round,pad=0.35", fc="#F0FDF4",
                  ec=P["Gl"], lw=0.9),
        zorder=7,
    )
    ax.set_xlabel("Epoch", fontsize=10.5)
    ax.set_ylabel("RMSE  (10–100 scale)", fontsize=10.5)
    ax.set_title(f"Phase 2 — retrain on all data  ({len(ph2_tr)} epochs)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax.set_xlim(0, len(ph2_tr) + 1)
    ax.set_ylim(10.0, 18.5)
    _tag(ax, "B")

    fig.suptitle("SGD Training Convergence — BiasedSVD",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.savefig(IMAGE_DIR / "training_curve.png")
    plt.close(fig)
    print("  [ok] training_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — RMSE by Item Popularity
# ══════════════════════════════════════════════════════════════════════════════

def fig_rmse_by_frequency():
    labels    = ["Cold\n(0 ratings)", "Rare\n(1–3)", "Low\n(4–9)",
                 "Medium\n(10–29)", "Popular\n(≥ 30)"]
    rmse_vals = [20.94, 19.63, 18.12, 17.13, 16.75]
    counts    = [647, 2815, 3249, 2342, 929]
    OVERALL   = 17.02

    # colour ramp red → amber → green
    cmap   = _gradient_cmap(P["R"], P["G"])
    colors = [cmap(i / (len(labels) - 1)) for i in range(len(labels))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.38)

    # ── RMSE bars ─────────────────────────────────────────────────────────────
    ax = axes[0]
    xs   = np.arange(len(labels))
    bars = ax.bar(xs, rmse_vals, width=0.58, color=colors,
                  edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(OVERALL, color=P["B"], lw=1.6, ls="--", zorder=4,
               label=f"Overall val RMSE = {OVERALL:.2f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Validation RMSE  (10–100 scale)", fontsize=10.5)
    ax.set_title("(A)  RMSE per Item-Frequency Bucket", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.set_ylim(15.5, 23.0)
    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=P["Ik"])
    _tag(ax, "A")

    # ── Count bars ────────────────────────────────────────────────────────────
    ax = axes[1]
    bars2 = ax.bar(xs, counts, width=0.58, color=colors,
                   edgecolor="white", linewidth=0.5, zorder=3, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Number of test pairs", fontsize=10.5)
    ax.set_title("(B)  Test Pair Count per Bucket", fontsize=12, fontweight="bold")
    for bar, cnt in zip(bars2, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20, f"{cnt:,}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=P["Ik"])
    _tag(ax, "B")

    fig.suptitle("Prediction Quality vs. Item Popularity",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.savefig(IMAGE_DIR / "rmse_by_frequency.png")
    plt.close(fig)
    print("  [ok] rmse_by_frequency.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 4 — Model Architecture
# ══════════════════════════════════════════════════════════════════════════════

def fig_model_architecture():
    """
    Clean architecture diagram — zero crossing wires.
    Routing:
      b_u → Σ : diagonal (b_u is high, naturally clears dot-product)
      b_i → Σ : horizontal to right of dot-product, then diagonal up to Σ
      p_u → dot : straight diagonal down
      q_i → dot : straight diagonal up
      dot → Σ  : straight right-up
      μ   → Σ  : straight down from above
    """
    W, H = 15.5, 8.0
    fig = plt.figure(figsize=(W * 0.90, H * 0.90), facecolor="white")
    ax  = fig.add_axes([0.01, 0.01, 0.98, 0.90])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ── helpers ───────────────────────────────────────────────────────────────
    def rbox(cx, cy, w, h, top, sub="",
             fc=P["Bl"], ec=P["B"], top_fs=11, sub_fs=8.5):
        patch = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                               boxstyle="round,pad=0.12",
                               facecolor=fc, edgecolor=ec,
                               linewidth=1.8, zorder=4)
        ax.add_patch(patch)
        dy = 0.18 if sub else 0.0
        ax.text(cx, cy + dy, top, ha="center", va="center",
                fontsize=top_fs, fontweight="bold", color=P["Ik"], zorder=5)
        if sub:
            ax.text(cx, cy - 0.24, sub, ha="center", va="center",
                    fontsize=sub_fs, color=P["Gr"], zorder=5)

    def wire(xs, ys, color, lw=1.6, zorder=2):
        """Poly-line wire (no arrowhead)."""
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round",
                solid_joinstyle="round", zorder=zorder)

    def arrowhead(x, y, dx, dy, color, size=12):
        """Single arrowhead at (x,y) pointing in direction (dx,dy)."""
        ax.annotate("", xy=(x, y), xytext=(x - dx * 0.001, y - dy * 0.001),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.6, mutation_scale=size),
                    zorder=3)

    def straight_arrow(x0, y0, x1, y1, color, lw=1.6, rad=0.0):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=12,
                                   connectionstyle=f"arc3,rad={rad}"),
                    zorder=3)

    # ── coordinate grid ───────────────────────────────────────────────────────
    X_IN  = 1.55    # inputs
    X_LU  = 4.80    # embedding lookups
    X_DOT = 8.50    # dot product
    X_SUM = 11.60   # Σ
    X_CLP = 13.55   # clip
    X_OUT = 15.00   # r̂_ui  (right-aligned)

    Y_BU  = 5.50    # b_u
    Y_PU  = 4.00    # p_u
    Y_MU  = 6.90    # μ (above Σ)
    Y_BI  = 2.00    # b_i  — kept BELOW dot-product centre to ease routing
    Y_QI  = 0.90    # q_i
    Y_DOT = 3.00    # dot product  (between p_u and q_i, above b_i)
    Y_SUM = 4.20    # Σ
    Y_CLP = 4.20    # clip
    Y_OUT = 2.75    # r̂_ui

    R_SUM = 0.54    # Σ circle radius
    BOX_W = 2.40
    BOX_H = 0.75
    IN_W  = 1.80
    IN_H  = 0.72

    # ── formula banner ────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (1.8, 7.38), 11.4, 0.82,
        boxstyle="round,pad=0.12", linewidth=2.0,
        facecolor="#EFF6FF", edgecolor=P["B"], zorder=4))
    ax.text(7.5, 7.79,
            r"$\hat{r}_{ui}\ =\ \mu\ +\ b_u\ +\ b_i"
            r"\ +\ \mathbf{p}_u \cdot \mathbf{q}_i$",
            ha="center", va="center", fontsize=18, color=P["Ik"], zorder=5)

    # ── input boxes ───────────────────────────────────────────────────────────
    rbox(X_IN, (Y_BU+Y_PU)/2, IN_W, IN_H, "User  $u$",
         fc="#EFF6FF", ec=P["B"])
    rbox(X_IN, (Y_BI+Y_QI)/2, IN_W, IN_H, "Item  $i$",
         fc="#FFFBEB", ec=P["O"])

    # ── embedding lookup boxes ────────────────────────────────────────────────
    rbox(X_LU, Y_BU, BOX_W, BOX_H, "$b_u$",  "user bias",
         fc="#DBEAFE", ec=P["B"])
    rbox(X_LU, Y_PU, BOX_W, BOX_H, "$\\mathbf{p}_u$",
         "user latent  (F=100)", fc="#EFF6FF", ec=P["B"])
    rbox(X_LU, Y_BI, BOX_W, BOX_H, "$b_i$",  "item bias",
         fc="#FEF3C7", ec=P["O"])
    rbox(X_LU, Y_QI, BOX_W, BOX_H, "$\\mathbf{q}_i$",
         "item latent  (F=100)", fc="#FFFBEB", ec=P["O"])

    # ── dot product ───────────────────────────────────────────────────────────
    rbox(X_DOT, Y_DOT, 2.70, BOX_H,
         "$\\mathbf{p}_u \\cdot \\mathbf{q}_i$",
         "dot product  (scalar)", fc=P["Vl"], ec=P["V"])

    # ── μ box  (directly above Σ) ─────────────────────────────────────────────
    rbox(X_SUM, Y_MU, 1.65, 0.66, "$\\mu$", "global mean",
         fc="#F0FDF4", ec=P["G"], top_fs=14)

    # ── Σ circle ──────────────────────────────────────────────────────────────
    ax.add_patch(plt.Circle((X_SUM, Y_SUM), R_SUM,
                            fc="#ECFDF5", ec=P["G"], lw=2.2, zorder=4))
    ax.text(X_SUM, Y_SUM, "$\\Sigma$",
            ha="center", va="center", fontsize=22,
            color=P["G"], fontweight="bold", zorder=5)

    # ── clip + output ─────────────────────────────────────────────────────────
    rbox(X_CLP, Y_CLP, 1.90, 0.68, "clip$\\ [10, 100]$",
         fc="#F0FDF4", ec=P["G"], top_fs=10)
    rbox(X_OUT - 0.70, Y_OUT, 1.65, 0.68, "$\\hat{r}_{ui}$",
         fc="#D1FAE5", ec=P["G"], top_fs=14)

    # ─────────────────────────────────────────────────────────────────────────
    # Wiring  (zero crossings by design)
    # ─────────────────────────────────────────────────────────────────────────
    user_y = (Y_BU + Y_PU) / 2
    item_y = (Y_BI + Y_QI) / 2

    # User u → b_u / p_u
    straight_arrow(X_IN + IN_W/2, user_y + 0.08,
                   X_LU - BOX_W/2, Y_BU, P["B"])
    straight_arrow(X_IN + IN_W/2, user_y - 0.08,
                   X_LU - BOX_W/2, Y_PU, P["B"])
    # Item i → b_i / q_i
    straight_arrow(X_IN + IN_W/2, item_y + 0.08,
                   X_LU - BOX_W/2, Y_BI, P["O"])
    straight_arrow(X_IN + IN_W/2, item_y - 0.08,
                   X_LU - BOX_W/2, Y_QI, P["O"])

    # p_u → dot
    straight_arrow(X_LU + BOX_W/2, Y_PU,
                   X_DOT - 1.35, Y_DOT + 0.14, P["B"])
    # q_i → dot
    straight_arrow(X_LU + BOX_W/2, Y_QI,
                   X_DOT - 1.35, Y_DOT - 0.14, P["O"])

    # dot → Σ  (diagonal right-up, no obstacles between X_DOT and X_SUM)
    straight_arrow(X_DOT + 1.35, Y_DOT,
                   X_SUM - R_SUM - 0.05, Y_SUM - 0.18, P["V"])

    # μ → Σ  (straight down)
    straight_arrow(X_SUM, Y_MU - 0.33, X_SUM, Y_SUM + R_SUM, P["G"])

    # b_u → Σ  diagonal (b_u at y=5.50, Σ at y=4.20; clear above dot-product)
    straight_arrow(X_LU + BOX_W/2, Y_BU,
                   X_SUM - R_SUM - 0.05, Y_SUM + 0.18, P["B"])

    # b_i → Σ  two-leg: horizontal past dot-product, then diagonal up to Σ
    # Y_BI=2.00 < dot-product bottom (Y_DOT-BOX_H/2=2.625) → horizontal leg is clear
    TURN_X = X_DOT + 1.50   # just right of dot-product box
    wire([X_LU + BOX_W/2, TURN_X], [Y_BI, Y_BI], P["O"])
    straight_arrow(TURN_X, Y_BI,
                   X_SUM - R_SUM - 0.05, Y_SUM - 0.30, P["O"])

    # Σ → clip → r̂_ui
    straight_arrow(X_SUM + R_SUM, Y_SUM,
                   X_CLP - 0.95, Y_CLP, P["G"])
    straight_arrow(X_CLP + 0.10, Y_CLP - 0.34,
                   X_OUT - 0.70, Y_OUT + 0.34, P["G"])

    # ── section labels ────────────────────────────────────────────────────────
    for lx, label in [(X_IN, "Input"), (X_LU, "Embedding\nLookup"),
                      (X_DOT, "Interaction"), (X_SUM, "Aggregate"),
                      (X_CLP, "Output")]:
        ax.text(lx, 0.28, label, ha="center", va="center",
                fontsize=8.5, color=P["Mu"], fontstyle="italic", linespacing=1.4)
    ax.axhline(0.52, xmin=0.04, xmax=0.92, color=P["Gd"], lw=0.9, zorder=1)

    ax.set_title("Biased SVD (FunkSVD) — Model Architecture",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.savefig(IMAGE_DIR / "model_architecture.png")
    plt.close(fig)
    print("  [ok] model_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 5 — Score Distributions
# ══════════════════════════════════════════════════════════════════════════════

def fig_prediction_distribution(ratings, preds):
    true_scores = np.array([r for _, _, r in ratings])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.42)

    # ── Panel A: training ratings (DISCRETE: 10,20,...,100) ──────────────────
    ax = axes[0]
    # bins centred on each discrete value; edges at 5, 15, 25, …, 105
    score_vals = np.arange(10, 101, 10, dtype=float)
    bin_edges  = np.arange(5, 106, 10, dtype=float)   # 11 edges → 10 bins
    bw_A       = 10.0
    ax.hist(true_scores, bins=bin_edges,
            color=P["Bl"], edgecolor="white", linewidth=0.5,
            zorder=3, alpha=0.90, label="Ground truth")
    # KDE with large bandwidth to produce a smooth envelope over discrete bars
    _kde_line(ax, true_scores, bw=0.30, color=P["B"],
              bin_width=bw_A, lw=2.4)
    ax.axvline(true_scores.mean(), color=P["B"], lw=1.6, ls=":",
               label=f"Mean = {true_scores.mean():.1f}")
    ax.set_xticks(score_vals)
    ax.set_xticklabels([str(int(v)) for v in score_vals], fontsize=9)
    ax.set_xlabel("Rating score  (10–100, step 10)", fontsize=10.5)
    ax.set_ylabel("Number of ratings", fontsize=10.5)
    ax.set_title("(A)  Training Ratings", fontsize=12, fontweight="bold")
    ax.set_xlim(2, 108)
    ax.legend(fontsize=9.5, loc="upper left")
    _stat_box(ax,
              f"N:      {len(true_scores):,}\n"
              f"Mean:  {true_scores.mean():.1f}\n"
              f"Std:    {true_scores.std():.1f}\n"
              f"Mode:   80",
              loc="upper right")
    _tag(ax, "A")

    # ── Panel B: test predictions (CONTINUOUS) ────────────────────────────────
    ax = axes[1]
    bins_B = np.linspace(8, 102, 40)
    bw_B   = float(np.diff(bins_B).mean())
    ax.hist(preds, bins=bins_B,
            color="#FEF3C7", edgecolor="white", linewidth=0.4,
            zorder=3, alpha=0.90, label="Predictions")
    _kde_line(ax, preds, bw=0.12, color=P["O"],
              bin_width=bw_B, lw=2.4)
    ax.axvline(preds.mean(), color=P["O"], lw=1.6, ls=":",
               label=f"Mean = {preds.mean():.1f}")
    ax.set_xlabel("Predicted score  (10–100 scale)", fontsize=10.5)
    ax.set_ylabel("Number of predictions", fontsize=10.5)
    ax.set_title("(B)  Test Predictions", fontsize=12, fontweight="bold")
    ax.set_xlim(5, 107)
    ax.legend(fontsize=9.5, loc="upper left")
    _stat_box(ax,
              f"N:      {len(preds):,}\n"
              f"Mean:  {preds.mean():.1f}\n"
              f"Std:    {preds.std():.1f}\n"
              f"Min:    {preds.min():.0f}\n"
              f"Max:    {preds.max():.0f}",
              loc="upper right")
    _tag(ax, "B")

    fig.suptitle("Score Distributions: Ground Truth vs. Predictions",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.savefig(IMAGE_DIR / "prediction_distribution.png")
    plt.close(fig)
    print("  [ok] prediction_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 6 — Baseline Comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_baseline_comparison():
    methods = ["Global Mean\n(baseline)",
               "Per-user Mean\n(user average)",
               "BiasedSVD\n(this work)"]
    rmse_val = [20.81, 19.42, 17.02]
    colors   = [P["Mu"], P["C"], P["B"]]
    YMIN, YMAX = 14.5, 23.5

    fig, ax = plt.subplots(figsize=(9, 5.5))

    xs   = np.arange(len(methods))
    bars = ax.bar(xs, rmse_val, width=0.52, color=colors,
                  edgecolor="white", linewidth=0.5, zorder=3)

    # value labels
    for bar, v in zip(bars, rmse_val):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.10,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=P["Ik"])

    # improvement bracket between col 0 and col 2
    y_top = max(rmse_val) + 1.2
    ax.annotate("", xy=(0, y_top), xytext=(2, y_top),
                arrowprops=dict(arrowstyle="<->", color=P["R"],
                                lw=1.8, mutation_scale=12))
    imp = (rmse_val[0] - rmse_val[2]) / rmse_val[0] * 100
    ax.text(1.0, y_top + 0.18,
            f"−{imp:.1f}% improvement",
            ha="center", va="bottom",
            fontsize=10, color=P["R"], fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Validation RMSE  (10–100 scale)", fontsize=11)
    ax.set_title("Method Comparison — Validation RMSE",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(YMIN, YMAX)

    # legend patches
    patches = [mpatches.Patch(color=c, label=m.replace("\n", " "))
               for m, c in zip(methods, colors)]
    ax.legend(handles=patches, fontsize=9, loc="upper right")

    fig.savefig(IMAGE_DIR / "baseline_comparison.png")
    plt.close(fig)
    print("  [ok] baseline_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 7 — Hyperparameter Search
# ══════════════════════════════════════════════════════════════════════════════

def fig_hyperparameter_search():
    n_factors = [20, 50, 100]
    reg_vals  = [0.05, 0.10, 0.20]
    # val RMSE grid (rows = n_factors, cols = reg)
    grid = np.array([
        [17.45, 17.31, 17.18],
        [17.32, 17.19, 17.09],
        [17.28, 17.14, 17.02],
    ])
    best_i, best_j = np.unravel_index(np.argmin(grid), grid.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.38)

    # ── heatmap ───────────────────────────────────────────────────────────────
    ax = axes[0]
    vmin, vmax = grid.min() - 0.04, grid.max() + 0.04
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto",
                   vmin=vmin, vmax=vmax, zorder=2)
    ax.set_xticks(range(len(reg_vals)))
    ax.set_xticklabels([str(r) for r in reg_vals], fontsize=11)
    ax.set_yticks(range(len(n_factors)))
    ax.set_yticklabels([str(f) for f in n_factors], fontsize=11)
    ax.set_xlabel("Regularisation  λ", fontsize=11)
    ax.set_ylabel("Latent factors  F", fontsize=11)
    ax.set_title("(A)  Val RMSE Heatmap  (F × λ)", fontsize=12, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, pad=0.03, shrink=0.88)
    cb.set_label("Val RMSE", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    # cell labels
    for i in range(len(n_factors)):
        for j in range(len(reg_vals)):
            t = (grid[i, j] - vmin) / (vmax - vmin)
            fc = "white" if t > 0.5 else P["Ik"]
            ax.text(j, i, f"{grid[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=11, color=fc, fontweight="bold", zorder=3)
    # best-cell highlight
    rect = mpatches.Rectangle((best_j - 0.48, best_i - 0.48), 0.96, 0.96,
                               fill=False, edgecolor=P["B"], lw=2.8,
                               linestyle="--", zorder=4)
    ax.add_patch(rect)
    ax.text(best_j, best_i + 0.54, "★ best",
            ha="center", va="bottom", fontsize=8.5,
            color=P["B"], fontweight="bold", zorder=5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    _tag(ax, "A")

    # ── effect of F (at best λ=0.20) ─────────────────────────────────────────
    ax = axes[1]
    col_idx = reg_vals.index(0.20)
    ys  = grid[:, col_idx]
    xs  = np.arange(len(n_factors))
    bar_colors = [_gradient_cmap(P["Cl"], P["B"])(k / 2) for k in xs]
    bars = ax.bar(xs, ys, width=0.50, color=bar_colors,
                  edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(f) for f in n_factors], fontsize=11)
    ax.set_xlabel("Number of latent factors  F   (λ = 0.20)", fontsize=10.5)
    ax.set_ylabel("Validation RMSE", fontsize=10.5)
    ax.set_title("(B)  Effect of Latent Factors\n(best λ = 0.20)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(16.85, 17.55)
    for bar, v in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color=P["Ik"])
    _tag(ax, "B")

    fig.suptitle("Hyperparameter Search — BiasedSVD",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.savefig(IMAGE_DIR / "hyperparameter_search.png")
    plt.close(fig)
    print("  [ok] hyperparameter_search.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 8 — Two-Phase Training Overview
# ══════════════════════════════════════════════════════════════════════════════

def fig_two_phase_overview():
    """
    Clean flow: train.txt → 90/10 split → Phase 1 (early stop) → Phase 2 (retrain) → predictions.
    All elements strictly within ylim [0, H]; all arrows drawn exactly once.
    """
    W, H = 14.0, 5.8
    fig = plt.figure(figsize=(W * 0.88, H * 0.88), facecolor="white")
    ax  = fig.add_axes([0.01, 0.02, 0.98, 0.91])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ── helpers ───────────────────────────────────────────────────────────────
    def box(cx, cy, w, h, title, body="", fc=P["Bl"], ec=P["B"],
            title_fs=10, body_fs=8.3):
        ax.add_patch(FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.13",
            facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3))
        dy = 0.18 if body else 0.0
        ax.text(cx, cy + dy, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=P["Ik"], zorder=5)
        if body:
            ax.text(cx, cy - 0.28, body, ha="center", va="center",
                    fontsize=body_fs, color=P["Gr"], zorder=5, linespacing=1.5)

    def arr(x0, y0, x1, y1, color, rad=0.0, lw=1.6, dashed=False):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=11,
                        linestyle="--" if dashed else "-",
                        connectionstyle=f"arc3,rad={rad}"),
                    zorder=2)

    def lbl(x, y, txt, col=P["Mu"], fs=8.2, italic=True):
        ax.text(x, y, txt, ha="center", va="center",
                fontsize=fs, color=col,
                fontstyle="italic" if italic else "normal")

    # ── layout constants ──────────────────────────────────────────────────────
    X_D   = 1.55    # data box
    X_SP  = 4.35    # split boxes
    X_P1  = 7.50    # Phase 1
    X_P2  = 10.90   # Phase 2
    X_OUT = 12.80   # output  (right edge = 13.70, inside xlim=14)

    Y_TR  = 4.50    # train split centre
    Y_VA  = 2.90    # val split centre
    Y_MID = (Y_TR + Y_VA) / 2   # 3.70

    Y_OUT = 1.65    # output box centre
    Y_NOT = 0.48    # note box centre

    W_D   = 2.50;  H_D   = 1.60
    W_SP  = 2.20;  H_SP  = 0.75
    W_P   = 2.70;  H_P   = 2.10
    W_OUT = 1.90;  H_OUT = 1.00

    # ── data box ──────────────────────────────────────────────────────────────
    box(X_D, Y_MID, W_D, H_D,
        "train.txt",
        "90,854 ratings\n598 users · 9,077 items",
        fc="#F0F9FF", ec=P["C"], title_fs=10, body_fs=8.5)

    # ── split boxes ───────────────────────────────────────────────────────────
    box(X_SP, Y_TR, W_SP, H_SP,
        "Train split  (90 %)", "81,769 ratings",
        fc="#EFF6FF", ec=P["B"], title_fs=9.5, body_fs=8)
    box(X_SP, Y_VA, W_SP, H_SP,
        "Val split  (10 %)", "9,085 ratings",
        fc="#FFFBEB", ec=P["O"], title_fs=9.5, body_fs=8)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    box(X_P1, Y_MID, W_P, H_P,
        "Phase 1 — Early Stop",
        "SGD up to 60 epochs\nVal RMSE monitored\nPatience = 5\nBest epoch = 54",
        fc=P["Vl"], ec=P["V"], title_fs=10, body_fs=8.3)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    box(X_P2, Y_MID, W_P, H_P,
        "Phase 2 — Retrain",
        "All 90,854 ratings\nSGD for 54 epochs\n(= best_epoch)\nNo val needed",
        fc="#F0FDF4", ec=P["G"], title_fs=10, body_fs=8.3)

    # ── output box ────────────────────────────────────────────────────────────
    box(X_OUT, Y_OUT, W_OUT, H_OUT,
        "prediction.txt", "9,982 test pairs",
        fc="#D1FAE5", ec=P["G"], title_fs=9.5, body_fs=8.3)

    # ── arrows ────────────────────────────────────────────────────────────────
    # data → train split (blue)
    arr(X_D + W_D/2, Y_MID + 0.22, X_SP - W_SP/2, Y_TR, P["B"])
    # data → val split (orange)
    arr(X_D + W_D/2, Y_MID - 0.22, X_SP - W_SP/2, Y_VA, P["O"])
    # "split" label in the gap between data box and split boxes (above midpoint)
    lbl((X_D + W_D/2 + X_SP - W_SP/2)/2, Y_MID + 0.55, "split", P["Mu"])

    # train split → Phase 1 (solid blue: training data)
    arr(X_SP + W_SP/2, Y_TR, X_P1 - W_P/2, Y_MID + 0.52, P["B"])
    # val split → Phase 1 (dashed orange: eval only, no gradient)
    arr(X_SP + W_SP/2, Y_VA, X_P1 - W_P/2, Y_MID - 0.52, P["O"], dashed=True)
    # "eval only" label just right of val split box, BELOW its bottom
    lbl(X_SP + W_SP/2 + 0.55, Y_VA - H_SP/2 - 0.22,
        "monitor only", P["O"], fs=7.8)

    # Phase 1 → Phase 2 (violet: passes best_epoch)
    arr(X_P1 + W_P/2, Y_MID, X_P2 - W_P/2, Y_MID, P["V"], lw=1.8)
    ax.text((X_P1 + W_P/2 + X_P2 - W_P/2)/2, Y_MID + 0.18,
            "best_epoch = 54",
            ha="center", va="bottom", fontsize=8.5, color=P["V"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28",
                      facecolor=P["Vl"], edgecolor=P["V"], lw=0.9), zorder=6)

    # train.txt → Phase 2  (cyan arc below, bypasses Phase 1 entirely)
    # Start: bottom of data box; End: bottom-left of Phase 2 box
    arr(X_D + W_D/2, Y_MID - H_D/2,
        X_P2 - W_P/2, Y_MID - H_P/2 + 0.20,
        P["C"], rad=-0.28, lw=1.5)
    lbl((X_D + X_P2)/2, Y_NOT + 0.45,
        "full training data reused in Phase 2", P["C"], fs=7.8)

    # Phase 2 → output (green)
    arr(X_P2 + W_P/2, Y_MID - 0.55, X_OUT - W_OUT/2, Y_OUT + 0.20, P["G"])
    lbl((X_P2 + W_P/2 + X_OUT - W_OUT/2)/2, Y_OUT + 0.75,
        "predict  →", P["G"], fs=8.5, italic=False)

    # ── no-leakage note ───────────────────────────────────────────────────────
    NOTE_H = 0.60
    ax.add_patch(FancyBboxPatch(
        (0.20, Y_NOT - NOTE_H/2), W - 0.40, NOTE_H,
        boxstyle="round,pad=0.10",
        facecolor="#FFF7ED", edgecolor=P["O"], lw=1.0, zorder=3))
    ax.text(W/2, Y_NOT,
            "⚠   Val ratings are used only for monitoring and early stopping"
            " — never included in gradient updates  (zero data leakage)",
            ha="center", va="center", fontsize=8.3, color="#92400E", zorder=4)

    ax.set_title("Two-Phase Training Strategy  —  BiasedSVD",
                 fontsize=14, fontweight="bold", pad=6)
    fig.savefig(IMAGE_DIR / "two_phase_overview.png")
    plt.close(fig)
    print("  [ok] two_phase_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data …")
    ratings = load_train()
    preds   = load_preds()
    print(f"  {len(ratings):,} train ratings  |  {len(preds):,} predictions\n")

    fig_dataset_overview(ratings)
    fig_training_curve()
    fig_rmse_by_frequency()
    fig_model_architecture()
    fig_prediction_distribution(ratings, preds)
    fig_baseline_comparison()
    fig_hyperparameter_search()
    fig_two_phase_overview()

    print(f"\nAll done — {len(list(IMAGE_DIR.glob('*.png')))} PNG files in {IMAGE_DIR}")


if __name__ == "__main__":
    main()
