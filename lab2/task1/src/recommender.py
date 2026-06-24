#!/usr/bin/env python3
"""Lab 2: Explicit rating prediction via Biased Matrix Factorisation (SVD).

Algorithm: regularised biased MF (FunkSVD / Koren 2009), trained with SGD.

  r̂_ui = μ + b_u + b_i + p_u · q_i

Design notes
------------
* Scores [10, 100] are normalised to [1, 10] before training so that
  standard SGD hyper-parameters apply; predictions are de-normalised.
* Biases are warm-started with shrinkage estimates (Koren 2009 §2.1) to
  avoid noisy starting points for items with very few ratings.
* A 10 % validation split provides early stopping with best-checkpoint
  restore; val labels are never used in gradient updates (no leakage).
* After early stopping finds the optimal epoch N*, the model is retrained
  on all training data for N* epochs so the full dataset is utilised.

References
----------
Koren, Bell & Volinsky, "Matrix Factorization Techniques for Recommender
Systems", IEEE Computer 42(8), 2009.

Hug, "Surprise: A Python library for recommender systems", JOSS 5(52), 2020.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

Rating    = Tuple[int, int, float]
TestPair  = Tuple[int, int]
GroupInfo = Tuple[int, int, int]

SCORE_LO, SCORE_HI = 10.0, 100.0
_NORM = 10.0  # divide raw scores by this to normalise [10,100] → [1,10]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_train(path: Path) -> List[Rating]:
    """Parse train.txt → list of (user_id, item_id, score)."""
    ratings: List[Rating] = []
    uid = -1
    with path.open(encoding="utf-8") as fh:
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


def load_test(path: Path) -> Tuple[List[TestPair], List[GroupInfo]]:
    """Parse test.txt → (pairs to predict, groups for output format)."""
    pairs: List[TestPair] = []
    groups: List[GroupInfo] = []
    uid = -1
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if "|" in line:
                uid_str, cnt_str = line.split("|", 1)
                uid = int(uid_str)
                groups.append((uid, len(pairs), int(cnt_str)))
            else:
                pairs.append((uid, int(line)))
    return pairs, groups


def write_predictions(
    path: Path,
    groups: List[GroupInfo],
    item_ids: List[int],
    scores: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for uid, start, count in groups:
            fh.write(f"{uid}|{count}\n")
            for k in range(start, start + count):
                fh.write(f"{item_ids[k]} {scores[k]:.6f}\n")


# ---------------------------------------------------------------------------
# Shrinkage bias warm-start (Koren 2009 §2.1)
# ---------------------------------------------------------------------------

def _shrinkage_biases(
    data: List[Tuple[int, int, float]],
    mu: float,
    shrink_item: float,
    shrink_user: float,
    n_users: int,
    n_items: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute regularised item and user biases for SGD warm-start.

    b̂_i = Σ_u (r̃_ui − μ) / (λ_i + |R(i)|)
    b̂_u = Σ_i (r̃_ui − μ − b̂_i) / (λ_u + |R(u)|)

    Items/users with few ratings are heavily shrunk toward 0.
    """
    item_sum: Dict[int, float] = defaultdict(float)
    item_cnt: Dict[int, int]   = defaultdict(int)
    for u, i, r in data:
        item_sum[i] += r - mu
        item_cnt[i] += 1
    bi = np.zeros(n_items)
    for i, s in item_sum.items():
        bi[i] = s / (shrink_item + item_cnt[i])

    user_sum: Dict[int, float] = defaultdict(float)
    user_cnt: Dict[int, int]   = defaultdict(int)
    for u, i, r in data:
        user_sum[u] += r - mu - bi[i]
        user_cnt[u] += 1
    bu = np.zeros(n_users)
    for u, s in user_sum.items():
        bu[u] = s / (shrink_user + user_cnt[u])

    return bu, bi


# ---------------------------------------------------------------------------
# Biased SVD (FunkSVD / Koren 2009)
# ---------------------------------------------------------------------------

class BiasedSVD:
    """Regularised biased matrix factorisation trained with SGD.

    Minimises (on normalised scores r̃ = r / _NORM):

        L = Σ (r̃_ui − μ̃ − b_u − b_i − p_u·q_i)²
              + λ (‖p_u‖² + ‖q_i‖² + b_u² + b_i²)

    Cold-start fallback: unknown users/items contribute 0 to their term,
    so prediction degrades gracefully to μ + known_bias.
    """

    def __init__(
        self,
        n_factors: int     = 100,
        n_epochs: int      = 60,
        lr: float          = 0.005,
        reg: float         = 0.2,
        patience: int      = 5,
        shrink_item: float = 25.0,
        shrink_user: float = 10.0,
        seed: int          = 42,
    ) -> None:
        self.n_factors   = n_factors
        self.n_epochs    = n_epochs
        self.lr          = lr
        self.reg         = reg
        self.patience    = patience
        self.shrink_item = shrink_item
        self.shrink_user = shrink_user
        self.rng         = np.random.default_rng(seed)
        self.best_epoch_ = n_epochs

    # ------------------------------------------------------------------

    def fit(
        self,
        ratings: List[Rating],
        val_ratings: Optional[List[Rating]] = None,
        n_epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> "BiasedSVD":
        users = sorted({u for u, _, _ in ratings})
        items = sorted({i for _, i, _ in ratings})
        self._uid: Dict[int, int] = {u: k for k, u in enumerate(users)}
        self._iid: Dict[int, int] = {i: k for k, i in enumerate(items)}

        n_u, n_i, f = len(users), len(items), self.n_factors

        # Normalise to [1, 10]; compute mu on same scale
        data: List[Tuple[int, int, float]] = [
            (self._uid[u], self._iid[i], r / _NORM) for u, i, r in ratings
        ]
        self.mu: float = float(np.mean([r for _, _, r in data]))

        # Warm-start biases
        self.bu, self.bi = _shrinkage_biases(
            data, self.mu, self.shrink_item, self.shrink_user, n_u, n_i
        )
        self.P: np.ndarray = self.rng.normal(0.0, 0.1, (n_u, f))
        self.Q: np.ndarray = self.rng.normal(0.0, 0.1, (n_i, f))

        lr, reg = self.lr, self.reg
        order   = np.arange(len(data), dtype=np.int64)
        max_ep  = n_epochs if n_epochs is not None else self.n_epochs

        best_val_rmse = float("inf")
        best_state: Optional[Dict] = None
        no_improve = 0

        for epoch in range(1, max_ep + 1):
            self.rng.shuffle(order)
            sq_err = 0.0
            for k in order:
                u, i, r = data[k]
                err = r - (self.mu + self.bu[u] + self.bi[i]
                           + self.P[u] @ self.Q[i])
                sq_err += err * err

                self.bu[u] += lr * (err - reg * self.bu[u])
                self.bi[i] += lr * (err - reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += lr * (err * self.Q[i] - reg * pu)
                self.Q[i] += lr * (err * pu         - reg * self.Q[i])

            train_rmse = math.sqrt(sq_err / len(data)) * _NORM

            if val_ratings is not None:
                val_pred = self.predict([(u, i) for u, i, _ in val_ratings])
                val_true = np.array([r for _, _, r in val_ratings])
                val_rmse = float(np.sqrt(np.mean((val_true - val_pred) ** 2)))
                if verbose:
                    print(f"  epoch {epoch:>3}/{max_ep}"
                          f"  train={train_rmse:.2f}  val={val_rmse:.2f}")

                if val_rmse < best_val_rmse - 1e-4:
                    best_val_rmse = val_rmse
                    best_state = {
                        "bu": self.bu.copy(), "bi": self.bi.copy(),
                        "P":  self.P.copy(),  "Q":  self.Q.copy(),
                    }
                    self.best_epoch_ = epoch
                    no_improve = 0
                else:
                    no_improve += 1
                    if self.patience > 0 and no_improve >= self.patience:
                        if verbose:
                            print(f"  Early stopping"
                                  f" (best val={best_val_rmse:.2f}"
                                  f" at epoch {self.best_epoch_})")
                        break
            else:
                if verbose:
                    print(f"  epoch {epoch:>3}/{max_ep}  train={train_rmse:.2f}")

        if best_state is not None:
            self.bu = best_state["bu"]; self.bi = best_state["bi"]
            self.P  = best_state["P"];  self.Q  = best_state["Q"]

        return self

    # ------------------------------------------------------------------

    def predict_one(self, uid: int, iid: int) -> float:
        u = self._uid.get(uid)
        i = self._iid.get(iid)
        bu  = self.bu[u] if u is not None else 0.0
        bi  = self.bi[i] if i is not None else 0.0
        dot = (self.P[u] @ self.Q[i]) if (u is not None and i is not None) else 0.0
        score = (self.mu + bu + bi + dot) * _NORM
        return max(SCORE_LO, min(SCORE_HI, score))

    def predict(self, pairs: List[TestPair]) -> np.ndarray:
        return np.array([self.predict_one(u, i) for u, i in pairs])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_val_split(
    ratings: List[Rating],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Rating], List[Rating]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ratings))
    cut = int(len(ratings) * val_ratio)
    val_set = set(idx[:cut].tolist())
    train = [r for k, r in enumerate(ratings) if k not in val_set]
    val   = [ratings[k] for k in idx[:cut]]
    return train, val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lab 2 recommender — BiasedSVD (FunkSVD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train",        default="data/train.txt")
    p.add_argument("--test",         default="data/test.txt")
    p.add_argument("--output",       default="result/prediction.txt")
    p.add_argument("--metrics",      default="result/metrics.json")
    p.add_argument("--n-factors",    type=int,   default=100)
    p.add_argument("--n-epochs",     type=int,   default=60)
    p.add_argument("--lr",           type=float, default=0.005)
    p.add_argument("--reg",          type=float, default=0.2)
    p.add_argument("--patience",     type=int,   default=5)
    p.add_argument("--shrink-item",  type=float, default=25.0)
    p.add_argument("--shrink-user",  type=float, default=10.0)
    p.add_argument("--val-ratio",    type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--quiet",        action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(__file__).parent.parent

    def resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else root / rel

    train_path   = resolve(args.train)
    test_path    = resolve(args.test)
    output_path  = resolve(args.output)
    metrics_path = resolve(args.metrics)

    t0 = time.time()

    print("Loading data …")
    all_ratings = load_train(train_path)
    test_pairs, test_groups = load_test(test_path)
    print(f"  {len(all_ratings):,} training ratings | {len(test_pairs):,} test pairs")

    train_ratings, val_ratings = train_val_split(
        all_ratings, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"  split → train {len(train_ratings):,} | val {len(val_ratings):,}")

    # ------------------------------------------------------------------
    # Phase 1: train on train split, find best epoch via early stopping
    # ------------------------------------------------------------------
    print(f"\nPhase 1 — fit on train split"
          f" (factors={args.n_factors}, lr={args.lr}, reg={args.reg}) …")
    model = BiasedSVD(
        n_factors   = args.n_factors,
        n_epochs    = args.n_epochs,
        lr          = args.lr,
        reg         = args.reg,
        patience    = args.patience,
        shrink_item = args.shrink_item,
        shrink_user = args.shrink_user,
        seed        = args.seed,
    ).fit(train_ratings, val_ratings=val_ratings, verbose=not args.quiet)

    val_pred = model.predict([(u, i) for u, i, _ in val_ratings])
    val_true = np.array([r for _, _, r in val_ratings])
    val_rmse_val = rmse(val_true, val_pred)
    best_ep = model.best_epoch_
    print(f"\nPhase 1 val RMSE = {val_rmse_val:.4f}  (best epoch = {best_ep})")

    # ------------------------------------------------------------------
    # Phase 2: retrain on ALL data for best_epoch epochs
    # val data becomes additional training data — standard practice;
    # no test labels are ever used.
    # ------------------------------------------------------------------
    print(f"\nPhase 2 — retrain on all {len(all_ratings):,} ratings"
          f" for {best_ep} epochs …")
    model.fit(
        all_ratings,
        val_ratings = None,
        n_epochs    = best_ep,
        verbose     = not args.quiet,
    )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    print("\nPredicting …")
    test_item_ids = [i for _, i in test_pairs]
    test_scores   = np.clip(model.predict(test_pairs), SCORE_LO, SCORE_HI)

    write_predictions(output_path, test_groups, test_item_ids, test_scores)
    print(f"Predictions → {output_path}")

    elapsed = round(time.time() - t0, 2)
    metrics: Dict = {
        "algorithm":   "BiasedSVD",
        "n_factors":   args.n_factors,
        "n_epochs":    args.n_epochs,
        "best_epoch":  best_ep,
        "lr":          args.lr,
        "reg":         args.reg,
        "patience":    args.patience,
        "shrink_item": args.shrink_item,
        "shrink_user": args.shrink_user,
        "seed":        args.seed,
        "val_ratio":   args.val_ratio,
        "val_rmse":    val_rmse_val,
        "n_train":     len(train_ratings),
        "n_val":       len(val_ratings),
        "n_test":      len(test_pairs),
        "elapsed_sec": elapsed,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Metrics     → {metrics_path}")
    print(f"Total time: {elapsed}s")


if __name__ == "__main__":
    main()
