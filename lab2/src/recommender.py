#!/usr/bin/env python3
"""Lab 2: Explicit rating prediction via Biased Matrix Factorisation (SVD).

Algorithm: Regularised biased matrix factorisation (FunkSVD / Koren 2009),
trained with stochastic gradient descent.

  r_ui ≈ μ + b_u + b_i + p_u · q_i

Scores (10-100) are normalised to [1, 10] before training so that
standard SGD hyper-parameters apply; predictions are de-normalised for output.

References
----------
Koren, Bell & Volinsky, "Matrix Factorization Techniques for Recommender
Systems", IEEE Computer 42(8), 2009.

Hug, "Surprise: A Python library for recommender systems", JOSS 5(52), 2020
— algorithm structure and default hyper-parameters.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
Rating = Tuple[int, int, float]   # (user_id, item_id, score)
TestPair = Tuple[int, int]        # (user_id, item_id)
GroupInfo = Tuple[int, int, int]  # (user_id, start_index, count)

SCORE_LO, SCORE_HI = 10.0, 100.0
_NORM = 10.0  # divide raw scores by this to get [1, 10] range


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_train(path: Path) -> List[Rating]:
    """Parse train.txt → list of (user_id, item_id, score) triples."""
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
    """Parse test.txt.

    Returns
    -------
    pairs  : ordered (user_id, item_id) list whose scores must be predicted
    groups : (user_id, start_index, count) triples for re-grouping output
    """
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
    """Write predictions in the ResultForm.txt format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for uid, start, count in groups:
            fh.write(f"{uid}|{count}\n")
            for k in range(start, start + count):
                fh.write(f"{item_ids[k]} {scores[k]:.6f}\n")


# ---------------------------------------------------------------------------
# Biased Matrix Factorisation (FunkSVD / Koren 2009)
# ---------------------------------------------------------------------------

class BiasedSVD:
    """Regularised biased matrix factorisation trained with SGD.

    Minimises (on normalised scores r̃ = r / _NORM)::

        L = Σ (r̃_ui − μ̃ − b_u − b_i − p_u · q_i)²
              + λ (||p_u||² + ||q_i||² + b_u² + b_i²)

    Parameters
    ----------
    n_factors : latent dimension.  Recommended: much smaller than n_users
                to avoid overfitting on sparse item ratings.
    patience  : early-stopping patience (epochs without val improvement).
                Requires val_ratings to be passed to fit().
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 50,
        lr: float = 0.005,
        reg: float = 0.2,
        patience: int = 5,
        seed: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.patience = patience
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def fit(
        self,
        ratings: List[Rating],
        val_ratings: Optional[List[Rating]] = None,
        verbose: bool = True,
    ) -> "BiasedSVD":
        users = sorted({u for u, _, _ in ratings})
        items = sorted({i for _, i, _ in ratings})
        self._uid: Dict[int, int] = {u: k for k, u in enumerate(users)}
        self._iid: Dict[int, int] = {i: k for k, i in enumerate(items)}

        n_u, n_i, f = len(users), len(items), self.n_factors

        # Normalise scores to [1, 10] — mu is on the same normalised scale
        data: List[Tuple[int, int, float]] = [
            (self._uid[u], self._iid[i], r / _NORM) for u, i, r in ratings
        ]
        self.mu: float = float(np.mean([r for _, _, r in data]))
        self.bu: np.ndarray = np.zeros(n_u)
        self.bi: np.ndarray = np.zeros(n_i)
        self.P: np.ndarray = self.rng.normal(0.0, 0.1, (n_u, f))
        self.Q: np.ndarray = self.rng.normal(0.0, 0.1, (n_i, f))

        lr, reg = self.lr, self.reg
        order = np.arange(len(data), dtype=np.int64)

        best_val_rmse = float("inf")
        best_state: Optional[dict] = None
        no_improve = 0

        for epoch in range(1, self.n_epochs + 1):
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

            # Report train RMSE on original 10-100 scale
            train_rmse = math.sqrt(sq_err / len(data)) * _NORM

            if val_ratings:
                val_pred = self.predict([(u, i) for u, i, _ in val_ratings])
                val_true = np.array([r for _, _, r in val_ratings])
                val_rmse = float(np.sqrt(np.mean((val_true - val_pred) ** 2)))
                if verbose:
                    print(f"  epoch {epoch:>3}/{self.n_epochs}"
                          f"  train={train_rmse:.2f}  val={val_rmse:.2f}")

                if val_rmse < best_val_rmse - 1e-4:
                    best_val_rmse = val_rmse
                    best_state = {
                        "mu": self.mu,
                        "bu": self.bu.copy(),
                        "bi": self.bi.copy(),
                        "P": self.P.copy(),
                        "Q": self.Q.copy(),
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch}"
                                  f" (best val RMSE={best_val_rmse:.2f})")
                        break
            else:
                if verbose:
                    print(f"  epoch {epoch:>3}/{self.n_epochs}"
                          f"  train={train_rmse:.2f}")

        # Restore the best checkpoint when early stopping is active
        if best_state is not None:
            self.mu = best_state["mu"]
            self.bu = best_state["bu"]
            self.bi = best_state["bi"]
            self.P  = best_state["P"]
            self.Q  = best_state["Q"]

        return self

    # ------------------------------------------------------------------

    def predict_one(self, uid: int, iid: int) -> float:
        u = self._uid.get(uid)
        i = self._iid.get(iid)
        bu = self.bu[u] if u is not None else 0.0
        bi = self.bi[i] if i is not None else 0.0
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
    val = [ratings[k] for k in idx[:cut]]
    return train, val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lab 2 recommender — Biased SVD (FunkSVD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train",     default="data/train.txt")
    p.add_argument("--test",      default="data/test.txt")
    p.add_argument("--output",    default="result/prediction.txt")
    p.add_argument("--metrics",   default="result/metrics.json")
    p.add_argument("--n-factors", type=int,   default=100)
    p.add_argument("--n-epochs",  type=int,   default=50)
    p.add_argument("--lr",        type=float, default=0.005)
    p.add_argument("--reg",       type=float, default=0.2)
    p.add_argument("--patience",  type=int,   default=5,
                   help="early-stopping patience in epochs (0 to disable)")
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="fraction of train held out for validation; 0 to disable")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(__file__).parent.parent  # lab2/

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

    val_ratings: List[Rating] = []
    train_ratings = all_ratings
    if args.val_ratio > 0:
        train_ratings, val_ratings = train_val_split(
            all_ratings, val_ratio=args.val_ratio, seed=args.seed
        )
        print(f"  split → train {len(train_ratings):,} | val {len(val_ratings):,}")

    print(
        f"\nFitting BiasedSVD  "
        f"(factors={args.n_factors}, epochs={args.n_epochs}, "
        f"lr={args.lr}, reg={args.reg}) …"
    )
    model = BiasedSVD(
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        lr=args.lr,
        reg=args.reg,
        patience=args.patience,
        seed=args.seed,
    ).fit(
        train_ratings,
        val_ratings=val_ratings or None,
        verbose=not args.quiet,
    )

    val_rmse_val: Optional[float] = None
    if val_ratings:
        val_pred = model.predict([(u, i) for u, i, _ in val_ratings])
        val_true = np.array([r for _, _, r in val_ratings])
        val_rmse_val = rmse(val_true, val_pred)
        print(f"\nFinal validation RMSE = {val_rmse_val:.4f}")

    print("\nPredicting test pairs …")
    test_item_ids = [i for _, i in test_pairs]
    test_scores   = model.predict(test_pairs)

    write_predictions(output_path, test_groups, test_item_ids, test_scores)
    print(f"Predictions → {output_path}")

    elapsed = round(time.time() - t0, 2)
    metrics: Dict = {
        "algorithm": "BiasedSVD",
        "n_factors": args.n_factors,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "reg": args.reg,
        "patience": args.patience,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "val_rmse": val_rmse_val,
        "n_train": len(train_ratings),
        "n_val": len(val_ratings),
        "n_test": len(test_pairs),
        "elapsed_sec": elapsed,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Metrics     → {metrics_path}")
    print(f"Total time: {elapsed}s")


if __name__ == "__main__":
    main()
