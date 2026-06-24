#!/usr/bin/env python3
"""
Reconstruct the Track-1 incremental-SVD judge data from the real MovieLens-20M
ratings, faithfully matching meta.json (138493 users, 26744 items, 20,000,263
ratings split 16M / 2M / 2M).

Pipeline
--------
1. Load ratings.csv (userId, movieId, rating, timestamp).
2. Factorize userId / movieId -> contiguous 0-based ids (assert counts).
3. TEMPORAL split by timestamp: oldest 16,000,210 -> base ("已训练"),
   next 2,000,026 -> incremental (streamed into update()),
   newest 2,000,027 -> test (held-out RMSE).  This mirrors a real online
   recommender: the model is pre-trained on the past, new ratings arrive,
   we are scored on the future.
4. global_mean = mean(base ratings).
5. Base model: truncated SVD of the centred base rating matrix via
   randomized_svd.  P = U*sqrt(S), Q = V*sqrt(S)  =>  P @ Q.T ~ R - mu.
   Rank K is chosen adaptively to fit RAM, then laid into the 1024-wide
   layout declared by meta.json (first K dims real, rest zero).  The
   incremental SGD only ever touches the first 16 dims, which are the top
   singular components -- the part that matters is fully real.
6. Dump P.npy, Q.npy, global_mean.npy, incremental.npy, test.npy, meta.json
   and pack judge_data.bin in the runner's binary layout.
"""
from __future__ import annotations
import json, struct, sys, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

HERE = Path(__file__).resolve().parent
OUT  = HERE / "secure_data_full_1024"
OUT.mkdir(exist_ok=True)
LATENT_DIM = 1024
RANK_CANDIDATES = [1024, 768, 512, 384, 256]   # try largest that fits RAM
N_BASE, N_INC, N_TEST = 16_000_210, 2_000_026, 2_000_027
SEED = 42

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ---------------------------------------------------------------- load
log("reading ratings.csv ...")
df = pd.read_csv(
    HERE / "ratings.csv",
    usecols=["userId", "movieId", "rating", "timestamp"],
    dtype={"userId": "int32", "movieId": "int32",
           "rating": "float32", "timestamp": "int64"},
)
log("rows:", len(df))
assert len(df) == N_BASE + N_INC + N_TEST, f"unexpected row count {len(df)}"

# ---------------------------------------------------------------- remap ids
u_codes, _ = pd.factorize(df["userId"].to_numpy(), sort=True)
i_codes, _ = pd.factorize(df["movieId"].to_numpy(), sort=True)
num_users = int(u_codes.max()) + 1
num_items = int(i_codes.max()) + 1
log("num_users:", num_users, "num_items:", num_items)
assert num_users == 138493, num_users
assert num_items == 26744,  num_items

u_codes = u_codes.astype(np.int32)
i_codes = i_codes.astype(np.int32)
rating  = df["rating"].to_numpy(np.float32)
ts      = df["timestamp"].to_numpy(np.int64)
del df; gc.collect()

# ---------------------------------------------------------------- temporal split
log("stable temporal sort ...")
order = np.argsort(ts, kind="stable")          # oldest -> newest
del ts; gc.collect()
u_codes, i_codes, rating = u_codes[order], i_codes[order], rating[order]
del order; gc.collect()

base = slice(0, N_BASE)
inc  = slice(N_BASE, N_BASE + N_INC)
test = slice(N_BASE + N_INC, N_BASE + N_INC + N_TEST)

global_mean = float(rating[base].mean())
log(f"global_mean (base) = {global_mean:.10f}")

incremental = np.stack(
    [u_codes[inc].astype(np.float32), i_codes[inc].astype(np.float32), rating[inc]], axis=1)
testset = np.stack(
    [u_codes[test].astype(np.float32), i_codes[test].astype(np.float32), rating[test]], axis=1)
np.save(OUT / "incremental.npy", incremental)
np.save(OUT / "test.npy", testset)
log("saved incremental.npy", incremental.shape, "test.npy", testset.shape)

# ---------------------------------------------------------------- base matrix
from scipy.sparse import csr_matrix
log("building centred base CSR ...")
R = csr_matrix(
    (rating[base] - global_mean,
     (u_codes[base], i_codes[base])),
    shape=(num_users, num_items), dtype=np.float32)
R.sum_duplicates()
log("nnz:", R.nnz)
del u_codes, i_codes, rating; gc.collect()

# ---------------------------------------------------------------- truncated SVD
U = S = Vt = None
for K in RANK_CANDIDATES:
    try:
        log(f"randomized_svd k={K} (n_iter=4) ...")
        t0 = time.perf_counter()
        U, S, Vt = randomized_svd(R, n_components=K, n_iter=4,
                                  power_iteration_normalizer="QR",
                                  random_state=SEED)
        log(f"  ok in {time.perf_counter()-t0:.1f}s  S[:3]={S[:3]}  S[-1]={S[-1]:.3f}")
        used_k = K
        break
    except MemoryError:
        log(f"  MemoryError at k={K}, trying smaller")
        U = S = Vt = None; gc.collect()
if U is None:
    log("FATAL: could not build SVD at any rank"); sys.exit(1)

sqrtS = np.sqrt(S).astype(np.float32)
P = np.zeros((num_users, LATENT_DIM), dtype=np.float32)
Q = np.zeros((num_items, LATENT_DIM), dtype=np.float32)
P[:, :used_k] = (U * sqrtS).astype(np.float32)
Q[:, :used_k] = (Vt.T * sqrtS).astype(np.float32)
del U, S, Vt, R; gc.collect()
log(f"P {P.shape} Q {Q.shape}  (real rank {used_k}, padded to {LATENT_DIM})")

np.save(OUT / "P.npy", P)
np.save(OUT / "Q.npy", Q)
np.save(OUT / "global_mean.npy", np.float32(global_mean))

meta = dict(num_users=num_users, num_items=num_items, latent_dim=LATENT_DIM,
            trainer="randomized_svd", real_rank=used_k,
            global_mean=global_mean, base_rows=N_BASE,
            incremental_rows=N_INC, test_rows=N_TEST, batch_size=100_000,
            split="temporal", seed=SEED, format_version=1)
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
log("saved meta.json", meta)

# ---------------------------------------------------------------- pack judge_data.bin
log("packing judge_data.bin ...")
hdr = struct.pack("<8s6i3f", b"SVDJUDGE", 1, num_users, num_items, LATENT_DIM,
                  N_INC, N_TEST, global_mean, 0.0, 0.0)
assert len(hdr) == 44, len(hdr)

rating_dt = np.dtype([("user", "<i4"), ("item", "<i4"), ("rating", "<f4")])
inc_rec = np.empty(N_INC, dtype=rating_dt)
inc_rec["user"] = incremental[:, 0].astype(np.int32)
inc_rec["item"] = incremental[:, 1].astype(np.int32)
inc_rec["rating"] = incremental[:, 2]
test_rec = np.empty(N_TEST, dtype=rating_dt)
test_rec["user"] = testset[:, 0].astype(np.int32)
test_rec["item"] = testset[:, 1].astype(np.int32)
test_rec["rating"] = testset[:, 2]

with open(OUT / "judge_data.bin", "wb") as f:
    f.write(hdr)
    f.write(np.ascontiguousarray(P).tobytes())
    f.write(np.ascontiguousarray(Q).tobytes())
    f.write(inc_rec.tobytes())
    f.write(test_rec.tobytes())
log("judge_data.bin bytes:", (OUT / "judge_data.bin").stat().st_size)
log("DONE")
