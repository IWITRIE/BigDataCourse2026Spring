#!/usr/bin/env python3
"""
Compile every solution variant against the OFFICIAL judge runner
(judge/runner/cpp/main.cpp) with the OFFICIAL flags and parse its JSON.
Produces results.json for the report's tables/plots.

Variants (all derived from the real src/solution.cpp by safe edits):
  final           : the submitted solution (UD=16)
  ud{K}           : UD sweep K in {4,8,16,32,64,128,256,512,1024}
  nobias          : user_scale=item_scale=0   (RMSE without the bias model)
  biasonly        : factor_lr=0               (RMSE without factor SGD)
  noprefetch      : remove __builtin_prefetch block
  nosimd          : remove GCC target/optimize + omp-simd pragmas
"""
from __future__ import annotations
import json, subprocess, sys, tempfile, shutil, time
from pathlib import Path

REPO   = Path("/mnt/d/HomeWork/BigData/lab2/task2")
RUNNER = REPO / "judge/runner/cpp/main.cpp"
TEMPL  = (REPO / "src/solution.cpp").read_text()
DATA   = Path("/tmp/claude-0/-mnt-d-HomeWork-BigData/6713f926-7f5a-4402-b950-8e067ef03d02"
              "/scratchpad/task2data/secure_data_full_1024/judge_data.bin")
OUT    = Path(__file__).resolve().parent / "results.json"
ROUNDS = 10
FLAGS  = ["-O3", "-std=c++17", "-march=native", "-fopenmp"]

PREFETCH_BLOCK = """            if (idx + 32 < Nbatch) {
                const Rating& nr = batch[idx + 32];
                if ((unsigned)nr.user < U) __builtin_prefetch(Pc + nr.user * UD, 1, 1);
                if ((unsigned)nr.item < I) __builtin_prefetch(Qc + nr.item * UD, 1, 1);
            }
"""

def must_replace(src: str, old: str, new: str) -> str:
    assert old in src, f"pattern not found: {old[:60]!r}"
    return src.replace(old, new)

def make_variant(name: str) -> str:
    s = TEMPL
    if name == "final":
        return s
    if name.startswith("ud"):
        k = int(name[2:])
        return must_replace(s, "static constexpr int UD = 16;",
                            f"static constexpr int UD = {k};")
    if name == "nobias":
        s = must_replace(s, "static constexpr float user_scale = 0.87f;",
                         "static constexpr float user_scale = 0.0f;")
        s = must_replace(s, "static constexpr float item_scale = 0.95f;",
                         "static constexpr float item_scale = 0.0f;")
        return s
    if name == "biasonly":
        return must_replace(s, "static constexpr float factor_lr = 0.050f;",
                            "static constexpr float factor_lr = 0.0f;")
    if name == "noprefetch":
        return must_replace(s, PREFETCH_BLOCK, "")
    if name == "nosimd":
        for pat in ['#pragma GCC optimize("O3,unroll-loops")\n',
                    '#pragma GCC target("avx2,bmi,bmi2,popcnt,fma")\n',
                    '#pragma omp simd reduction(+:pred)\n',
                    '#pragma omp simd reduction(+:score)\n',
                    '#pragma omp simd\n']:
            s = s.replace(pat, "")
        return s
    raise ValueError(name)

def bench(name: str) -> dict:
    src = make_variant(name)
    with tempfile.TemporaryDirectory(prefix=f"svd_{name}_") as td:
        td = Path(td)
        (td / "solution.cpp").write_text(src)
        shutil.copy2(RUNNER, td / "main.cpp")
        exe = td / "run"
        c = subprocess.run(["g++", *FLAGS, "main.cpp", "-o", str(exe)],
                           cwd=td, capture_output=True, text=True, timeout=120)
        if c.returncode != 0:
            return {"variant": name, "error": "compile: " + c.stderr[-400:]}
        r = subprocess.run([str(exe), str(DATA), "0.001", str(ROUNDS)],
                           cwd=td, capture_output=True, text=True, timeout=900)
        line = [l for l in r.stdout.splitlines() if l.strip()][-1]
        p = json.loads(line)
        if p.get("status") != "success":
            return {"variant": name, "error": p.get("error", "run failed")}
        runs = [float(x) for x in p["time_runs"]]
        return {"variant": name,
                "time_runs": runs,
                "time_first": runs[0],          # the only round that actually trains
                "time_total": float(p["time_sec"]),
                "rmse_base": float(p["rmse_base"]),
                "rmse": float(p["rmse"]),
                "improvement": float(p["rmse_base"]) - float(p["rmse"]),
                "valid": bool(p["valid"])}

VARIANTS = (["final"] +
            [f"ud{k}" for k in (4, 8, 16, 32, 64, 128, 256, 512, 1024)] +
            ["nobias", "biasonly", "noprefetch", "nosimd"])

def main():
    results = {}
    for name in VARIANTS:
        t0 = time.perf_counter()
        try:
            res = bench(name)
        except Exception as e:
            res = {"variant": name, "error": repr(e)}
        results[name] = res
        wall = time.perf_counter() - t0
        if "error" in res:
            print(f"[{name:10s}] ERROR {res['error'][:80]}", flush=True)
        else:
            print(f"[{name:10s}] first={res['time_first']:.3f}s total={res['time_total']:.3f}s "
                  f"rmse {res['rmse_base']:.4f}->{res['rmse']:.4f} "
                  f"d={res['improvement']:+.4f} valid={res['valid']} ({wall:.0f}s)", flush=True)
        OUT.write_text(json.dumps(results, indent=2))
    print("DONE ->", OUT, flush=True)

if __name__ == "__main__":
    main()
