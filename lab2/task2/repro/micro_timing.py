#!/usr/bin/env python3
"""Compile each micro variant ONCE, run the exe N times, report median round-1 time."""
import json, subprocess, tempfile, shutil, statistics as st
from pathlib import Path
import run_experiments as R

REPS = 11
OUT = Path(__file__).resolve().parent / "micro_timing.json"
res = {}
for name in ["final", "noprefetch", "nosimd", "biasonly", "nobias"]:
    src = R.make_variant(name)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "solution.cpp").write_text(src)
        shutil.copy2(R.RUNNER, td / "main.cpp")
        exe = td / "run"
        subprocess.run(["g++", *R.FLAGS, "main.cpp", "-o", str(exe)],
                       cwd=td, check=True, capture_output=True, timeout=120)
        firsts = []
        rmse = rmse_base = None
        for _ in range(REPS):
            r = subprocess.run([str(exe), str(R.DATA), "0.001", "10"],
                               cwd=td, capture_output=True, text=True, timeout=300)
            p = json.loads([l for l in r.stdout.splitlines() if l.strip()][-1])
            firsts.append(float(p["time_runs"][0]))
            rmse, rmse_base = float(p["rmse"]), float(p["rmse_base"])
        firsts.sort()
        res[name] = dict(median_ms=st.median(firsts) * 1000,
                         min_ms=min(firsts) * 1000,
                         rmse=rmse, rmse_base=rmse_base,
                         samples_ms=[round(x * 1000, 2) for x in firsts])
        print(f"{name:11s} median={res[name]['median_ms']:6.2f}ms "
              f"min={res[name]['min_ms']:6.2f}ms rmse={rmse:.4f}", flush=True)
OUT.write_text(json.dumps(res, indent=2))
print("saved", OUT)
