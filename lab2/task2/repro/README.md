# Task2 评测数据复原与实验复现

官方隐藏了真实 judge 数据（仓库仅有 `meta.json`）。本目录用**真实 MovieLens-20M**
复原整条评测流水线，并用**官方 runner** 跑出报告中的全部定量结果。

## 流水线

1. **`build_base.py`** — 下载/读取 ML-20M `ratings.csv`，重映射 ID、按**时间戳**切分
   16M/2M/2M（与 `meta.json` 逐项对齐，复原出的 `μ=3.5126163960` 与官方逐位一致），
   用 `randomized_svd`（秩 1024）复原基础模型 `P,Q`，打包 `judge_data.bin`
   （runner 二进制格式）及 `P/Q/incremental/test.npy`。
2. **`run_experiments.py`** — 由 `src/solution.cpp` 派生所有消融变体（UD 扫描 4→1024、
   nobias、biasonly、noprefetch、nosimd），每个变体用官方 `judge/runner/cpp/main.cpp`
   以官方编译选项编译运行，解析其 JSON → `results.json`。
3. **`micro_timing.py`** — 对关键变体各编译一次、运行 11 次取中位数 → `micro_timing.json`
   （稳定的微优化耗时）。
4. **`report/generate_plots.py`** — 读取上述 JSON 与奇异谱，生成报告的 3 张论文级图。

## 复现

```bash
# 1) 准备数据（需 ratings.csv，或脚本内的下载逻辑）
python3 build_base.py
# 2) 主实验 + 消融
python3 run_experiments.py
python3 micro_timing.py
# 3) 出图
python3 ../report/generate_plots.py
```

> 注：脚本内的数据路径指向运行时的 scratch 目录；迁移时改 `DATA`/`OUT` 即可。
> 所有耗时在 Intel i7-13620H（8 核）实测，仅反映优化效果（与 OJ 的 e5-2696v3 不同）。
> `results.json` / `micro_timing.json` 为本次运行的真实输出，随附以便核对。
