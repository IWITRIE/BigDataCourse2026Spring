# Lab 1: PageRank

本目录用于存放 NKU 大数据课程第一次实验 PageRank 相关材料。

## 实验要求摘要

- 在给定数据集 `Data.txt` 上计算 PageRank 分数。
- 输入格式为每行一条有向边：`FromNodeID ToNodeID`。
- 至少需要报告 teleport parameter 为 `0.85` 时 PageRank 最高的 100 个节点。
- 程序需要处理 dead-ends 和 spider-traps，并迭代至收敛。
- 禁止直接调用现成 PageRank API。
- 必须使用 Sparse Matrix 和 Block Matrix 等方式优化内存使用。
- 输出结果文件命名为 `Res.txt`，格式为：`NodeID Score`。
- 参考约束：运行时间低于 60 秒，最大内存使用低于 80 MB。
- 截止时间：2026-05-15 12:00 (UTC+8)。

## 目录结构

- `Data.txt`: 实验数据集。
- `docs/requirement.pdf`: 实验要求原文。
- `examples/`: C/C++ 与 Python 示例提交结构。
- `tools/`: 内存使用统计脚本。
