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
- `src/`: PageRank C++ 实现与编译参数。
- `result/Res.txt`: teleport parameter 为 `0.85` 时的 Top 100 PageRank 结果。
- `bin/PageRank.exe`: 静态链接后的 Linux x86-64 可执行文件。
- `tools/`: 内存使用统计脚本。

## 实现说明

当前实现位于 `src/PageRank.cpp`，使用 C++17 编写，主要设计如下：

- 按 PageRank 常用约定，将实验要求中的 teleport parameter `0.85` 作为 damping factor 使用。
- 使用 CSR 结构存储稀疏有向图，避免构造稠密邻接矩阵。
- 按源节点区间分块遍历 CSR 边表，体现 Block Matrix 的分块计算思想。
- 使用 `double` 保存 PageRank 分数。
- 每轮迭代显式统计 dead-end 节点的 PageRank 总量，并平均分配回所有节点。
- 每轮归一化 PageRank 向量，降低浮点误差累积。
- 使用 L1 residual 判断收敛，阈值为 `1e-12`。

## 编译与运行

```bash
g++ -O2 -std=c++17 -static lab1/src/PageRank.cpp -o lab1/bin/PageRank.exe
lab1/bin/PageRank.exe lab1/Data.txt lab1/result/Res.txt
```

本地测试结果：

- 节点数：9500
- 边数：150000
- 迭代次数：17
- L1 residual：约 `8.06e-13`
- 最大内存：约 7.5 MB
- 运行时间：低于 1 秒
