# Lab 1: PageRank 实验报告

本目录用于存放 NKU 大数据课程第一次实验 PageRank 相关材料。实验代码位于 `src/PageRank.cpp`，使用 C++17 实现，未调用任何现成 PageRank API。

## 一、实验要求

本实验要求在给定有向图数据集 `Data.txt` 上计算 PageRank 分数，并输出 teleport parameter 为 `0.85` 时 PageRank 最高的 100 个节点。输入文件每一行表示一条有向边，格式为：

```text
FromNodeID ToNodeID
```

程序需要正确处理 dead-ends 和 spider-traps，迭代至收敛，并将结果写入 `Res.txt`，输出格式为：

```text
NodeID Score
```

实验同时要求使用 Sparse Matrix、Block Matrix 等方式优化内存使用，参考约束为运行时间低于 60 秒、最大内存使用低于 80 MB。

## 二、目录结构

- `Data.txt`: 实验数据集。
- `src/PageRank.cpp`: PageRank C++ 源码。
- `src/compile-parameter.txt`: 编译参数说明。
- `result/Res.txt`: teleport parameter 为 `0.85` 时的 Top 100 结果。
- `bin/PageRank.exe`: 静态链接后的 Linux x86-64 可执行文件。
- `tools/`: 内存和运行时间统计脚本。
- `examples/`: 课程提供的 C/C++ 与 Python 示例提交结构。

## 三、算法原理

本实现按 PageRank 常用定义，将实验要求中的 teleport parameter `0.85` 作为 damping factor，记为 `d`。设图中共有 `N` 个节点，节点 `v` 在第 `t + 1` 轮的 PageRank 更新公式为：

```text
PR_{t+1}(v) = (1 - d) / N
              + d * dead_sum / N
              + d * sum(PR_t(u) / out_degree(u)), u -> v
```

其中 `dead_sum` 表示当前轮所有 dead-end 节点的 PageRank 总和。对于没有出边的节点，其 PageRank 会在下一轮平均分配给所有节点，从而避免分数泄漏。teleport 部分 `(1 - d) / N` 保证随机游走可以跳转到任意节点，因此 spider-traps 不会永久吸收全部 PageRank。

初始时所有节点 PageRank 均为 `1 / N`。每轮迭代后计算 L1 residual：

```text
residual = sum(abs(PR_{t+1}(i) - PR_t(i)))
```

当 residual 小于 `1e-12` 时认为收敛；若未提前收敛，则最多迭代 `1000` 轮。

## 四、实现设计

### 4.1 节点压缩

原始节点 ID 不一定连续，直接使用原始 ID 建数组会浪费大量空间。因此程序首先扫描 `Data.txt`，收集所有起点和终点 ID，排序去重后得到压缩节点表 `node_ids`，再把原始节点 ID 映射到 `[0, N)` 范围内的连续下标。

为了降低读图阶段内存占用，程序采用两遍扫描：

1. 第一遍只收集节点 ID，并统计原始边数。
2. 第二遍根据压缩下标生成边表。

映射过程由 `NodeIndexer` 完成。当前数据集的节点 ID 范围较紧凑时，程序使用 dense lookup 数组进行 O(1) 查询；若换成 ID 范围很大且稀疏的数据，则自动回退到在有序 `node_ids` 上二分查找，避免建立过大的映射数组。

### 4.2 稀疏图存储

程序没有构造 `N * N` 的稠密邻接矩阵，而是使用 CSR（Compressed Sparse Row）存储有向图：

- `row_ptr[i]` 和 `row_ptr[i + 1]` 表示节点 `i` 的出边在 `col_idx` 中的区间。
- `col_idx` 连续保存每条出边的目标节点下标。
- `out_weight[i]` 预先保存非 dead-end 节点每条出边的分配系数 `d / out_degree(i)`。
- `dead_nodes` 单独保存所有出度为 0 的节点。

压缩边使用 64 位整数打包为 `(from, to)`，排序后去重，再直接生成 CSR。这样可以去除重复边，同时降低边排序阶段的比较开销。

### 4.3 分块迭代

PageRank 迭代阶段按源节点区间进行分块遍历，块大小为 `1024`。每个块内部只访问该区间源节点的 CSR 出边，等价于按稀疏矩阵的行块进行矩阵向量乘法，体现 Block Matrix 的计算思想。

每轮迭代主要步骤如下：

1. 累加 `dead_nodes` 中所有节点的 PageRank，计算 dead-end 平均贡献。
2. 使用 teleport 贡献和 dead-end 贡献初始化 `next_rank`。
3. 按块遍历 CSR 出边，将源节点 PageRank 按 `out_weight` 分配给目标节点。
4. 对 `next_rank` 做归一化，修正浮点误差带来的总和偏移。
5. 在同一次遍历中计算 L1 residual，并判断是否收敛。

### 4.4 Top 100 输出

收敛后程序只需要输出 PageRank 最高的 100 个节点，因此使用 `partial_sort` 获取 Top 100，而不是对全部节点完整排序。当两个节点 PageRank 相同时，按照原始节点 ID 升序输出，保证结果稳定。

## 五、优化策略

当前版本针对内存和时间进行了以下优化：

- 使用两遍读图替代长期保存原始边缓存，降低峰值内存。
- 删除 `unordered_map`，改用 dense lookup 或二分查找完成节点压缩，减少哈希表额外开销。
- 使用 64 位整数打包压缩边，减少 `pair<int, int>` 排序时的对象开销。
- 使用 CSR 存储稀疏图，只保存真实存在的边。
- 预计算 `out_weight`，避免迭代中重复计算出度和除法。
- 单独保存 `dead_nodes`，避免每轮扫描所有节点判断是否为 dead-end。
- 将归一化和 residual 计算合并到一次遍历中，减少对 PageRank 向量的访问次数。
- 使用 `partial_sort` 只排序 Top 100，降低结果输出阶段时间开销。

## 六、复杂度分析

设节点数为 `N`，去重后的边数为 `M`。

- 节点 ID 排序去重复杂度为 `O(M log M)`，因为每条边会贡献两个节点 ID。
- 边压缩后排序去重复杂度为 `O(M log M)`。
- CSR 存储空间复杂度为 `O(N + M)`。
- 每轮 PageRank 迭代需要遍历 dead-end 节点和所有出边，复杂度为 `O(N_dead + M)`。
- Top 100 输出使用 `partial_sort`，复杂度约为 `O(N log 100)`。

由于实验数据规模为 `9500` 个节点、`150000` 条边，CSR 和分块遍历可以使内存使用远低于 80 MB 限制。

## 七、编译与运行

编译命令如下：

```bash
g++ -O2 -std=c++17 -static lab1/src/PageRank.cpp -o lab1/bin/PageRank.exe
```

运行命令如下：

```bash
lab1/bin/PageRank.exe lab1/Data.txt lab1/result/Res.txt
```

如果在提交目录内编译，也可以参考 `src/compile-parameter.txt`：

```bash
g++ -O2 -std=c++17 -static PageRank.cpp -o PageRank.exe
```

## 八、实验结果

Linux x86-64 静态可执行文件测试结果如下：

| 指标 | 结果 |
| --- | --- |
| 节点数 | 9500 |
| 去重后边数 | 150000 |
| 迭代次数 | 17 |
| 最终 L1 residual | 约 `8.06e-13` |
| 峰值 RSS | 约 10.61 MB |
| 运行时间 | 约 0.16 秒 |

`result/Res.txt` 中 PageRank 最高的前 10 个节点如下：

| 排名 | NodeID | Score |
| --- | --- | --- |
| 1 | 75 | 0.000198721357091 |
| 2 | 8686 | 0.000190205478321 |
| 3 | 9678 | 0.000188337048374 |
| 4 | 5104 | 0.00018779534425 |
| 5 | 725 | 0.000186793163631 |
| 6 | 3257 | 0.000185721642407 |
| 7 | 468 | 0.000184900195771 |
| 8 | 7730 | 0.000183938392007 |
| 9 | 7175 | 0.000183838928246 |
| 10 | 5526 | 0.000182843425465 |

完整 Top 100 结果见 `result/Res.txt`。

## 九、实验总结

本实验实现了基于稀疏图结构的 PageRank 计算。通过 CSR 存储、节点压缩、dead-end 显式处理和分块迭代，程序在保证收敛精度的同时将内存控制在较低水平。优化后的实现避免了稠密矩阵和大规模哈希表带来的额外开销，在当前数据集上 17 轮即可达到 `1e-12` 级别的 L1 residual，运行时间和峰值内存均满足实验要求。
