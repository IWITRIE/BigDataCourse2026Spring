# Lab 2: 推荐系统任务一

本目录用于存放 NKU 大数据课程第二次实验推荐系统任务一相关材料。

## 一、任务说明

本组选择任务一。任务要求根据训练集中的用户评分记录，预测测试集给出的用户-物品对的评分，并按照指定格式提交预测结果。

训练集、测试集和结果格式说明均来自任务一原始材料：

- `data/train.txt`: 训练数据，用于训练推荐模型。
- `data/test.txt`: 测试数据，需要预测其中用户-物品对的评分。
- `data/DataFormatExplanation.txt`: 数据格式说明。
- `data/ResultForm.txt`: 结果文件格式示例。
- `docs/requirement.docx`: 任务一作业要求。

## 二、目录结构

- `data/`: 任务一数据集和格式说明。
- `docs/`: 任务一要求文档。
- `src/`: 推荐算法源码目录。
- `bin/`: 可执行文件目录。
- `result/`: 实验输出结果目录。
- `report/`: 实验报告目录。
- `examples/`: 任务一提交内容结构示例。

## 三、数据格式

训练集格式为：

```text
<user id>|<numbers of rating items>
<item id>   <score>
```

测试集格式为：

```text
<user id>|<numbers of rating items>
<item id>
```

结果文件需要按照 `ResultForm.txt` 中的格式输出预测评分。

## 四、Conda 环境与运行

本实验使用 `numpy`、`scipy`、`scikit-learn`、`pandas` 库。创建环境：

```bash
conda env create -f lab2/environment.yml
conda activate bigdata-lab2
```

运行推荐器（从仓库根目录执行）：

```bash
lab2/bin/run_recommender.sh
```

脚本自动使用 `bigdata-lab2` conda 环境。如需覆盖 Python 路径：

```bash
BIGDATA_LAB2_PYTHON=/path/to/python lab2/bin/run_recommender.sh
```

支持的命令行参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n-factors` | 100 | 隐向量维度 |
| `--n-epochs` | 50 | 最大训练轮次 |
| `--lr` | 0.005 | SGD 学习率 |
| `--reg` | 0.2 | L2 正则化系数 |
| `--patience` | 5 | 早停耐心（0 禁用）|
| `--val-ratio` | 0.1 | 验证集比例（0 禁用）|
| `--seed` | 42 | 随机种子 |
| `--quiet` | — | 不打印每轮指标 |

输出文件：

- `result/prediction.txt`：按 ResultForm 格式输出的预测评分。
- `result/metrics.json`：验证集 RMSE、超参数和运行时间。

## 五、算法说明

采用**带偏置的矩阵分解**（Biased Matrix Factorisation / FunkSVD），这是推荐系统领域预测显式评分的社区标准方法。

预测公式：

```
r̂_ui = μ + b_u + b_i + p_u · q_i
```

其中 μ 为全局均值，b_u / b_i 为用户 / 物品偏置，p_u / q_i 为隐向量。

训练细节：

1. 将评分从 `[10, 100]` 归一化到 `[1, 10]` 后训练，使标准 SGD 超参数有效；预测时反归一化。
2. 用 SGD 优化带 L2 正则的平方损失，防止过拟合。
3. 随机划分 10% 的训练评分为验证集，使用早停（patience=5）在验证 RMSE 不再提升时停止训练，自动恢复最优 checkpoint。
4. 冷启动（测试集中训练未见的用户 / 物品）：缺失项贡献为 0，退化为均值 + 已知偏置。

当前验证集 RMSE：约 **17.0**（全局均值基准约 20.8，改善约 18%）。

参考文献：
- Koren, Bell & Volinsky, *Matrix Factorization Techniques for Recommender Systems*, IEEE Computer 42(8), 2009.
- Hug, *Surprise: A Python library for recommender systems*, JOSS 5(52), 2020.
