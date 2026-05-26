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
