# Task A 正式题集：编号数组指针

人工批准日期：2026-11-09。候选 `pointer40_lookup`，版本 `pointer_numeric_v1`，seed=7404。

- 数据：[synthesis_pointer_numeric_dataset.jsonl](synthesis_pointer_numeric_dataset.jsonl)
- 来源、人工审核、校准与校验和：[manifest](synthesis_pointer_numeric_dataset.manifest.json)
- [全量验证](../task_a/task_a_final_validation.json) / [人工批准](../task_a/task_a_human_review.json) / [定题报告](../task_a/task_a_lookup_review.md)

| 子集 | 单元 | 每单元题数 | 总数 | 实际深度 | 工作量 |
|---|---|---:|---:|---|---|
| serial | d=1..20 | 100 | 2000 | d | W=W_atomic=d |
| parallel | w=1..10 | 100 | 1000 | w=1时1，其余2 | W=W_atomic=2w−1 |

每题固定40个编号（0..39），执行 `current = next[current]` 恰好指定次数，答案为最终编号。
串行访问路径不重复；并行各链节点互不重叠，共享同一只读表，链长 `[1,2,...,2]`。
并行与串行奇数深度1..19配平；w=1是退化对照。多答案读出成本仍是混杂，不宣称纯深度因果隔离。

## 字段与使用

只把 **q** 用作模型 user 题干，DA/CoT system prompt 由 `experiment_task_a.ipynb` 的 `build_messages` 提供。
**不要把整条 JSON 发给模型**：ans / answers / chains.trace / chains.final 包含标准答案。

| 字段 | 含义 |
|---|---|
| qid / version / seed / config / split | 题目身份与生成来源；split固定formal |
| kind / d / w / W / W_atomic / matched_serial_d | 布局、实际最长依赖链、链数、工作量与串行配平深度 |
| q / table | 英文题干、40行index/value表 |
| ans / answers | ans为字符串；并行按链顺序逗号分隔；answers为整数列表 |
| chains | 每条链的initial、steps、trace、final与ans；含gold，不作为提示 |
| lookup | START、OPS、TABLE_SIZE、INDEX_MIN、INDEX_MAX、FIRST、LAST |

FIRST/LAST 为印出的首/末表行，不能直接沿用其它任务的同名约束语义。Task B 仍需独立重标定。

## 复现与验收

在仓库运行 `Experiments/task_a/experiment_task_a.ipynb`，保留 RUN_STAGE="offline"、SELECTED="pointer40_lookup"。
默认只复核已有结果；若需重建正式题，设置 GENERATE_FINAL=True，读取已批准的审核文件并校验结果指纹。
需要完整仓库内的真实 calib_v3_task.jsonl 与审核文件；无需GPU。已有文件内容不同会拒绝覆盖。

3000题均已独立复算、datasets回读和确定性重建；无重复，也不与全部600条历史校准题重叠。
数据 SHA256：`6b76b73f6ba3b523cbc8c59de8f5fc904deccd0e1b316e16364325820a9382ed`。

校准 CoT(20)=0.70、DA(20)=0、gap增长0.45；CA双侧p=0.013未达0.01，人工接受。
CoT曲线非单调，11个错误均是少走/多走一步，不能等同于纯查表能力随深度下降。
正式题集尚未运行完整模型矩阵；校准准确率不能冒充这3000题上的准确率。
