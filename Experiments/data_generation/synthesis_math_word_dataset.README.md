# Task A 正式题集：自然语言数学加减法应用题（`math_word_v1`）

人工批准日期：2026-11-27。选定候选 `word_int3_carry`，版本 `math_word_v1`，`seed=7404`，校准模型 `Qwen/Qwen3-4B`（commit `1cfa9a7208912126459214e8b04321603b3df60c`，`enable_thinking=False`）。

- 数据：[synthesis_math_word_dataset.jsonl](synthesis_math_word_dataset.jsonl)
- 来源、人工审查、校准与校验和：[manifest](synthesis_math_word_dataset.manifest.json)
- [全量验证](../task_a_math_word/task_a_word_final_validation.json) / [人工批准](../task_a_math_word/task_a_word_human_review.json) / [定题报告](../task_a_math_word/task_a_word_review.md)

| 子集 | 单元 | 每单元题数 | 总数 | 实际深度 `d` | 工作量 `W` |
|---|---|---:|---:|---|---|
| `serial` | `d = 1..20` | 100 | 2000 | `d` | `W = W_atomic = d` |
| `parallel` | `w = 1..10` | 100 | 1000 | `w=1` 时为 `1`，其余为 `2` | `W = W_atomic = 2w − 1` |

- **题目规格（`word_int3_carry`）**：纯自然语言日常物品计数加减法应用题；初值 `140..460`，每步变化量 `36..189`，全过程中间值 `40..960`，每一步强制触发十进制进位或借位；无 `if/else` 分支、无无关干扰句、无隐含英文数词、无乘除取模、无 `min/max` 聚合。
- **串/并行配平**：并行题由 `w` 位互不重名主人公各自独立完成加减事件，链长固定为 `[1, 2, ..., 2]`（`w=1` 时 `[1]`），总工作量 `W = 2w − 1` 与串行奇数深度 `matched_serial_d ∈ {1, 3, ..., 19}` 一一配平；多答案逗号读出成本仍为潜在混杂，不宣称纯深度无混杂因果隔离。

## 字段与使用规范

- **只把 `q` 作为模型 user 题干传入**；对称 2-shot 示例、`DA`（`continue_final_message=True` 续写 `"#### "`）与 `CoT` 提示词及严格判分器 `score_answer` 均由 `Experiments/task_a_math_word/experiment_task_a_word.ipynb` 提供。
- **严禁把整条 JSON 传给模型**：`ans` / `answers` / `solution` / `chains` 包含标准答案与中间状态轨迹。
- **DA 判分口径**：正式实验须沿用严格单行正确率 `strict_da_ok`（拦截高深度下模型在 `#### ` 后偷写算式或多行步骤的行为）。

| 字段 | 含义 |
|---|---|
| `qid` / `version` / `seed` / `config` / `split` | 题目唯一身份与生成参数；`split` 固定为 `"formal"` |
| `kind` / `d` / `w` / `W` / `W_atomic` / `matched_serial_d` | `"serial"` 或 `"parallel"`、最长串行链深、并行链数、原子运算工作量与配平串行深度 |
| `q` | 纯自然语言英文应用题题干（唯一可传给模型的题目字段） |
| `ans` / `answers` | `ans` 为答案字符串（并行按主人公顺序逗号分隔）；`answers` 为各链答案字符串列表 |
| `solution` / `chains` | 参考分步解与每条链的完整操作/状态轨迹（含 gold，禁止传入模型） |
| `lookup` | Task B 查表约束元数据：`PERSON`、`ITEM`、`START`、`OPS`、`FIRST_DELTA`、`LAST_DELTA`、`N_ADD`、`N_SUB`、`MAXDELTA` |

注：Task B 约束集在此新任务上的遵循率与 CoT 成本仍须独立标定与交叉复核，不可直接假设旧任务结论迁移。

## 复现与校验摘要

- 在仓库运行 `Experiments/task_a_math_word/experiment_task_a_word.ipynb`（默认 `RUN_STAGE="offline"`、`SELECTED="word_int3_carry"`、`GENERATE_FINAL=False`），无需 GPU 即可复核两轮共 1,065 条真实校准记录。若需重验正式生成，设 `GENERATE_FINAL=True`（存在字节不一致文件时会拒绝覆盖）。
- 全部 3,000 题已通过独立英文解析求解器复算、`datasets` 回读、二次确定性重建、唯一性检查、`W` 配平检查，且与全部 1,065 条历史校准题及 few-shot 示例零重叠。
- 数据文件 SHA256：`daca8723447c4206bd23823ee22813bc548d7ba04af72358aeeb41f09151be6d`。
- 校准指标（`word_int3_carry`，n=30/深度）：`DA(1)=100%`、`DA(2)=63.3%`、`DA(3)=16.7%`、`DA(20)=0%`、`CoT(20)=66.7%`（命中 `[0.40, 0.75]` 硬目标）、`gap(20)−gap(1)=+0.67`、CA 趋势 `CoT p=3.04e-9`。正式 3,000 题尚未执行全矩阵模型推理，校准集准确率不能代替正式矩阵测量结果。
