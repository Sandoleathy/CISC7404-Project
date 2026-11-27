# Task A（`taskA-gemini`）：自然语言数学加减法应用题（`math_word_v1`）

入口 Notebook：[experiment_task_a_word.ipynb](experiment_task_a_word.ipynb)

本目录在 `taskA-gemini` 分支全新设计并实现**自然语言数学加减法应用题**生成与校准流水线，模型统一使用 **`Qwen/Qwen3-4B`**（显式关闭内置 `<think>`，使 `DA` 与 `CoT` 严格由提示词区分）。已交付的历史指针数据（`synthesis_pointer_numeric_dataset.jsonl`）保持原样未改动。

---

## 1. 为什么重新设计（对照 `cursor` 分支 DA 低正确率问题）

`cursor` 分支的 `cond_wordproblem_v1` 在 `d=1/2/3/5/20` 上严格 `DA` 正确率仅 `0.70 / 0.08 / 0.03 / 0.00 / 0.02`（首轮无预填仅 `0.10`），核心原因有三点：

1. **单步复合逻辑负荷过重**：每步都塞入 `If v > T ... otherwise ...` 阈值判断 + `more/fewer` 极性随机翻转 + `100%` 必带无关干扰数字句，导致哪怕 `d=1` 题干里也有 **5 个数字**，`d=2` 高达 **8 个数字**，模型单次前向在 `d=2` 即跌至地板（`8%`）。
2. **对结合律的过度担忧**：纯自然语言加减法中，加减符号隐藏在英文动词语义里，4B 模型在单次前向（零中间 token）下根本无法对 10~20 句英文事件做并行精确求和，不需要靠人工 `if/else` 锁死 DA。
3. **模型与提示词不对称**：旧实验早期使用 `Qwen3-4B-Instruct-2507`，且仅给 DA 单侧加 few-shot。

### 本分支的新设计（`math_word_v1`）

- **回归纯自然语言串行加减法应用题**：贴纸/图书/烘焙库存、千克/升/米计量、美元零花钱收支三类自然情境；每一步恰好是一个无歧义增减事件、仅含 **1 个变化量数字**，无 `if/else` 分支、无无关干扰句、无乘除取模、无 `min/max` 聚合。
- **反捷径与数学不变量**：
  - 静态模板禁用一切阿拉伯数字与英文数词（`one`..`twelve`、`dozen`、`pair`、`half`、`twice` 等）；
  - 最终答案绝不等于初值（`echo_initial` 必判错）、绝不等于题干任意数字（无法照抄）；
  - `d ≥ 2` 强制加减混合，最终答案绝不等于「全加」或「全减」，且禁止相邻 `2~3` 步局部抵消；
  - 一位/两位小数内部全程按 `10` / `100` 倍整数精确运算，无浮点误差；`*_carry` 配置按十进制 Kummer 定理强制每步发生进位或借位。
- **模型固定为 `Qwen/Qwen3-4B`（`enable_thinking=False`）**：
  - **DA / CoT 对称 2-shot 示例**：共享来自独立 `fewshot` 划分的 2 道示例题（串行为 1 加 1 减 `d=1` 题；并行为 `w=2` 题）；
  - **DA 续写预填**：使用 `continue_final_message=True` 在 `<|im_start|>assistant\n<think>\n\n</think>\n\n#### ` 后直接续写答案，无需 `stop=['\n']` 硬截断即可避免思维链泄漏；主指标采用**严格单行正确率 `strict_da_ok`**。

---

## 2. 五档同构候选阶梯

| 候选名 | 情境 | 数值精度与范围 | 进退位要求 | 用途 |
|---|---|---|---|---|
| `word_int1_easy` | 日常物品计数 | 整数：初值 `18..55`，每步 `±3..14` | 不强制 | 若两位数 `DA(1)<0.80` 时的保底高 DA 轻量档 |
| **`word_int2_carry`** | 日常物品计数 | **2 位整数：初值 `35..95`，每步 `±12..39`** | **每步强制进/退位** | **默认搜索起点（`SEARCH_START`）** |
| `word_int3_carry` | 日常物品计数 | 3 位整数：初值 `140..460`，每步 `±36..189` | 每步强制进/退位 | `CoT(20)>0.75` 时的第一级整数提难档 |
| `word_dec1_carry` | 重量/容量/长度计量 | 1 位小数：初值 `25.1..88.9`，每步 `±3.1..27.9` | 每步强制进/退位 | 一位小数精确加减提难档 |
| `word_dec2_money` | 零花钱与账本（`$`） | 2 位小数：初值 `$28.01..$95.99`，每步 `±$3.01..$27.99` | 每步强制进/退位 | 两位小数美分进退位最高提难档 |

---

## 3. Colab 运行与交付流程

1. 在 Colab 从 `taskA-gemini` 分支打开 `experiment_task_a_word.ipynb`，选择 **L4 GPU**（推荐，`bfloat16`）或 **T4 GPU**（自动回退 `float16`）。
2. 点击 **「全部运行」**（首轮用 `RUN_STAGE = "search"`；**当前默认 `RUN_STAGE = "compare"`**，见下方「预注册对比」）：
   - 自动执行双向搜索（`probe`: `d=1 × 100 × DA`；`screen`: `d=20 × 25 × CoT`）并选定最优候选；
   - 自动在独立 `slope` 划分上运行 **`d ∈ {1, 2, 3, 5, 10, 15, 20}` × 30 题 × (DA, CoT) = 420 次推理**；
   - 生成曲线图 `task_a_word_calibration.png`、汇总表 `task_a_word_summary.json`，并自动下载 `task_a_word_results.zip`。
3. **人工审查后生成正式 3000 题**：
   - 将下载的 `task_a_word_results.zip` 解压回本目录，**人工核查曲线与原文错误样本**；
   - 按 Notebook 第 9 节模板填写 `task_a_word_human_review.json`（`approved: true` 并填入 `evidence_digest`），将 `GENERATE_FINAL` 设为 `True`（可在本地 CPU 下 `RUN_STAGE="offline"` 直接运行），即可生成并全量校验 `Experiments/data_generation/synthesis_math_word_dataset.jsonl`（串行 `d=1..20 × 100 = 2000` 条 + 并行 `w=1..10 × 100 = 1000` 条，`W = 2w - 1`）。

## 预注册对比：`word_int3_carry` vs `word_int2_carry`（2026-11-27 已完成并获人工批准，正式 3000 题已交付）

- 背景：首轮 `search` 中 `word_int2_carry` 的 slope `CoT(20)=25/30=0.833` 高于 `0.75`，`DA(1)=100/100` 高于描述上界 `0.95`（见 `task_a_word_review.md`）。
- `RUN_STAGE = "compare"` 只测 `word_int3_carry`：probe `d=1×100×DA` + 独立 slope `7 深度×30×(DA, CoT)` = 520 条推理。
- 判定规则（`COMPARE_RULE`，digest `1626304fed56134b6c0be06e5e04cabdd384d83c87100585454318f9ddc8388e`，运行时校验）：
  - probe `DA(1) ≥ 0.85`（≥ 85/100），且
  - slope `DA(2) ≥ 0.50`（≥ 15/30），且
  - slope `CoT(20) ∈ [0.40, 0.75]`（12/30..22/30）。
  - 三条同时满足才改选 `word_int3_carry`；否则保留 `word_int2_carry`，不再上探 `dec1/dec2`、不增样本、不改阈值。
- 输出：`task_a_word_compare_word_int3_carry.json`、`task_a_word_calibration_word_int3_carry.png`（不覆盖首轮 `task_a_word_summary.json`）。
- 无论哪一档胜出，正式 3000 题仍须开发者人工审查后写入 `task_a_word_human_review.json` 才能生成。

- **实测结果（Colab L4，`Qwen/Qwen3-4B` commit `1cfa9a7208912126459214e8b04321603b3df60c`，520 条）**：
  - probe `DA(1) = 100/100 (100.0%) ≥ 0.85` ✅
  - slope `DA(2) = 19/30 (63.3%) ≥ 0.50`（`d=3` 为 `16.7%`，低深度平滑衰减）✅
  - slope `CoT(20) = 20/30 (66.7%) ∈ [0.40, 0.75]`（正面命中 `task.md` A.0 唯一硬目标）✅
  - `gap(20) - gap(1) = +0.6667`（95% CI `[0.500, 0.833]`），CA 趋势 `CoT p = 3.04e-09`、`DA p = 2.38e-16`。
  - 三条预注册条件全部满足，自动推荐 `word_int3_carry`；当前 Notebook 默认已切回安全离线模式 `RUN_STAGE = "offline"`，2026-11-27 经开发者回复「批准 word_int3_carry」，已写入 `task_a_word_human_review.json` 并交付正式 3,000 题 `Experiments/data_generation/synthesis_math_word_dataset.jsonl`（SHA256 `daca8723…151be6d`）。
