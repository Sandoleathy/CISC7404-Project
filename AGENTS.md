# AGENTS.md

本文件是给 AI 编程助手（以及新加入的人类）的项目说明书。**修改代码前先读完本文件**，尤其是
第 3 节「vLLM 统一加载参数」——那是硬性要求，不可自行改动。

---

## 1. 项目是什么

CISC7404 课程项目。研究问题是：**思维链（CoT）带来的准确率提升，是否随推理深度 `d` 增长？**

核心做法是构造一个 `d × k` 的测试矩阵：

- `d` —— 推理深度 / 串行计算链长度（必须一步一步算的步数）
- `k` —— 输出约束（instruction-following constraint）的条数
- `W` —— 工作量（workload），即解题真正需要的原子运算次数；串行题 `W = d`，pointer 并行题
  `W = d × w`（`w` 个起点各走 `d` 步），Kaggle 模运算 pilot 的并行题 `W = 2w - 1`

实验自变量是「深度」，但深度天然伴随工作量变化，因此必须用同 `W`、不同 `d` 的对照
（串行 `d` 对齐并行 `w`，使 `d == 2w - 1`）把「深度效应」和「工作量效应」拆开。

数据来源（当前）：

1. **合成 pointer tracking**：`synthesis_dataset.ipynb` 生成 `synthesis_q_k_dataset.jsonl`（串行 `d × k`）和 `synthesis_d_w_dataset.jsonl`（并行 `w / d / W`）
2. **合成 arithmetic chain**：`synthesis_math_dataset.ipynb` 生成 `synthesis_math_dataset.jsonl`（小数加减链，求含起点的最小值）
3. **GSM8K**：真实数学推理数据，用 `<<expr=result>>` 注解图算真实 `d` / `W`
4. **IFEval**：指令遵循数据（目前只做到下载/预览）

对照条件固定为两种系统提示：`DA`（只输出最终答案）与 `CoT`（在输出中显式写出中间步骤）。

**当前实验条件（2026-11 定稿方向）：**

- **最终正式实验模型：`Qwen3-4B-Instruct-2507`**，HF id `Qwen/Qwen3-4B-Instruct-2507`，
  本地目录 `models/Qwen3-4B-Instruct-2507`（尚未下载）。它是**非 thinking 模型**，
  chat template 里没有 thinking 开关，正好符合「CoT 完全由 prompt 控制」的设计。
- **DA / CoT 只由 system prompt 区分**：DA 只给最终答案，CoT 先显式写出中间步骤。构造方式见第 4 节。
- 不再做 thinking on/off 对比；Qwen3-1.7B 的 thinking / no-thinking pilot 只是历史对照。
- `W` 配平仍用于 serial / parallel 对照，拆分深度 `d` 和工作量 `W`。
- 当前已跑的 pilot 仍是 1.7B 的 `experiment_math.ipynb`；4B + DA/CoT 的正式实验**还没开始跑**。

> 当前阶段：GPU 资源受限（RTX 3070 8GB），1.7B 已跑通全链路；4B 的显存方案尚未标定
> （bf16 权重约 7.5 GiB，8GB 卡放不下，见第 3.3 节）。
> `Experiment.ipynb` 与 `Experiments/` 下的笔记本都是历史记录，不代表正式实验。
> 笔记与结论沉淀在 `record.md`，踩坑记录在 `vllm_problems_solutions.md`。

---

## 2. 环境与常用命令

### 2.1 平台约束（很重要）

- 项目在 **Windows** 上开发，但 **vLLM 必须在 WSL (Ubuntu) 中运行**（README 第 52 行）。
  仓库里的 `.venv` 是 **Linux venv**（`.venv/lib/python3.14/...`），不存在 `.venv/Scripts`。
- 因此：`uv sync`、`jupyter notebook`、跑 `.ipynb` 全部在 **WSL 终端**里执行。
  在 Windows 原生 PowerShell 里运行含 `vllm` 的代码会失败。
- GPU 只有一张 8GB 卡，**不要**为了「更快」改 `tensor_parallel_size` 或多进程并行加载模型。
- 8GB 卡跑 bf16 的 4B 模型（权重约 7.5 GiB）**一定 OOM**。4B 的加载方案见第 3.3 节，
  不要把 1.7B 的 `0.80 / 8192` 直接套过去。

### 2.2 已安装版本（实测，改动需同步更新本表）

| 组件 | 版本 |
|---|---|
| Python | 3.14.4 |
| PyTorch | 2.14.1+cu132（CUDA 13.2） |
| vLLM | 0.32.0 |
| transformers | 5.21.0 |
| datasets | 5.1.0 |
| accelerate | 1.15.0 |
| triton | 3.8.0 |
| flashinfer-python | 0.7.0.post1 |
| pandas / matplotlib | 3.1.0 / 3.11.2 |

### 2.3 命令

```bash
uv sync                                  # 按 pyproject.toml / uv.lock 同步环境
uv add <pkg>                             # 新增依赖（会改 pyproject.toml + uv.lock）
uv lock                                  # 更新锁文件

# 按机器分平台安装（依赖由三份 requirements 分别管理）
uv pip sync configs/cpu/requirements.txt            # CPU 机器
uv pip sync configs/windows-gpu/requirements.txt    # Windows GPU
# configs/linux-gpu/ 在 README 里有引用，但目录尚未创建

# PyTorch 必须手动装（CPU / GPU 平台不同）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

jupyter notebook
jupyter server --ServerApp.allow_remote_access True   # 需要远程访问时
```

- 包源镜像配置在 `uv.toml`（清华源），下载慢时加 `UV_HTTP_TIMEOUT=300`。
- 当前目录里没有安装 `.venv` 的 Windows 侧解释器，不要用 `uv run python main.py` 期待能跑 vLLM。

### 2.4 目录

```
.
├── README.md                     安装 / 启动说明（英文）
├── AGENTS.md                     本文件
├── record.md                     研究计划与实验记录（中文，持续更新）
├── vllm_problems_solutions.md    vLLM 踩坑与解决方法（中文）
├── pyproject.toml / uv.lock      uv 依赖定义与锁文件
├── uv.toml                       镜像源
├── main.py                       占位入口，目前只 print 一句话
├── configs/
│   ├── cpu/requirements.txt
│   └── windows-gpu/requirements.txt
├── models/                       gitignore；Qwen3-1.7B 已下载，Qwen3-4B-Instruct-2507 未下载
├── synthesis_q_k_dataset.jsonl   pointer tracking 串行 d×k（已提交，4000 行）
├── synthesis_d_w_dataset.jsonl   pointer tracking 并行 d×w（未提交）
├── synthesis_math_dataset.jsonl  小数加减链（未提交，4000 行）
├── pilot_math_results.jsonl      1.7B / 2048 token 的旧跑结果
├── pilot_math_results_v2.jsonl   1.7B / 6144 token 的 v2 结果
├── pilot_results.jsonl           pointer tracking pilot 结果
├── gsm8k_dataset/                gitignore；由 notebook 生成
├── ifeval/                       gitignore；由 notebook 生成
├── *.ipynb                       当前实验代码（数据生成 + experiment_math）
├── Experiments/                  早期 notebook（vLLM_test / gsm8k_test / Qwen3-4B-Test /
│                                 experiment_pointer_track），历史参考
├── Thoughts/                     零散想法（Pointer_Track.md）
└── Papers/                       参考论文
```

`gsm8k_dataset/`、`ifeval/`、`models/`、`.venv/` 都在 `.gitignore` 中——**不要**把数据集或权重提交进仓库。
`synthesis_q_k_dataset.jsonl` 是唯一已纳管的合成数据集（约 3.6MB），它由
`synthesis_dataset.ipynb` 生成，**删了要能重新生成**（seed 固定，见第 5.5 节）。

> `synthesis_d_w_dataset.jsonl`、`synthesis_math_dataset.jsonl`、`pilot_*.jsonl` 目前**未提交**，
> 也不要默认加进 git：它们体积大（十几 MB 到几百 MB），需要时用 notebook 重新生成。
> 具体用途见第 5 节。

### 2.5 路径约定

notebook 里所有路径都是**相对仓库根目录**的：

```python
model_path = 'models/Qwen3-1.7B'
ds = load_dataset('json', data_files='synthesis_q_k_dataset.jsonl')
df = load('gsm8k_dataset/train.jsonl')
```

所以 **jupyter 必须从仓库根目录启动**，否则会 FileNotFoundError。不要改成绝对路径。

### 2.6 测试与校验

仓库**没有** pytest / ruff / CI。验证方式就是「把相应 cell 跑通、看输出」。
改完 notebook 后建议做一次 JSON 合法性检查，避免产出坏文件：

```bash
python -c "import json,sys; [json.load(open(f, encoding='utf-8')) for f in sys.argv[1:]]" *.ipynb
```

---

## 3. vLLM 统一加载参数（必须遵守）

**要求：所有 vLLM 推理代码都从本节模板出发，不要从零手写 `LLM(...)`。**
采样参数全项目统一；显存参数按「模型档位」分开——**档位一经标定就必须照抄并留档**。

### 3.1 两个档位

| 档位 | 模型 | 本地目录 | 用途 | 状态 |
|---|---|---|---|---|
| A | Qwen3-1.7B | `models/Qwen3-1.7B` | pilot、链路自测、回归 | 已下载，已跑通 |
| B | **Qwen3-4B-Instruct-2507** | `models/Qwen3-4B-Instruct-2507` | **最终正式实验** | 未下载，显存方案待标定 |

- HF repo id 是 `Qwen/Qwen3-4B-Instruct-2507`。**不要**换成 `Qwen/Qwen3-4B`（base）或
  `Qwen/Qwen3-4B-Thinking-2507`（thinking-only）：base 版没有对齐好的对话能力，thinking 版会
  强制自带思考块，两者都破坏「CoT 由 prompt 控制」的设计。
- 档位 B 是**非 thinking 模型**：`tokenizer_config.json` 里的 chat template **没有**
  `enable_thinking` 分支，generation prompt 就是 `<|im_start|>assistant\n`。
  因此对档位 B **不要**传 `chat_template_kwargs={"enable_thinking": ...}`。

### 3.2 档位 A 模板（Qwen3-1.7B，现状）

```python
import os

# 必须在 import vllm 之前设置：关闭 FlashInfer 采样器，避免没有 nvcc 时的 JIT 采样扩展编译失败
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from vllm import LLM, SamplingParams

model_path = "models/Qwen3-1.7B"

# ---- 档位 A 的显存参数：8GB 卡上的稳定值，不要动 ----
vllm_model = LLM(
    model=model_path,
    gpu_memory_utilization=0.80,
    max_model_len=8192,
)

# ---- 全项目统一的采样参数 ----
PARAMS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,          # 见 3.5：目前只定义、不传
    "max_tokens": 6144,   # 按条件定，见第 4.3 节
}
sampling_params = SamplingParams(
    temperature=PARAMS["temperature"],
    top_p=PARAMS["top_p"],
    max_tokens=PARAMS["max_tokens"],
)

# ---- 生成（档位 A 必须显式关掉 thinking，否则输出被思考块占满；档位 B 不传）----
outputs = vllm_model.chat(
    messages,
    sampling_params,
    chat_template_kwargs={"enable_thinking": False},
)
```

### 3.3 档位 B：Qwen3-4B-Instruct-2507（最终实验，参数待标定）

**先看显存账。** 4B 的 bf16 权重约 **7.5 GiB**（4.02B params × 2 B），而 8GB 卡在
`gpu_memory_utilization=0.80` 下只有约 6.4 GiB 预算——**档位 B 不可能照抄档位 A 的 0.80 / 8192**。

`config.json` 已核对：36 层、hidden 2560、32 个 Q head / 8 个 KV head、head_dim 128、
vocab 151936、bf16、`tie_word_embeddings=true`。
每 token KV = 2 × 36 × 8 × 128 × 2 B = 144 KiB → 8192 上下文约 **1.13 GiB**。
**瓶颈是权重，不是 KV cache。**

三条可行路径，选一条，实测后把数字回填：

| 路径 | 做法 | 估算占用 | 前提 |
|---|---|---|---|
| B1 本地 fp8 | `quantization="fp8"`（Ampere 走 FP8 Marlin） | 权重约 3.7 GiB + KV 1.1 GiB + 激活 ≈ 5.5–6.0 GiB | 必须先 smoke test 确认 vLLM 0.32 在 sm86 上起得来 |
| B2 本地 4bit | 换 AWQ / GPTQ int4 权重 | 权重约 2.3 GiB | 需要现成量化 checkpoint；vLLM 不能在线做 4bit |
| B3 学校 GPU | 不量化，按卡型设利用率 | 权重 7.5 GiB + KV 1.1 GiB | 需要知道卡型/显存，并把数字记下来 |

**在这张表填满、并通过一次 smoke test 之前，不要开始跑正式数据。**
smoke test 通过标准：`LLM()` 初始化成功、日志里有 `Free memory on device` / `Actual usage`、
连续生成 1 条 prompt 不 OOM。

| 参数 | 档位 B 取值 | 说明 |
|---|---|---|
| `model` | `models/Qwen3-4B-Instruct-2507` | 本地权重（首次下载约 8GB） |
| `quantization` | **待填**（`fp8` / `awq` / 不填） | 由 B1 / B2 / B3 决定 |
| `gpu_memory_utilization` | **待填** | 8GB 卡建议 ≤ 0.85；更大显存可 0.90 |
| `max_model_len` | **待填**（起始 8192） | 显存紧就先降到 4096 |
| `max_num_seqs` | **待填**（起始 8） | 和 batch 一起定，防止 KV 峰值打爆 |

### 3.4 参数清单（哪部分能改、哪部分不能）

| 参数 | 档位 A | 档位 B | 允许改？ |
|---|---|---|---|
| `gpu_memory_utilization` | `0.80` | 待标定 | 否（只能按 3.3 的流程标定后回填） |
| `max_model_len` | `8192` | 待标定（≥ 2048） | 否（同上） |
| `VLLM_USE_FLASHINFER_SAMPLER` | `"0"` | `"0"` | 否 |
| `chat_template_kwargs` | `{"enable_thinking": False}` | **不传** | 否 |
| `temperature` | `0.6` | `0.6` | 否 |
| `top_p` | `0.95` | `0.95` | 否 |
| `max_tokens` | 按条件定（pilot 用 6144） | 按条件定 | 是（必须按第 4.3 节记录长度与截断） |

- `gpu_memory_utilization=0.80` 的来历：0.90 会申请约 7.2 GiB，而启动时可用只有 6.81–6.95 GiB，直接 OOM。
- `max_model_len=8192` 的来历：模型默认 40960 时 KV cache 需 4.38 GiB，装不下；8192 时约 2.3 GiB。
- `VLLM_USE_FLASHINFER_SAMPLER="0"` 的来历：缺 nvcc / CUDA Toolkit 时 FlashInfer 采样扩展的 JIT 会失败；
  日志会确认 `FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0`。

### 3.5 采样参数

- `temperature=0.6`、`top_p=0.95` 与原始 `vLLM_test` 一致，**不要动**。
- `top_k=20` 在 `Experiments/vLLM_test.ipynb` 里**只定义、没传进 `SamplingParams`**，当前 pilot 也沿用「不传」。
  要统一传入属于协议变更：必须同时改本文件、`record.md`，并重跑对照。
- `max_tokens` 是安全上限，不是「控制在 N token」的手段；DA / CoT 分别定，见第 4.3 节。

### 3.6 规则

1. 新增推理代码先抄模板，别从零写 `LLM(...)`。
2. **不要**为了让 4B 跑起来就随手调 `gpu_memory_utilization` / `max_model_len`，也不要加
   `enforce_eager=True`、`tensor_parallel_size`、`dtype` 之类来「优化」——这些都会改变显存账本和吞吐。
   要改就按 3.3 的流程走，并把最终值写进本文件。
3. `VLLM_USE_FLASHINFER_SAMPLER` 必须在 `import vllm` 之前设置；notebook 里改这一行要**重启 Kernel**。
4. 同进程「先 transformers 后 vLLM」必须先释放显存：
   `del model; gc.collect(); torch.cuda.empty_cache()`。
5. 换模型 / 换机器：**先更新本节表格和 `record.md`，再改代码**。
6. 后台 `vllm serve` 不是当前用法，需要时另开一节说明。
7. `models/` 已 gitignore；4B 权重不要提交。

### 3.7 参考收益（档位 A）

`Experiments/vLLM_test.ipynb` 中同一模型、同一 4 条 prompt、同样 512 max tokens：

| 后端 | 耗时 |
|---|---|
| transformers (`model.generate`) | 24.28 s |
| vLLM (`LLM.chat`) | 6.30 s |

---

## 4. DA / CoT：只由 system prompt 决定

**最终实验没有 thinking 开关。** 档位 B 本身就不是 thinking 模型，条件只有两个，全部靠提示词区分。
这也是本项目「CoT 是否有用」的唯一操作化定义。

| 条件 | system prompt 里的指令 | 期望输出 |
|---|---|---|
| `DA`（direct answer） | 不写推理、不解释，直接给最终答案 | 只有最终答案（仍要满足 C1–Ck） |
| `CoT`（chain-of-thought） | 先显式写出每一步的中间结果，再给最终答案 | 推理 trace + 最终答案 |

### 4.1 拼装规则（不遵守则 k 轴不可比）

```python
DA_PREAMBLE = (
    "Solve the problem silently. Do not show intermediate steps and do not explain "
    "your reasoning. Output only what the rules below require.\n"
)
COT_PREAMBLE = (
    "Solve the problem step by step. Before the final answer, explicitly write the "
    "value after each operation, one per line. Then apply the rules below to the "
    "final answer.\n"
)

system = PREAMBLE[mode] + row["cons"]       # preamble 在前，约束原样跟在后面
messages = [
    {"role": "system", "content": system},
    {"role": "user",   "content": row["q"]},
]
```

硬性要求：

1. 两种条件的 `user` message **完全相同**，`cons` **完全相同**，只差 preamble 里那一句。
2. preamble 放在 `cons` **之前**，不要插进约束中间；否则 `C1..Ck` 的编号和数据集里的 `cons` 对不上。
3. 一次实验只用一个 `prompt_version` 字符串（例如 `v3-da-cot`），写进每条结果记录。
4. 不允许用 `enable_thinking` 造第三个条件（档位 B 没有这个功能，档位 A 也不再用）。
5. 不允许给两种条件加不同的 few-shot 示例：那是第三种变量，会把 `k` 和示例混在一起。

### 4.2 判定口径

- 最终答案从 marker 提取（math 数据是 `####ANSWER=`）；档位 B 不再切 `<think>`，
  `split_thinking` 退化为恒等，保留函数只为兼容旧结果。
- C1–C10 只作用在**最终答案部分**；CoT 的 trace 不参与约束判定，但要计入长度统计。
- DA 条件如果写出了逐步推理，记为 **DA 违约**（单独统计 `da_violation`），不能悄悄当成 CoT。
- CoT 条件如果只给答案，记为 **CoT 违约**。两种违约率都要报告。

### 4.3 Token 预算与截断

- CoT 天然更长：`max_tokens` 要按条件分别定，**两个值都要写进结果元数据**，不能用同一个值硬套。
- 必须记录 `finish_reason`、`n_tokens`、`trunc`；报告 DA vs CoT 的长度分布与 stop 子集指标。
- `prompt` 里限制步数/长度只是**软约束**，不能当保证；不要用后处理硬截断来制造「预算公平」——
  截断会同时改变答案和约束结果。
- 如果长度差异过大，优先做预算匹配或报告长度分布 / L*，而不是直接比较两个条件的原始准确率。

### 4.4 提示词污染（已经踩过，写新约束前先看）

- 约束里不要放**完整可抄**的答案模板：C5 的 `123 -> 321` 被模型抄成答案，C6 的
  `####ANSWER=RESULT:<answer>` 被原样抄成字面 `<answer>`，两次都污染了 k≥6 的数据。
- 示例只能用抽象形式（如「整数部分数字为 d1 d2 … dn，输出 dn … d2 d1」），并明确写「示例仅说明格式，不要照抄」。
- 新增/修改约束后，先用少量样本检查模型是否在复制示例文本，再跑全量。

---

## 5. Notebook 内容说明

每个 notebook 的用途如下。改动前先确认它属于哪一类：**数据生成**、**数据加工**、**模型基线**、
**探索性 pilot**、**历史参考**。

**最终正式实验（档位 B + DA/CoT）目前还没有对应的 notebook**；新建时在本节登记，
命名与注释风格沿用第 6 节。

### 5.1 总览

| 文件 | 定位 | 关键产出 | 需要 GPU |
|---|---|---|---|
| `Experiments/vLLM_test.ipynb` | vLLM 可行性 + 加速比验证（档位 A 的来源） | 统一加载参数、速度对照 | 是（WSL） |
| `Experiments/Qwen3-4B-Test.ipynb` | **旧** base `Qwen3-4B` 的 transformers 基线 | GSM8K 输出格式约定 | 是（历史，未跑通） |
| `Experiments/gsm8k_test.ipynb` | GSM8K 下载 + 注解图深度分析 | `gsm8k_dataset/*.jsonl`、`*_d.jsonl`、诊断图 | 否（CPU/pandas） |
| `Experiments/experiment_pointer_track.ipynb` | pointer tracking pilot | `pilot_results.jsonl` | 是（WSL） |
| `synthesis_dataset.ipynb` | pointer tracking 合成数据（串行 + 并行） | `synthesis_q_k_dataset.jsonl`、`synthesis_d_w_dataset.jsonl` | 否（CPU/random） |
| `synthesis_math_dataset.ipynb` | 小数加减链合成数据 + C1–C10 约束 | `synthesis_math_dataset.jsonl` | 否（CPU/random） |
| `experiment_math.ipynb` | 1.7B 的 thinking vs no-thinking pilot（历史） | `pilot_math_results_v2.jsonl` | 是（WSL） |
| `prior_investigation_kaggle.ipynb` | Kaggle 上的 pilot 探索（S1/S2/S3） | `pilot_recs.jsonl`、Δ–d 曲线 | 是（Kaggle T4） |
| `Experiment.ipynb` | 早期 pointer pilot / 历史参考 | — | — |

### 5.2 `Experiments/vLLM_test.ipynb`（11 个 cell）

目的：验证「本机能不能跑 vLLM」，并量化相对 transformers 的加速比。**本仓库 vLLM 用法的唯一事实来源。**

1. 设置 `VLLM_USE_FLASHINFER_SAMPLER=0`（注释：规避 flashinfer 采样报错）
2. 导入 `time / torch / transformers / vllm / gc`
3. 打印 `torch.cuda.is_available()`、torch 版本、CUDA 版本
4. 定义 `PARAMS`（temperature 0.6 / top_p 0.95 / top_k 20 / max_tokens 512）
5. markdown：`Load transformers model`
6. `AutoModelForCausalLM.from_pretrained('models/Qwen3-1.7B', device_map="auto")` + tokenizer
7. 基线测速：4 条固定 prompt（自我介绍 / 美国总统 / 法国首都 / AI 的未来），
   `apply_chat_template(enable_thinking=False)` → `generate(max_new_tokens=512)`，
   打印原始输出与总耗时（24.28 s），随后 `del model` + `gc.collect()` + `torch.cuda.empty_cache()` 释放显存
8. markdown：`Load vLLM model`
9. 构造 `SamplingParams` 并 `LLM(model, gpu_memory_utilization=0.80, max_model_len=8192)`
   （日志确认：V1 engine、dtype bfloat16、max_seq_len 8192、KV cache 2.3 GiB / 21,568 tokens）
10. `vllm_model.chat(messages, sampling_params, chat_template_kwargs={"enable_thinking": False})`，
    同样的 4 条 prompt，打印输出与总耗时（6.30 s）
11. 空 cell

注意：cell 7 是「先 transformers 再 vLLM」的典型顺序，OOM 的根因就在这；顺序敏感，别把释放显存的
三行删掉。

### 5.3 `Experiments/Qwen3-4B-Test.ipynb`（8 个 cell）

目的：在换到 vLLM 之前，用 transformers 检查 Qwen3-4B 的基础能力与输出格式。

1. 导入 `torch / numpy / datasets.load_dataset / transformers`
2. 检查 CUDA
3. `model_name = "Qwen/Qwen3-4B"`，本地路径 `models/Qwen3-4B`（**该目录当前不存在**，需要先下载权重）
4. 单轮对话「Hello, who are you?」，`enable_thinking=True`，`max_new_tokens=1024`
5. 加载 `gsm8k_dataset/train.jsonl`（依赖 `Experiments/gsm8k_test.ipynb` 先跑过）
6. 打印前 10 题题干与完整解答
7. 定义 `SYSTEM_PROMPT`：要求逐步推理，且**最后一行必须是 `#### <数字>`**；
   对前 3 道 GSM8K 题生成，`enable_thinking=True`、`max_new_tokens=1024`
8. 空 cell

这个 notebook 用的是 **base `Qwen/Qwen3-4B`**（thinking 默认开启），**不是**最终实验的
`Qwen3-4B-Instruct-2507`；只作历史基线，不要拿它的输出格式当正式协议。

### 5.4 `Experiments/gsm8k_test.ipynb`（9 个 cell）

目的：把 GSM8K 的 `<<expr=result>>` 注解还原成依赖图，量出真实深度 `d` 与工作量 `W`，
判断它能否作为合成数据的真实对照。

1. `load_dataset("openai/gsm8k", "socratic")`（联网下载）
2. 打印前 5 条
3. 把各 split 落盘为 `./gsm8k_dataset/{split}.jsonl`
4. **核心分析 cell**：
   - `parse_anns`：抽取所有 `<<表达式=结果>>` 注解
   - `depths`：把「注解 A 的结果值」与「后续注解 B 的操作数」按数值相等连边，得到每个注解的深度；
     支持 `rule="recent"`（只连最近的匹配，默认）与 `rule="all"`（连所有匹配）两种规则，并统计歧义匹配
   - `analyse`：逐题计算 `d`（最大链深）、`W`（注解总数，即工作量）、`d_all`、无父节点数、
     最终答案（`####` 后的值）、题干长度；输出有注解比例、断链率、`d/W` 分布、
     `d` 与 `W`/题长的 Spearman 相关、答案类型分布
   - 结果写 `gsm8k_dataset/train_d.jsonl`、`test_d.jsonl`；画三张诊断图存
     `gsm8k_dataset/gsm8k_diagnostics.png`
5. 统计「真实断链率（扣除链首）」与「干净单链子集」（`no_parent==1 & d==W & W>=2`）的规模
6. `load_dataset("google/IFEval")`
7. 打印 IFEval 第一条样例
8. **有 bug 的 cell**：本意是把 IFEval 存到 `./ifeval/{split}.jsonl`，但循环遍历的是 GSM8K 的 `ds`，
   会把 GSM8K 内容写进 `ifeval/` 目录。用之前先改成 `for split_name, split_dataset in ds_ifeval.items():`
9. 空 cell

### 5.5 当前合成数据（2026-11 更新）

旧版 `synthesis_dataset.ipynb` 的 `+ - * /` 算术生成器已被 pointer tracking 取代，不再是主数据源；当前有两个合成任务。

1. **Pointer tracking**（`synthesis_dataset.ipynb`）
   - `synthesis_q_k_dataset.jsonl`：串行 `d × k`，`d=1..20`、`k=1..10`、每格 20 题，共 4000 行；
     指针按偏移量跳转，题干与答案由构造保证。
   - `synthesis_d_w_dataset.jsonl`：并行 `w / d / W` 数据，用于和串行 `d` 做 `W` 配平对照。
   - 旧 notebook 里的 `AMAX / _apply / _draw / make_question_with_d_and_k` 描述已过期；以当前 notebook 代码为准。

2. **Arithmetic chain**（`synthesis_math_dataset.ipynb`）
   - `synthesis_math_dataset.jsonl`：小数 `+ / -` 链，`d=1..20`、`k=1..10`、`i=1..20`，共 4000 行；
     pilot 只取 `i=1..5`（1000 题）。
   - 题目从 `x` 出发，按顺序执行 `d` 步，问**包含起点在内的最小值**；`ans` 为一位小数，比较时按两位小数舍入。
   - `CONSTRAINS` 是 C1–C10 累积拼接；C3 是小数逐位英文词，C9 是数字和。当前 C3/C9 接近地板，
     **不能再作为 cumulative k 的硬门槛**；后续要先做单约束校准，再决定哪些进入 k 轴。
   - 数据更新只改 `cons` 时，`q/ans/W/d/k/i` 应保持不变；改生成逻辑会破坏与既有 pilot 的可比性。

3. **已跑 pilot**（`experiment_math.ipynb`）
   - 1.7B 的 hidden thinking vs no-thinking v2 已跑完，结论见 `record.md`。
   - 它是**历史对照**：正式实验换成 4B-Instruct-2507 + DA/CoT（第 4 节），不再用 thinking 开关。
   - 约束集和 token 预算仍要在正式跑之前重新设计；提示词层面的改动不要回写进数据集，保持 `cons` 不变。

### 5.6 `prior_investigation_kaggle.ipynb`（18 个 cell）

目的：在 Kaggle T4 上花 10 分钟做 pilot，验证三件事——S1 模型能否算对一步、S2 从第几步开始错、
S3 写推理过程是否提升准确率。**这是既往探索记录，不是最终实验代码**（用的是 transformers，
不是 vLLM；任务用模运算避免除法取整问题）。

1. markdown：S1/S2/S3 三段目标表 + 运行前设置（Accelerator = GPU T4 x2、Internet = On）+ 产出说明
2. 配置：`MODEL_NAME='Qwen/Qwen3-1.7B'`、`AMAX=12`、`MAX_TOKENS=768`、
   `OUTDIR=/kaggle/working`、`OUT_PATH=pilot_recs.jsonl`；`HF_HOME=/tmp/hf`（避免吃满 20GB 配额）、
   `CUDA_VISIBLE_DEVICES=0`、`TOKENIZERS_PARALLELISM=false`；pip 安装并打印 GPU 信息
3. 加载模型：优先 bf16，Turing 卡降级 fp16；tokenizer `padding_side='left'`；
   `dtype=` 失败时回退 `torch_dtype=`
4. 任务生成器（**模运算版**）：`make_serial`（`d` 步串行，默认 `mod=97`，只用 `+ - *`）、
   `make_parallel`（`w` 个互不依赖运算后求和取模，`d=2`、`W=2w-1`）
5. 打印一个串行样例和并行样例
6. 提示词与推理：`SYS_DA`（只输出数字）/ `SYS_COT`（写出中间值）、统一后缀
   `Final answer: <number>`、`NUMRE` 抽取（失败则取最后一个数字）、`build_prompt`（关 thinking）、
   `infer()` 贪心解码 `batch=8`，记录 `cond/kind/d/W/ans/pred/ok/ntok/text`
7. 工具函数：`save()` 追加写 JSONL（多次运行不丢数据）、`report()` 按
   `kind/d/cond` 聚合出 `acc/ntok/parse_fail`
8. markdown：S1 判读标准（准确率 60–90% 选该模数，>95% 调大 `AMAX`，<40% 换 `mod=11`）
9. S1 执行：mod 97/13/11 各 10 道 `d=1`，只跑 DA，`max_tokens=64`
10. markdown：S2——先按 S1 结果改 `MOD`，找准确率掉到 30–50% 的 `d`
11. S2 执行：`MOD=97`、`DEPTHS=(1,2,3,4,6,8,10)`、每档 10 题、`max_tokens=128`
12. markdown：S3——同模型同题，只改系统提示（DA vs CoT），并加并行对照组
13. S3 执行：串行 `DEPTHS=(2,4,6,8)`、并行 `w=(3,5)`、每组 20 题，
    输出 DA/CoT/Delta 表
14. 画 `Delta = CoT - DA` 随 `d` 的曲线（serial vs parallel），存 `delta_vs_depth.png`
15. markdown：debug 指引（格式不遵循 / 提示词失效 / CoT 被截断）
16. 打印原始输出片段，便于排查
17. markdown：结果判读表（假设成立 / 任务太简单 / 任务太难 / 提示词没生效 / 抽取失败）+ 需要回报的三样结果
18. `infer2` + `analyse`（v2）：按条件硬截断（DA 32、CoT 768）、用 EOS 下标算真实 `ntok`、
    `W` 配平 `(d,w) = (3,2),(5,3),(7,4),(9,5)` 使 `d == 2w-1`，输出含 `err_cut` 的对比表

---

## 6. 代码风格与协作约定

1. **一切实验代码写在 notebook 里**；`main.py` 只是占位。新增独立脚本前先说明理由。
2. 注释与讨论用**中文**，变量名/函数名/文件路径用**英文**；已有命名风格
   （`make_serial`、`q_matrix`、`ds_ifeval`、`recs_s3`）请沿用。
3. 数据流是「生成 → 落盘 JSONL → 用 `datasets` 回读」，不要长期只在内存里传递 DataFrame。
4. 结果记录：重要结论写进 `record.md`；报错与解法写进 `vllm_problems_solutions.md`。
   新踩的坑按「问题 → 原因 → 解决方法」的格式追加。
5. 每个 notebook 末尾通常留一个空 cell，属于既有格式，不要当成垃圾清掉。
6. 改 notebook 时保留 `nbformat 4.5`、cell `id`、已有 outputs；不要用「全部重写」的方式覆盖，
   尽量只改需要改的 cell。改完跑一次第 2.6 节的 JSON 校验。
7. 不要重跑并提交带大数据输出的 cell（例如全量生成、GSM8K 全量分析），会让 diff 无法阅读。
8. 用中文提交信息（现有 git log 都是中文，`main` 分支，直接提交）。

### 6.1 禁止事项

- ❌ 在 Windows 原生环境跑 vLLM（必须 WSL）
- ❌ 改动第 3 节的加载 / 采样参数而不更新本文件与 `record.md`
- ❌ 把档位 A 的 `gpu_memory_utilization=0.80 / max_model_len=8192` 直接套到 4B 上
- ❌ 对 `Qwen3-4B-Instruct-2507` 传 `chat_template_kwargs={"enable_thinking": ...}`
- ❌ 再用 thinking 开关制造第三个条件；DA / CoT 只能由 system prompt 区分
- ❌ 用后处理硬截断制造 DA / CoT 的「预算公平」
- ❌ 提交 `models/`、`gsm8k_dataset/`、`ifeval/`、`.venv/`（已在 `.gitignore`）
- ❌ 删除或覆盖 `synthesis_q_k_dataset.jsonl` 而不重新生成、不说明原因
- ❌ 在没有 GPU 的机器上执行需要 CUDA 的 cell 后把报错当成代码 bug 修

---

## 7. 已知坑（详见 `vllm_problems_solutions.md`）

| 现象 | 原因 | 处理 |
|---|---|---|
| `Free memory ... (6.95/8.0 GiB) ... desired (0.92/0.90)` | 显存被前一个 transformers 模型占着 | `del` + `gc.collect()` + `torch.cuda.empty_cache()`，或用 `gpu_memory_utilization=0.80` 并重启 Kernel |
| KV cache 不足（需 4.38 GiB，只有 1.89 GiB） | 模型默认 `max_model_len=40960` | 显式设 `max_model_len=8192` |
| 4B 模型一加载就显存不足（`Free memory ... desired (0.80, 6.4 GiB)`） | bf16 权重约 7.5 GiB，本来就超过 8GB 卡的预算 | 按第 3.3 节换 fp8 / int4 / 换机器；**不要**靠降利用率硬试 |
| Triton 报 `Python.h: No such file` | 缺 Python 3.14 开发头文件 | `sudo apt install -y python3.14-dev` |
| `Could not find nvcc and default cuda_home='/usr/local/cuda'` | 只装了 CUDA runtime，没有 Toolkit | `VLLM_USE_FLASHINFER_SAMPLER=0`（在 `import vllm` 之前） |
| 输出被 `<think>` 段占满 / 格式解析失败 | Qwen3 默认开 thinking（仅档位 A） | `chat_template_kwargs={"enable_thinking": False}`；档位 B 不传该参数 |
| 模型把约束里的示例原样抄进答案（`321`、字面 `<answer>`） | 约束里放了完整的答案模板 | 改成抽象描述；见第 4.4 节 |
| Windows 下 `import vllm` 失败 | 无 Windows 版 vLLM | 切到 WSL |

---

## 8. 待办与未定项

**最终实验落地前必须先做完的：**

- 下载 `Qwen/Qwen3-4B-Instruct-2507` 到 `models/Qwen3-4B-Instruct-2507`；按第 3.3 节在目标机器上标定
  `quantization` / `gpu_memory_utilization` / `max_model_len` / `max_num_seqs`，回填表格并写进 `record.md`。
- 新建正式实验 notebook（目前还没有）：实现第 4 节的 DA / CoT 拼装、最终答案提取、
  `da_violation` / `cot_violation` 统计、长度与截断统计；建好后登记到第 5.1 节。
- 约束集重设计：C3（小数英文词）和 C9（数字和）当前接近 0，不能继续当 cumulative `k` 的硬门槛；
  先做**单约束校准**，只把有区分度的约束放进 `k` 轴。
- 新约束不得包含可照抄的完整答案模板（C5 的 `123 -> 321`、C6 的 `<answer>` 都污染过 k≥6 的数据）。
- 给 DA / CoT 分别校准 `max_tokens`，并在结果里记录 `finish_reason`、实际长度和截断率。

**已确认、不要再重复讨论的：**

- 不再做 thinking on/off 对比；1.7B 的 `pilot_math_results*.jsonl` 只作历史，其中 k≥6 的 no-thinking
  数据被 `<answer>` 模板污染，不可用。
- `W` 配平（serial vs parallel）保留，用来拆开深度 `d` 和工作量 `W`。

**遗留小问题：**

- `Experiments/gsm8k_test.ipynb` 第 8 个 cell 的 IFEval 保存逻辑有 bug（遍历了 `ds` 而非 `ds_ifeval`）。
- `configs/linux-gpu/requirements.txt` 被 README 引用但尚未创建。
- `top_k=20` 未传入 `SamplingParams`，需要决定是否统一。
- vLLM 批处理（`LLM.chat` 一次传多条 messages）尚未在正式实验中启用，启用前需重新标定显存。
- `Experiment.ipynb` 与 `Experiments/` 下的 notebook 是历史参考，不要当作正式实验的接口或数据格式依据。
