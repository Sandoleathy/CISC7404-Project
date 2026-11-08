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

---

## 2. 环境与常用命令
### 2.1 命令

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

### 2.4 目录

```
.
├── README.md                     安装 / 启动说明（英文）
├── AGENTS.md                     本文件
├── task.md                       当前两个任务的执行清单（问题集 / 约束集）
├── record.md                     研究计划与实验记录（中文，持续更新）
├── Proposal.pdf                  课程项目开题报告
├── pyproject.toml / uv.lock      uv 依赖定义与锁文件
├── uv.toml                       镜像源
├── prior_investigation_kaggle.ipynb   前期调研（Kaggle 模运算 pilot）
├── configs/
│   ├── cpu/requirements.txt
│   ├── linux-gpu/               空目录（README 有引用，尚未填依赖）
│   └── windows-gpu/requirements.txt
├── Experiments/                  实验代码与结果
│   ├── data_generation/          合成数据生成
│   │   ├── synthesis_dataset.ipynb          pointer tracking 生成（串行 d×k / 并行 d×w）
│   │   ├── synthesis_q_k_dataset.jsonl      串行 d×k，4000 行
│   │   ├── synthesis_d_w_dataset.jsonl      并行 d×w，4000 行
│   │   ├── synthesis_math_dataset.ipynb     小数加减链生成
│   │   └── synthesis_math_dataset.jsonl     小数加减链，4000 行
│   ├── argmin_math_experiment/   min-chain 实验
│   │   ├── experiment_math.ipynb            主实验 notebook
│   │   ├── pilot_math_results.jsonl         1.7B / 2048 token 的旧跑结果
│   │   ├── pilot_math_results_v2.jsonl      1.7B / 6144 token 的 v2 结果
│   │   ├── pilot_results.jsonl              pointer tracking pilot 结果
│   │   └── vllm_problems_solutions.md       vLLM 踩坑与解决方法（中文）
│   ├── vLLM_test.ipynb          以下 4 个为早期 notebook，历史参考
│   ├── gsm8k_test.ipynb
│   ├── Qwen3-4B-Test.ipynb
│   ├── experiment_pointer_track.ipynb
│   ├── experiment_calibration_v2(1).ipynb   4B 校准 v2 的实际代码（未提交）
│   ├── calib_v2_results_tierB.jsonl         校准 v2 原始结果（未提交，2420 条）
│   └── calibration_v2_summary_tierB.csv     校准 v2 聚合表（未提交）
├── gsm8k_dataset/                gitignore；由 notebook 生成
├── ifeval/                       gitignore；由 notebook 生成
├── models/                       gitignore；Qwen3-1.7B 已下载；4B 权重在 Colab，本地不下载
└── Thoughts/                     零散想法（Pointer_Track.md）
```

---

## 3. vLLM 统一加载参数（适用于RTX3070，如果是更好GPU可以自行调节）

**要求：所有 vLLM 推理代码都从本节模板出发，不要从零手写 `LLM(...)`。**
采样参数全项目统一；

### 3.1 两个档位

| 档位 | 模型 | 本地目录 | 用途 | 状态 |
|---|---|---|---|---|
| A | Qwen3-1.7B | `models/Qwen3-1.7B` | pilot、链路自测、回归 | 已下载，已跑通 |

- HF repo id 是 `Qwen/Qwen3-4B-Instruct-2507`。**不要**换成 `Qwen/Qwen3-4B`（base）或
  `Qwen/Qwen3-4B-Thinking-2507`（thinking-only）：base 版没有对齐好的对话能力，thinking 版会
  强制自带思考块，两者都破坏「CoT 由 prompt 控制」的设计。
- 档位 B 是**非 thinking 模型**：`tokenizer_config.json` 里的 chat template **没有**
  `enable_thinking` 分支，generation prompt 就是 `<|im_start|>assistant\n`。
  因此对档位 B **不要**传 `chat_template_kwargs={"enable_thinking": ...}`。

### 3.2 档位 A 模板（Qwen3-1.7B，RTX3070能够运行，其他GPU不必一定遵守）

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


---

## 4. DA / CoT：只由 system prompt 决定

**最终实验没有 thinking 开关。** 档位 B 本身就不是 thinking 模型，条件只有两个，全部靠提示词区分。
这也是本项目「CoT 是否有用」的唯一操作化定义。

| 条件 | system prompt 里的指令 | 期望输出 |
|---|---|---|
| `DA`（direct answer） | 不写推理、不解释，直接给最终答案 | 只有最终答案（仍要满足 C1–Ck） |
| `CoT`（chain-of-thought） | 先显式写出每一步的中间结果，再给最终答案 | 推理 trace + 最终答案 |


## 5. 代码风格与协作约定

1. **一切实验代码写在 notebook 里**
2. 注释与讨论用**中文**，变量名/函数名/文件路径用**英文**；已有命名风格
   （`make_serial`、`q_matrix`、`ds_ifeval`、`recs_s3`）请沿用。
3. 数据流是「生成 → 落盘 JSONL → 用 `datasets` 回读」，不要长期只在内存里传递 DataFrame。
4. 结果记录：重要结论写进 `record.md`。
   新踩的坑按「问题 → 原因 → 解决方法」的格式追加。
5. 不要重跑并提交带大数据输出的 cell（例如全量生成、GSM8K 全量分析），会让 diff 无法阅读。
6. 用中文提交信息（现有 git log 都是中文，`main` 分支，直接提交）。

### 5.1 禁止事项

- ❌ 在 Windows 原生环境跑 vLLM（必须 WSL）
- ❌ 改动第 3 节的加载 / 采样参数而不更新本文件与 `record.md`
- ❌ 把档位 A 的 `gpu_memory_utilization=0.80 / max_model_len=8192` 直接套到 4B 上
  （档位 B 的 `0.90 / 8192 / 32` 是 T4 专属值）
- ❌ 在 T4 上忘了 `dtype="half"`（Turing 不支持 bf16）
- ❌ 对 `Qwen3-4B-Instruct-2507` 传 `chat_template_kwargs={"enable_thinking": ...}`
- ❌ 用后处理硬截断制造 DA / CoT 的「预算公平」
- ❌ 把 min-chain 任务再当正式任务跑全曲线（已证明无可用深度区间，见第 1 节）
- ❌ 提交 `models/`、`gsm8k_dataset/`、`ifeval/`、`.venv/`（已在 `.gitignore`）
- ❌ 在没有 GPU 的机器上执行需要 CUDA 的 cell 后把报错当成代码 bug 修

### 5.5 Task A v3：寄存器任务（代码就绪，待 GPU 校准）

- 入口：`Experiments/task_a/experiment_task_a.ipynb`。按用户确认，本次先交付可在 Colab/T4
  运行的 notebook，真实校准稍后运行；**尚未选定正式任务，不能声称 p1 或深度曲线达标**。
- 首选双寄存器条件更新；每步比较旧值，再同时一加一减，等号走 otherwise。两个增量不相等，
  避免总和守恒捷径；寄存器在编码基数范围内，禁止 mod / 截断 / 聚合。
  双寄存器默认 `ans=A*100+B`，三寄存器候选保留所有状态的编码。
- `W` 是提案约定的复合更新次数，另记 `W_atomic=3W`（一次比较、两次加减）。
  并行独立链长度为 `[1,2,...,2]`，`W=2w-1`，实际 `d=1`（w=1）或 `2`，
  `matched_serial_d=2w-1`。仅匹配串行奇数深度 1..19；w=1 是退化对照。
  并行输出每条链的答案，需人工评估读出长度混杂，不能把配平当作无混杂因果对照。
- 顺序：100 条 DA 单步探针 → 独立的 5 深度 × 20 题 × DA/CoT → **人工审核** → 正式生成。
  默认关闭 GPU 与正式生成；复用第 3 节的 A 档和既有 T4 B 档加载/采样设置。
  A 档结果仅自测，不用于 B 档定题。模拟结果不得写入或冒充真实校准。
- `task.md` 的 DA(1) 0.80–0.95 与探针 0.955–0.986 不相交，分别报告。
  `p1**20` 仅为启发式预测；Cochran–Armitage 检验对象明确为 CoT 准确率随深度下降，
  gap 端点差另用配对 bootstrap 报区间。不能将该检验说成 gap 趋势的检验。
- 真实结果写 `Experiments/task_a/calib_v3_task.jsonl`（断点续跑，保留原文、模型版本、采样参数、
  answer_ok、解析和截断诊断）；题目和摘要写 `work/task_a/`。
  审核后才写 `Experiments/data_generation/synthesis_register_dataset.jsonl` 与 `.manifest.json`，
  默认 2000 串行 + 1000 并行；正式题不与校准题重叠。
- 查表字段为各链的 `START / OPS / FIRST / LAST / MAXDELTA`；`NEG` 不适用，不能机械迁移。
  此处仅提供字段，不做 Task B 的约束筛选。实测后须回填本节、`record.md` 和 `task.md`。

## 6. 待办与未定项

**一周双轨（2026-11-07 起，详见 `record.md`「一周行动计划」）：**

- **轨道 1：问题集** —— 找一个 CoT 在 `d=1..20` 上有可测梯度的任务。
  - 第一天对 3 个候选任务各跑 `d=1 × 100 题 × DA`，量 p1；目标窗口 `0.955–0.986`。
  - 候选优先级：双寄存器条件更新（首选）> 编号数组指针 > 定长槽位覆盖写 > 栈操作。
  - 验收：`CoT(d=20) ∈ 0.40–0.75`、`DA(d=20) ≤ 0.10`（接受地板）、`gap(20) − gap(1) ≥ 0.30`、
    Cochran-Armitage `p < 0.01`。
- **轨道 2：约束集** —— 找 10 条能进 k 轴、且在 CoT 侧有成本的约束。
  - 只用指令遵循判分，不看问题正确率；测试床用 min-chain 的 `d=5..10`。
  - 三道门槛（第 4.2 节）：非空转、DA 单条 ≥ 0.95、CoT 比 DA 低 ≥ 0.15。
  - 主力是查表型：`START` / `OPS` / `FIRST` / `LAST` / `NEG` / `MAXDELTA`。
- **合并复核**：新任务 × 10 约束 × 2 模式 × 100 题 = 2000 条，不可假设轨道 2 的结论直接迁移。


