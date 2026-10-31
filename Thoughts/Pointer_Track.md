# 指针追踪（Pointer Tracking）数据集设计

> 用途：作为 CoT 实验的**主任务族**，替代模运算链。
> 核心优势：**每一步的动作完全相同**，深度 `d` 是唯一在变的量。

---

## 0. 一句话

给一张「每格指向下一格」的表，从某格出发沿表走 `d` 步，问最后停在哪个标签。

```
表： A -> F    B -> K    C -> A    D -> H    E -> C
     F -> M    G -> B    H -> E    I -> D    J -> L
     K -> I    L -> G    M -> J    ...

题目：从 A 出发，沿表走 4 步，最后停在哪个字母？

       A → F → M → J → L
       第1步 第2步 第3步 第4步
答案：L
```

**每一步都是"查一次表"**——没有加减乘除，没有取模。

---

## 1. 为什么换掉算术任务

在模运算链（`x → +3 → ×2 → −5 → …`）上跑 pilot 时，发现三个混淆：

| 混淆 | 表现 | 指针追踪怎么解决 |
|---|---|---|
| **① 单步难度不均** | `+3` 和 `×47` 差一个数量级。同是 `d=5`，抽到容易运算的题和抽到难的题，准确率能差一倍 | **每步完全同构**，单步难度是常数 |
| **② 忘记取模** | DA 条件里只有 **30%** 的轨迹真的做了 `mod 97`，CoT 是 96%。这个差异和"是否外化推理"无关，混进了 `Δ` 里 | 没有 mod，值天然在合法范围内 |
| **③ 值溢出** | 大数乘法会溢出成 5 位数（如 `5630`），模型答出界外的数 | 中间值永远是表里的标签 |

**结论**：算术任务的 `d` 里混着"每步有多难"，而指针追踪把这一项固定住了。

---

## 2. 形式化定义

一张表就是一个函数 `f: L → L`，`L` 是标签集合，`|L| = N`。

- **串行任务**：给定起点 `s`，求 `f^d(s)`（`f` 的 `d` 次迭代）
- **并行任务**：给定 `w` 个起点 `s_1..s_w`，求 `[f(s_1), …, f(s_w)]`

**三个量**（都由构造直接确定，不需要任何解析或标注）：

| 量 | 串行版 | 并行版 |
|---|---|---|
| 串行深度 `d` | = 步数 | = 1 |
| 并行宽度 `w` | = 1 | = 起点个数 |
| 工作量 `W` | = `d`（`d` 次查表）| = `w`（`w` 次查表）|

**注意**：串行版**没有**"合并"步骤，所以 `W = d` 而不是 `2d−1`。这让 W 配平变成精确的 `d = w`。

---

## 3. 三个关键设计决策

### 决策 1：固定 `N`，不要随 `d` 变化 ⚠️

如果表的大小随 `d` 增长（`N = d+2`），那么**题目文本长度**也在随 `d` 增长——这会把"深度"和"prompt 长度"混在一起。

**做法**：`N` 固定为 24（`A`–`Z` 去掉易混的 `I`、`O`），`d` 在 1–16 之间变化。

这样所有题的**表长完全一致**，`d` 是唯一变量。

### 决策 2：保证 `d` 步内节点不重复

如果路径上有环，模型可能算出"周期性"而抄近路。而且重复访问会让"走了几步"变得难数。

**做法**：先随机生成一条长度为 `d+1`、**节点互不重复**的随机游走，再按这条游走填表。用 `rng.sample` 即可保证。

### 决策 3：标签避开易混字符

去掉 `I`（像 1）和 `O`（像 0），避免"看错标签"这类与深度无关的错误。

```
LABELS = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # 24 个
```



## 5. 生成算法函数需求

```python

LABELS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")   # 24 个，去掉 I / O


def make_table(rng, N, forced=None):
    """每个节点恰好一条出边，无自环。forced 用于种植指定路径。"""


def make_serial(rng, N, d):
    """串行：从 walk[0] 走 d 步到 walk[d]，游走节点互不重复。"""


def make_parallel(rng, N, w):
    """并行：w 个起点各走 1 步，答案是目的地有序列表。"""

```

---

## 6. Prompt 规范

**关于输出格式的提示词要放到约束1中**，因此需要更新约束列表

### 串行版

```
<系统提示>

The table below defines a single pointer for each entry: from an entry, the
pointer tells you which entry comes next.

A -> F
B -> K
C -> A
...（共 24 行）

Start at A. Follow the pointers for exactly 4 steps.
Question: which entry do you end at?
<约束（若有）>
End your reply with: Final answer: <one label>
```

### 并行版

```
...（同一张表）

Start at these entries: B, D, G. From each one, follow the pointer exactly
ONE step.
Question: what are the resulting entries, in the order given?
End your reply with: Final answer: <label>, <label>, <label>
```

### 两个条件只改系统提示

```python
SYS_DA = ("You are a precise lookup engine. Do the lookups silently and output "
          "only the final answer. Do not explain your steps.")

SYS_COT = ("You are a precise lookup engine. Work through the pointers one at a "
           "time, writing each intermediate entry, then state the final answer.")
```


---

## 7. 判定与答案抽取

```python
import re

SERIAL_RE = re.compile(r'Final answer:\s*[*]*\s*([A-Z]{1,2})')
PARA_RE   = re.compile(r'Final answer:\s*([A-Z]{1,2}(?:\s*[,，]\s*[A-Z]{1,2})*)')


def judge_serial(txt, gold):
    m = SERIAL_RE.search(txt)
    return m is not None and m.group(1) == gold


def judge_parallel(txt, gold_list):
    m = PARA_RE.search(txt)
    if not m:
        return False
    return re.findall(r'[A-Z]{1,2}', m.group(1)) == gold_list
```

**为什么比算术简单**：答案空间是离散标签，不需要数值归一化（英文数词→数字那类问题不存在）。


## 9. 四个坑

### ① 答案空间小，猜对率被抬高 ⚠️

`N=24` 时乱猜约 4%；算术题（0–96）只有 1%。

**后果**：DA 的地板被抬高，深度效应在底部会被压缩。

**对策**：`N` 保持 24（猜对率 4% 可接受）。不要再降到 8–12，那会让猜对率到 8–12%。

### ② 表占 prompt 空间

`N=24` 的表约 24 × 10 token ≈ 240 token 固定开销。可接受，且**因为 `N` 固定，这份开销对所有 `d` 相同**，不构成混淆。

### ③ 理论上的"置换复合"捷径

`d` 次迭代一个函数，理论上可以用**反复平方**（compose `f` 与自身）在 `O(log d)` 步算出来。这和模运算链的"仿射复合"是同一类漏洞。

**但代价完全不同**：复合要求把**整张表**当成代数对象操作；顺着走只要看当前一格。

> 实践上安全。报告里写一句即可，不必专门做实验。

### ④ 每步"太容易"

指针追踪的每一步都极简单（看一眼表），深度困难全来自**累积**。

**这不是缺陷，是特性**：它测的正是"把中间状态外化到输出流"的价值——模型能写下中间节点，就不需要内部记住路径。

**副作用**：如果模型本来就能把 `d=5` 的路径在脑子里过完，`Δ` 会接近 0。

