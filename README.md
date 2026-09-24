# Ilya：针对 Jev 结构性弱点的"开放世界 · 呈现不变 · 可认证"类型化决策模型

> **一句话**：Jev 把决策做成"带置信度的分类"；Ilya 把决策做成**带契约的信念**：默认就能说"以上都不是"（NONE），
> 对选项的顺序与命名严格不变，服从调用方声明的逻辑，并把置信度变成有统计保证的承诺。

2026 年 9 月 15 日，TypeSafe AI 发布了 Jev，号称第一个 "System One model"：不生成文本，只返回
`choice / score / noul` 这类类型化值和概率。一周之内，独立评测（PriorBench、Awesome-Jev 综述，
以及 8 篇以上 arXiv 预印本）揭示了它的结构性弱点。Ilya 针对每一个弱点给出**机制层面**的解决方案，
并在同预算、同数据、同训练流程下与 Jev 范式的忠实复刻版做了正面对照。

- Jev 深度研究笔记（机制、证据、弱点、出处）：[`docs/JEV_RESEARCH.md`](docs/JEV_RESEARCH.md)
- 架构与关键发现：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- 完整对照报告（由代码自动生成）：[`results/REPORT.md`](results/REPORT.md)

## 1. 五个范式突破

| Jev 的已知弱点（独立证据） | Ilya 的机制 | 保证类型 |
|---|---|---|
| **W1 永远会回答**：蛋糕食谱被判为"技术问题"（置信 0.94）；没有"以上都不是"选项时，30 条越界消息 0 条被标出 | 每个问题都有与候选竞争的**隐式 NONE 结果**（choice / score / noul 都适用，调用方无需记得加选项），并用越界样本训练 | 构造性（NONE 永远可表达）+ 经验性（检测率，见下方结果与局限） |
| **W2 置信度不是契约**：阈值 0.50–0.95 之间准确率几乎持平，只有 0.99 才有用 | 每步都用严格恰当评分规则训练；温度缩放；**Learn-then-Test 认证门限**（以 ≥1−δ 的概率保证被接受答案的错误率 ≤ ε）；**共形预测集** | 统计性（有限样本、与分布无关） |
| **W3 对呈现方式敏感**：选项顺序可移动 13 个百分点；名称与 rubric 互换后 AUROC .81→.58 | **候选即集合**（无位置编码，逐候选读出）→ 对顺序精确等变；**rubric 权威**：有 rubric 时名称根本不进入模型；**证据由问题驱动**：选项不去上下文里"找自己" | 构造性（精确到浮点误差） |
| **W4 问题之间不一致**：平均分相同，个体判决却会随请求配置翻转 | 问题工作区**互相隔离**（答案与同批其他问题无关）；调用方声明的逻辑约束通过**精确条件化**施加，返回满足全部约束的联合 MAP 与"支持度"自相矛盾警报 | 构造性 |
| **W5 固定算力，不会深想**：计数、多跳都不可靠，需要外部 LLM | 权重共享的**循环 anytime 核心**：每一步的信念都是校准的后验；按置信度逐题停机，可以比训练时跑得更久而不崩溃 | 经验性（本实验中**多迭代并未带来额外准确率**，见局限） |

## 2. 架构一览

```
context ──► 共享上下文流（与问题无关，只编码一次；权重共享的双向块循环 T 次）
                 │  只读，且只有"问题槽 / 答案槽"会读
question ──► [指令 token ; ANS/NONE ; 候选集合] ──► 循环工作区 ──► 每步读出 softmax([NONE, c1..cK])
                                                              └─ 置信度 ≥ τ 即停机
beliefs ──► 条件化一致性层（声明的约束）──► 统计保证层（认证门限 / 共形集合 / 代价敏感动作）──► 类型化答案
```

详见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 3. 实验：与 Jev 范式复刻版的公平对照

**为什么是复刻版**：TypeSafe 没有公开 Jev 的架构和权重。所有开源 Jev 式实现（simple-jev、jev-visual、OpenJev、AnyJev）
以及 Visual Jev 论文共享同一机制：因果语言模型读取"上下文 + 问题 + 选项"，上下文作为共享前缀只算一次，
每个问题是一个隔离后缀，在答案位读取选项 token 的 logits 并在给定选项上做 softmax，
用恰当评分规则训练使概率在分布内校准。`ilya/jev.py` 精确实现了这一机制（含前缀 KV cache）。

**公平性**：同一数据生成器、同一课程、同样的批大小 / 优化器 / 学习率计划 / 步数；参数量接近；
两个家族获得完全相同的检索先验（RoPE + 恒等感知注意力）。`Jev + NOTA` 变体总是附加显式的
"none of these" 选项，并使用与 Ilya 相同的越界训练数据（比真实 Jev 更强：真实 Jev 的 score / noul 无法附加选项）。
所有模型都先在各自的校准集上做温度缩放，再计算任何指标。

**为什么是合成世界**：本环境没有 GPU，且 Hugging Face / arXiv 被网络策略拦截，无法使用预训练权重。
合成世界的好处是真值完全已知，所以准确率、校准、越界检测、顺序不变性、名称绑定、多跳深度、约束一致性
都可以被精确测量。

<!-- RESULTS -->
### 结果摘要（自动生成，详见 [`results/REPORT.md`](results/REPORT.md)）

| 指标 | Ilya | Jev 复刻（封闭世界） | Jev 复刻 + 显式 NOTA 选项 |
|---|---|---|---|
| 分布内准确率 | 89.8% | 82.8% | 83.8% |
| 分布内 ECE（越低越好） | 0.012 | 0.014 | 0.022 |
| 越界输入判为 NONE 的比例（5 类平均） | 84.2% | 0.0% | 95.2% |
| 越界输入上的高置信错误（≥0.9，5 类平均） | 7.5% | 14.1% | 0.3% |
| 全新越界类型检测 AUROC（训练中从未出现） | 0.964 | 0.761 | 0.996 |
| 选项洗牌后判决翻转率 | 0.0% | 0.7% | 6.1% |
| 误导性选项名下的准确率 | 89.9% | 4.3% | 3.4% |
| 按名称而非 rubric 作答的比例 | 4.4% | 88.9% | 88.2% |
| 约束违反率：隔离判决 → 条件化后（颜色束） | 14.2% → 0.0% | 64.6% → 0.0% | 14.0% → 0.0% |
| 约束违反率：隔离判决 → 条件化后（计数束） | 52.0% → 0.0% | 92.8% → 0.0% | 42.8% → 0.0% |
| 认证覆盖率（部署混合，风险 ≤5% @ 90%） | 85.1% | 0.0% | 82.7% |
| 长链泛化：7–8 个物体的关系题（训练最多 6 个） | 79.1% | 80.0% | 79.4% |
| 吞吐（每次调用 32 题，决策/秒，CPU） | 821 | 1690 | 1438 |
<!-- /RESULTS -->

### 结果解读（逐条对照，不回避不利结果）

**Ilya 明确胜出的地方**
- **呈现不变性（构造性）**：选项洗牌后 Ilya 的概率逐位不变（翻转率 0%，最大总变差 0.0000）；
  复刻版翻转 0.7%–6.1%，单题概率最多移动 0.72。
- **rubric 才是契约**：选项名与 rubric 矛盾时，复刻版准确率跌到 3.4%–4.3%，约 88% 的时候按**名称**作答
  （与论文 "Type-Safe Is Not Error-Free" 对托管 Jev 的观察一致）；Ilya 维持 89.9%，与诚实命名时完全相同。
- **分布内准确率与学习速度**：同样数据、同样步数，Ilya 89.8%（可回答题 96.4%），复刻版 82.8% / 83.8%；
  差距主要来自颜色 / 形状这类选项名是代号或同义词的题（Ilya 约 91%，复刻版约 72%）。
  Ilya 在第 4000 步就达到 87%，复刻版此时约 64%–66%。
- **对封闭世界的 Jev**：复刻版在所有越界输入上 0% 回答 NONE，并在 5 类越界上平均有 14% 的高置信（≥0.9）错误；
  在部署混合流量上**无法得到任何认证门限**，Ilya 可以在风险 ≤5% 的保证下自动回答 85.1% 的流量。

**基本持平的地方**
- **校准**：温度缩放后三者 ECE 都在 0.01–0.02。
- **显式 NOTA 选项 + 同样的越界训练**能让复刻版在越界检测和认证覆盖上追平 Ilya（82.7% vs 85.1%）。
  也就是说，"隐式 NONE"的价值主要在于：它对 choice / score / noul **全部题型默认存在**，而真实 Jev 的 score / noul 无法附加选项。
- **一致性层**是与模型无关的：它让每个模型的约束违反率都降到 0，并提高准确率（Ilya 颜色束 90.2%→94.6%，
  封闭复刻版 78.2%→94.8%）。它是范式的一部分，也可以直接加在 Jev 上。
- **长链泛化**（7–8 个物体）：三者都在 79%–80%，并且都呈现"远距离好、近距离差"的同一模式，说明都学到了依赖长度的启发式，而不是真正的逐跳推理。

**Ilya 更差的地方**
- **全新类型的乱码上下文（soup）**：Ilya 只有 46.8% 回答 NONE，30.4% 高置信错误；显式 NOTA 的复刻版是 94.2% / 1.5%。
- **吞吐**：在同一台 4 核 CPU 上，Ilya 约 821 决策/秒，复刻版 1438–1690 决策/秒（Ilya 平均用 3.4 次循环迭代）。
- **"想得更久"没有带来收益**：Ilya 的准确率在第 2 次迭代就饱和，之后到 32 次都保持稳定（不会崩溃），
  自适应停机也确实把更多迭代花在不确定的题上，但额外迭代并没有把这些题做对。本实验**没有**证明循环核心的深度推理优势。

## 4. 快速开始

```bash
pip install -e '.[dev]'          # torch + numpy (+ pytest)
python -m pytest -q              # 40 个测试：精确不变性、条件化 = 暴力枚举、统计保证、API

# 仓库已附带训练好的 checkpoints/（每个约 3.4 MB），可以跳过训练直接评测
# 训练（三者使用完全相同的设置；4 核 CPU 上各约 1.5–2.5 小时）
COMMON="--rope --ident --lr 7e-4 --warmup 300 --batch 128 --group 16 --steps 15000 --curriculum 0.5"
python -m ilya.train --model ilya     $COMMON --t_min 4 --t_max 8 --out checkpoints/ilya.pt
python -m ilya.train --model jev      $COMMON --out checkpoints/jev.pt
python -m ilya.train --model jev_none $COMMON --out checkpoints/jev_none.pt

# 正面对照评测 → results/results.json + results/REPORT.md
python -m ilya.evaluate --ilya checkpoints/ilya.pt --jev checkpoints/jev.pt --jev_none checkpoints/jev_none.pt --readme README.md

# 为 API 校准（温度、共形阈值、认证门限；使用与所有评测不重叠的种子），写回 checkpoint
python -m ilya.calibrate --ckpt checkpoints/ilya.pt

# 演示：越界、误导性选项名、声明的逻辑
python -m ilya.demo --ckpt checkpoints/ilya.pt
```

### API（与 Jev 请求形状兼容，返回更丰富）

```python
from ilya.api import Engine
from ilya.coherence import iff

engine = Engine.load("checkpoints/ilya.pt")
out = engine.decide(
    state="alpha red . alpha cube . beta blue . alpha left beta .",
    questions={
        "color":   {"type": "choice", "instructions": "what color is alpha ?",
                    "criteria": {"x1": "crimson", "x2": "azure"}},      # rubric 决定语义，名称只是 id
        "size":    {"type": "score", "instructions": "how big is beta ?",
                    "criteria": ["tiny", "small", "medium", "large", "huge"]},
        "is_red":  {"type": "noul", "instructions": "is alpha red ?"},
    },
    constraints=[iff("color", "x1", "is_red", "true")],                  # 声明的逻辑被精确满足
)
# 每个答案: choice/score/noul, none (NONE 概率), probabilities, confidence, iterations,
#           support (约束一致性), 以及校准后的 set (共形集合) / accept (认证门限)
```

## 5. 诚实的边界

- 这是**从零训练的小模型（约 87 万参数）在合成世界上**的结果，不是在自然语言基准上击败托管的 Jev。
  结论的形态是：在相同条件下，Ilya 的**机制**消除了 Jev 范式的结构性失败，而这些机制与规模无关、可以移植到预训练骨干上。
- "构造性"保证（顺序不变、rubric 权威、问题隔离、约束一致）对任何权重都成立，由测试验证；
  "统计性"保证依赖可交换性（校准数据与部署数据同分布）；其余是本实验中的经验测量。
- 复刻版复现的是 Jev 公开的**机制**，不是其专有权重或 RLCD 训练细节。
- 两个家族都获得了人工设计的检索先验（恒等感知注意力），用来代替本环境无法获得的预训练。
- 复刻版的读出是"选项名 token"（jev-visual 的 native-token 读出）；按字母标签读出的版本在本预算下从零学不会（见 `docs/ARCHITECTURE.md`）。
- 每个模型只训练了一个随机种子；表中差距没有置信区间。越界检测与多步推理上的不足见上文"Ilya 更差的地方"。

## 6. 路线图

1. 把 Ilya 的工作区、读出与保证层接到预训练 LLM / 视觉语言骨干上（上下文流由骨干的前缀 KV 充当）。
2. 用 RLCD 风格的强化学习在真实工作负载上继续校准，并在部署分布漂移时使用加权共形。
3. 把条件化一致性层扩展到可学习的因子（在不破坏问题隔离的前提下）。

## 参考

Jev 相关出处见 [`docs/JEV_RESEARCH.md`](docs/JEV_RESEARCH.md)。方法上的前人工作：循环深度 / 潜在推理
（Universal Transformers；Geiping et al. 2025）、outlier exposure（Hendrycks et al. 2019）、Set Transformer（Lee et al. 2019）、
Learn-then-Test 与共形预测（Angelopoulos & Bates）、温度缩放（Guo et al. 2017）、阅读理解中的精确匹配特征（DrQA, Chen et al. 2017）。
