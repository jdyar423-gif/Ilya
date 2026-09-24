# Ilya：超越 Jev 的"开放世界 · 随时 · 一致"类型化决策模型

> **一句话**：Jev 把决策做成"带置信度的分类"；Ilya 把决策做成**带契约的信念**。它知道自己不知道（NONE），
> 对选项的顺序与命名严格不变，按需多想，服从调用方声明的逻辑，并把置信度变成有统计保证的承诺。

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
| **W1 永远会回答**：蛋糕食谱被判为"技术问题"（置信 0.94）；没有"以上都不是"选项时，30 条越界消息 0 条被标出 | 每个问题都有与候选竞争的**隐式 NONE 结果**（choice / score / noul 都适用），并用越界样本训练 | 构造性（NONE 永远可表达）+ 经验性（检测率） |
| **W2 置信度不是契约**：阈值 0.50–0.95 之间准确率几乎持平，只有 0.99 才有用 | 每步都用严格恰当评分规则训练；温度缩放；**Learn-then-Test 认证门限**（以 ≥1−δ 的概率保证被接受答案的错误率 ≤ ε）；**共形预测集** | 统计性（有限样本、与分布无关） |
| **W3 对呈现方式敏感**：选项顺序可移动 13 个百分点；名称与 rubric 互换后 AUROC .81→.58 | **候选即集合**（无位置编码，逐候选读出）→ 对顺序精确等变；**rubric 权威**：有 rubric 时名称根本不进入模型；**证据由问题驱动**：选项不去上下文里"找自己" | 构造性（精确到浮点误差） |
| **W4 问题之间不一致**：平均分相同，个体判决却会随请求配置翻转 | 问题工作区**互相隔离**（答案与同批其他问题无关）；调用方声明的逻辑约束通过**精确条件化**施加，返回满足全部约束的联合 MAP 与"支持度"自相矛盾警报 | 构造性 |
| **W5 固定算力，不会深想**：计数、多跳都不可靠，需要外部 LLM | 权重共享的**循环 anytime 核心**：每一步的信念都是校准的后验；按置信度逐题停机，可以比训练时跑得更久 | 经验性 |

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
（运行 `python -m ilya.evaluate` 后自动填入）
<!-- /RESULTS -->

## 4. 快速开始

```bash
pip install -e '.[dev]'          # torch + numpy (+ pytest)
python -m pytest -q              # 40 个测试：精确不变性、条件化 = 暴力枚举、统计保证、API

# 训练（三者使用完全相同的设置；CPU 上各约 1–2 小时）
COMMON="--rope --ident --lr 7e-4 --warmup 300 --batch 128 --group 16 --steps 15000 --curriculum 0.5"
python -m ilya.train --model ilya     $COMMON --t_min 4 --t_max 8 --out checkpoints/ilya.pt
python -m ilya.train --model jev      $COMMON --out checkpoints/jev.pt
python -m ilya.train --model jev_none $COMMON --out checkpoints/jev_none.pt

# 正面对照评测 → results/results.json + results/REPORT.md
python -m ilya.evaluate --ilya checkpoints/ilya.pt --jev checkpoints/jev.pt --jev_none checkpoints/jev_none.pt

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

## 6. 路线图

1. 把 Ilya 的工作区、读出与保证层接到预训练 LLM / 视觉语言骨干上（上下文流由骨干的前缀 KV 充当）。
2. 用 RLCD 风格的强化学习在真实工作负载上继续校准，并在部署分布漂移时使用加权共形。
3. 把条件化一致性层扩展到可学习的因子（在不破坏问题隔离的前提下）。

## 参考

Jev 相关出处见 [`docs/JEV_RESEARCH.md`](docs/JEV_RESEARCH.md)。方法上的前人工作：循环深度 / 潜在推理
（Universal Transformers；Geiping et al. 2025）、outlier exposure（Hendrycks et al. 2019）、Set Transformer（Lee et al. 2019）、
Learn-then-Test 与共形预测（Angelopoulos & Bates）、温度缩放（Guo et al. 2017）、阅读理解中的精确匹配特征（DrQA, Chen et al. 2017）。
