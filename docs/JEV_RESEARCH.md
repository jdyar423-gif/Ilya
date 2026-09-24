# Jev 深度研究笔记（截至 2026-09-24）

> 本文整理 TypeSafe AI 的 Jev 及其衍生工作的公开证据，是 Ilya 设计的出发点。
> 所有数字均为原作者 / 厂商 / 独立评测者自报，出处见文末。TypeSafe **未公开** Jev 的架构。

## 1. Jev 是什么

| 项目 | 事实 |
|---|---|
| 发布 | 2026-09-15 有限早期访问，同时宣布 DCVC 领投 4000 万美元种子轮 |
| 公司 | TypeSafe AI（旧金山，2024 年成立；创始人 Diogo Almeida（前 OpenAI）、Erik Gafni、Sasha Sheng） |
| 定位 | 自称第一个 "System One model"（取自 Kahneman 的快思考）：**不生成文本**，只返回**带概率的类型化值** |
| 接口 | `state`（上下文）+ `questions`；题型 `choice`（1–255 个选项）、`score`（有序等级）、`noul`（0–1 概率） |
| 训练 | RLCD（Reinforcement Learning for Calibrated Decisions）：奖励与"所报概率是否匹配实际正确频率"挂钩。官方材料**未指明**使用哪种 proper scoring rule |
| 采用 | Vercel：AI Gateway 史上采用最快的模型，24 小时内约 13% 付费团队使用 |
| 厂商宣称 | 比前沿 LLM 快 193.6×、便宜 444.6×（被批评非同类对比、无可复现工件） |

### 公开可推断的机制
所有开源复现（simple-jev、jev-visual、OpenJev、AnyJev、Vision-JEV）与 Visual Jev 论文都采用同一机制：

```
context  ──► 共享前缀（只算一次，KV cache）
question ──► 每个问题一个隔离后缀，批量执行
options  ──► 按位置分配标签 A/B/C…（或 0..9）
answer   ──► 在答案位读取候选标签的 logits，只在给定选项上做 softmax
```

Visual Jev（arXiv 2609.25845）：N=32 个问题时共享批处理比独立串行快 8.9×；答案监督后训练把宏观准确率从 70.6% 提升到 76.1%。

## 2. 独立证据揭示的结构性弱点

| # | 弱点 | 证据 |
|---|---|---|
| W1 | **封闭世界：永远会回答** | 蛋糕食谱被分类为"技术问题"置信 0.94；随机字母 0.97；不提供"以上都不是"选项时，30 条越界消息 **0 条**被标出，且置信 0.99（PriorBench） |
| W2 | **置信度不是契约** | 阈值 0.50–0.95 之间准确率几乎持平，只有 0.99 才跳到 100%（覆盖 60.2%）；校准随任务变化，存在高置信错误，概率被量化到两位小数（PriorBench、awesome-jev-survey F2） |
| W3 | **对呈现方式敏感** | 选项顺序可造成最多 13 个百分点的变动；名称–评分标准互换后 AUROC 从 .8146 跌到 .5806（"跟随选项名而非绑定的 rubric"，arXiv 2609.26758）；错误的 criteria 描述导致 16.7%，低于 25% 随机下限 |
| W4 | **问题之间没有一致性** | ContractNLI 上"平均分相近，个体判决却不同"，改变请求配置会翻转判决（arXiv 2609.27678） |
| W5 | **固定算力、不会深想** | 计数超过少数几个就不可靠；StarCraft II 中仅用 Jev 从不扩张，必须由 GPT-6 写计划（JEV-Star, 2609.27331） |
| W6 | **准确率上限** | 计算社会科学 15 个任务中 14 个落后最佳 LLM（−11.6 F1，但便宜 44×）；钓鱼邮件 62.6% vs Claude Haiku 81.3% |
| W7 | **工程约束** | 托管 API 约 430 ms 固定延迟下限；仅文本；32k 上下文；超过 255 个选项需两阶段 |

综述（awesome-jev-survey）的结论：**最稳定的用法是"廉价首轮 + 基于置信度的升级"**——这恰恰要求置信度可信、能说"不知道"、能在需要时多想。Jev 在这三点上都没有给出保证。

## 3. 由此得到的设计要求（Ilya 的出发点）

1. 开放世界：任何题型都必须能表达 NONE（越界 / 以上皆非），无需调用方记得加选项。→ W1
2. 置信度必须能变成**有统计保证的契约**（可认证的选择性风险、共形预测集）。→ W2
3. 输出必须对选项顺序**严格不变**，并且语义由 rubric 决定、名称只是标识符。→ W3
4. 问题之间要**隔离**（答案不随同批问题改变），而调用方声明的逻辑关系要被**精确**满足。→ W4
5. 计算量要**按需**：简单题一步出结论，难题自动多迭代，且每一步的信念都是校准的。→ W5

## 参考来源
- Jev (AI model) — Wikipedia: https://en.wikipedia.org/wiki/Jev_(AI_model)
- TypeSafe 文档: https://docs.typesafe.ai/introduction
- MarkTechPost 发布报道: https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/
- Vercel 采用速度: https://startupfortune.com/typesafe-ais-decision-model-jev-becomes-vercels-fastest-adopted-launch/
- 融资与团队: https://en.wowtale.net/2026/09/21/235190/ ， https://yourstory.com/ai-story/what-is-jev-typesafe-ai-decision-model
- RLCD 解读: https://www.mindstudio.ai/blog/typesafe-jev-rlcd-vs-rlhf
- PriorBench 预注册独立评测: https://github.com/priorbench/jev
- Awesome Jev 证据综述: https://github.com/Eurekaleo/awesome-jev-survey
- simple-jev（开源共享前缀 + 标签 logit 读出）: https://github.com/featherless-ai/simple-jev
- jev-visual: https://github.com/hr98w/jev-visual ；Vision-JEV: https://github.com/arnodjiang/Vision-JEV
- Visual Jev: https://arxiv.org/abs/2609.25845
- Same Scores, Different Decisions: https://arxiv.org/abs/2609.27678
- Can Jev Judge Radiology Reports?: https://arxiv.org/abs/2609.27607
- Type-Safe Is Not Error-Free: https://arxiv.org/abs/2609.26758
- 延迟 / 成本声明核查: https://dev.to/arifulislamat/typesafes-jev-model-is-it-really-193x-faster-and-444x-cheaper-56oa ， https://docs.litellm.ai/blog/jev-auto-router-benchmark
