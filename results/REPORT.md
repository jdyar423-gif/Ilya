# Ilya vs Jev-paradigm — head-to-head report

All numbers are produced by `python -m ilya.evaluate`; every split is seeded and disjoint from the training stream. Probabilities are temperature-scaled on each model's own calibration split before any metric is computed.

## Budget

| model | params | train steps | temperature | ms / call (32 q) | decisions / s | mean loop iters |
|---|---|---|---|---|---|---|
| Ilya | 867,234 | 15000 | 0.88 | 39.0 | 821 | 3.4 |
| Jev-replica (closed world) | 813,216 | 15000 | 0.95 | 18.9 | 1690 | – |
| Jev-replica + explicit NOTA option | 813,216 | 15000 | 0.89 | 22.3 | 1438 | – |

## Learning curves (validation accuracy during training, same data stream)

| model | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 | 10000 | 11000 | 12000 | 13000 | 14000 | 15000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ilya | 45.7% | 73.4% | 77.1% | 87.1% | 87.1% | 89.8% | 87.7% | 88.3% | 90.2% | 88.9% | 88.5% | 88.5% | 88.1% | 88.9% | 89.5% |
| Jev-replica (closed world) | 50.0% | 51.0% | 63.7% | 66.0% | 64.3% | 66.4% | 66.4% | 65.6% | 65.2% | 68.4% | 67.8% | 81.6% | 83.2% | 83.4% | 84.6% |
| Jev-replica + explicit NOTA option | 48.6% | 60.9% | 66.8% | 64.3% | 71.7% | 70.9% | 81.0% | 81.2% | 84.0% | 85.0% | 83.4% | 83.4% | 82.8% | 82.4% | 84.0% |

## 1. In-distribution decisions

| model | acc | NLL | Brier | ECE | conf AUROC | acc (answerable) | NLL (unanswerable) |
|---|---|---|---|---|---|---|---|
| Ilya | 89.8% | 0.191 | 0.114 | 0.012 | 0.963 | 96.4% | 1.050 |
| Jev-replica (closed world) | 82.8% | 0.330 | 0.193 | 0.014 | 0.942 | 88.2% | 1.009 |
| Jev-replica + explicit NOTA option | 83.8% | 0.312 | 0.179 | 0.022 | 0.955 | 89.7% | 1.100 |

Accuracy by question family:

| model | attr | color | count | rel | shape | size |
|---|---|---|---|---|---|---|
| Ilya | 92.6% | 90.6% | 83.1% | 89.9% | 92.1% | 88.0% |
| Jev-replica (closed world) | 85.6% | 72.6% | 82.8% | 89.0% | 74.2% | 88.8% |
| Jev-replica + explicit NOTA option | 92.3% | 72.8% | 83.7% | 89.7% | 72.0% | 88.3% |

## 2. Open world: out-of-scope inputs (correct answer = NONE)

`weather` and `soup` never appear in any training data.

| model | absent NONE-rate | missing NONE-rate | recipe NONE-rate | weather NONE-rate | soup NONE-rate |
|---|---|---|---|---|---|
| Ilya | 100.0% | 84.7% | 100.0% | 89.7% | 46.8% |
| Jev-replica (closed world) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Jev-replica + explicit NOTA option | 100.0% | 88.4% | 100.0% | 93.5% | 94.2% |

| model | absent confident-wrong | missing confident-wrong | recipe confident-wrong | weather confident-wrong | soup confident-wrong |
|---|---|---|---|---|---|
| Ilya | 0.0% | 0.0% | 0.0% | 7.2% | 30.4% |
| Jev-replica (closed world) | 22.7% | 0.2% | 3.1% | 4.0% | 40.6% |
| Jev-replica + explicit NOTA option | 0.0% | 0.0% | 0.0% | 0.0% | 1.5% |

| model | absent OOD AUROC | missing OOD AUROC | recipe OOD AUROC | weather OOD AUROC | soup OOD AUROC |
|---|---|---|---|---|---|
| Ilya | 1.000 | 0.997 | 1.000 | 0.996 | 0.931 |
| Jev-replica (closed world) | 0.743 | 0.881 | 0.836 | 0.841 | 0.682 |
| Jev-replica + explicit NOTA option | 1.000 | 0.997 | 1.000 | 0.997 | 0.995 |

## 3. Presentation invariance

| model | option-shuffle flip rate | mean TV shift | max TV shift | acc honest names | acc misleading names | picked-by-name rate |
|---|---|---|---|---|---|---|
| Ilya | 0.0% | 0.0000 | 0.0000 | 89.9% | 89.9% | 4.4% |
| Jev-replica (closed world) | 0.7% | 0.0006 | 0.0323 | 89.1% | 4.3% | 88.9% |
| Jev-replica + explicit NOTA option | 6.1% | 0.0107 | 0.7229 | 88.5% | 3.4% | 88.2% |

## 4. Multi-hop depth (7–8 objects; training used ≤ 6 objects)

| model | overall | 1 hops | 2 hops | 3 hops | 4 hops | 5 hops | 6 hops | 7 hops | iters |
|---|---|---|---|---|---|---|---|---|---|
| Ilya [T=2] | 78.9% | 64.6% | 63.5% | 70.9% | 75.6% | 91.0% | 100.0% | 100.0% | 2.0 |
| Ilya [T=4] | 79.8% | 63.1% | 65.7% | 73.4% | 78.4% | 90.3% | 100.0% | 100.0% | 4.0 |
| Ilya [T=8] | 79.8% | 63.3% | 65.5% | 72.4% | 79.1% | 90.8% | 100.0% | 100.0% | 8.0 |
| Ilya [T=16] | 79.1% | 61.1% | 63.8% | 71.9% | 78.6% | 91.7% | 100.0% | 100.0% | 16.0 |
| Ilya [T=24] | 79.1% | 61.8% | 63.1% | 71.9% | 78.6% | 91.5% | 100.0% | 100.0% | 24.0 |
| Ilya [T=32] | 79.1% | 61.8% | 63.1% | 71.9% | 78.6% | 91.5% | 100.0% | 100.0% | 32.0 |
| Ilya [anytime] | 79.1% | 61.8% | 63.1% | 71.9% | 78.6% | 91.5% | 100.0% | 100.0% | 13.8 |
| Jev-replica (closed world) [fixed] | 80.0% | 64.2% | 66.9% | 73.0% | 81.2% | 89.8% | 97.7% | 100.0% | – |
| Jev-replica + explicit NOTA option [fixed] | 79.4% | 64.6% | 68.3% | 68.3% | 76.8% | 91.2% | 100.0% | 100.0% | – |

## 5. Anytime beliefs (Ilya, in-distribution, no temperature)

| iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| acc | 51.7% | 89.4% | 89.7% | 89.8% | 89.7% | 89.7% | 89.8% | 89.8% |
| NLL | 0.992 | 0.192 | 0.191 | 0.191 | 0.191 | 0.190 | 0.191 | 0.191 |
| ECE | 0.046 | 0.013 | 0.011 | 0.009 | 0.008 | 0.008 | 0.007 | 0.008 |

## 6. Coherence under declared constraints

| model | bundle | violations (isolated) | violations (conditioned) | acc (isolated) | acc (conditioned) | NLL (isolated) | NLL (conditioned) | mean support |
|---|---|---|---|---|---|---|---|---|
| Ilya | color | 14.2% | 0.0% | 90.2% | 94.6% | 0.123 | 0.100 | 0.871 |
| Ilya | count | 52.0% | 0.0% | 88.3% | 93.3% | 0.159 | 0.132 | 0.688 |
| Jev-replica (closed world) | color | 64.6% | 0.0% | 78.2% | 94.8% | 0.446 | 0.119 | 0.298 |
| Jev-replica (closed world) | count | 92.8% | 0.0% | 45.3% | 65.1% | 0.983 | 0.590 | 0.099 |
| Jev-replica + explicit NOTA option | color | 14.0% | 0.0% | 90.8% | 94.2% | 0.133 | 0.106 | 0.866 |
| Jev-replica + explicit NOTA option | count | 42.8% | 0.0% | 89.0% | 92.9% | 0.183 | 0.152 | 0.653 |

## 7. Certified decisions on a deployment mix (70% ID + 30% out-of-scope)

Learn-then-Test gate with ε = 0.05, δ = 0.1; conformal sets with α = 0.05; 200 random calibration/test splits.

| model | raw acc | certified coverage | test selective risk | risk > ε freq | no-certificate freq | conformal coverage | mean set size |
|---|---|---|---|---|---|---|---|
| Ilya | 88.2% | 85.1% | 3.5% | 0.0% | 0.0% | 95.1% | 1.22 |
| Jev-replica (closed world) | 57.6% | 0.0% | – | 0.0% | 100.0% | 70.0% | 3.41 |
| Jev-replica + explicit NOTA option | 86.2% | 82.7% | 3.0% | 0.0% | 0.0% | 95.0% | 1.27 |

