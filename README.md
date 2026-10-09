## Experimental Results: LLM Tool Routing Benchmark

### 1. Experimental Setup

| Configuration | Value |
|---|---|
| Model | GPT-6 Luna |
| Reasoning Effort | None |
| Dataset | BFCL V3 |
| Validation Samples | 25 per scenario |
| Test Samples | 100 per scenario |
| Retrieval | BM25 + LLM Ranking |
| Top-K Candidates | 3, 5, 8, 12 |
| Concurrency | 4 |
| Metrics | Accuracy, F1, p95 Latency |
| API Success | 100% across all reported evaluations |

Five approaches were evaluated:

1. **Numeric Router:** Predict a numeric Tool ID.
2. **Joint Router:** Predict Tool ID and Search Query simultaneously.
3. **ReAct-style Router:** Reconsider routing through a recovery step.
4. **Parallel Multi-tool Selection:** Select multiple Tools.
5. **Sequential Tool Combination:** Predict ordered Tool calls.

Both single-turn and multi-turn scenarios were evaluated. Multi-turn evaluation includes conversation history but does not reproduce a fully stateful tool execution environment.

---

### 2. Validation: LLM Only vs Retrieval + Ranking

| Router | Strategy | K | Single Acc. | Multi Acc. | Average |
|---|---|---:|---:|---:|---:|
| **Numeric** | **LLM Only** | **0** | **0.960** | **0.760** | **0.860** |
| Joint | LLM Only | 0 | 0.960 | 0.680 | 0.820 |
| Numeric | Retrieval | 3 | 0.960 | 0.680 | 0.820 |
| Joint | Retrieval | 3 | 0.880 | 0.600 | 0.740 |
| Numeric | Retrieval | 5 | 0.960 | 0.680 | 0.820 |
| Joint | Retrieval | 5 | 0.960 | 0.640 | 0.800 |
| Numeric | Retrieval | 8 | 0.960 | 0.640 | 0.800 |
| Joint | Retrieval | 8 | 0.960 | 0.720 | 0.840 |
| Numeric | Retrieval | 12 | 0.960 | 0.520 | 0.740 |
| Joint | Retrieval | 12 | 0.960 | 0.680 | 0.820 |

**Best Validation Configuration: Numeric Router + LLM Only (Average Accuracy 86.0%)**

#### Key Findings

- Numeric Router + LLM Only achieved the highest average validation accuracy of **86.0%**.
- Joint Router + Retrieval K=8 was the strongest retrieval-based configuration, reaching **84.0%** average accuracy.
- Numeric Router exceeded Joint Router by 8 percentage points on multi-turn validation without retrieval (76% vs 68%).
- Single-turn accuracy remained at 96% for nearly all configurations, suggesting limited discrimination between strategies on this subset.
- Numeric Router retrieval accuracy decreased from 68% at K=3/5 to 52% at K=12 on multi-turn samples.
- Retrieval did not improve routing accuracy over LLM Only in this experiment.

**Interpretation:** For the evaluated candidate sets, additional BM25 candidate filtering did not provide a consistent routing benefit. Direct numeric selection was more reliable, particularly for multi-turn queries.

These results do not establish that retrieval is generally unnecessary. Larger Tool registries, different distractors, and semantic retrieval models may change the outcome.

---

### 3. Test: Numeric vs Joint Routing

| Method | Scenario | Accuracy | F1 | p95 Latency |
|---|---|---:|---:|---:|
| **Numeric / LLM Only** | Single-turn | **1.000** | **1.000** | 1.83s |
| **Numeric / LLM Only** | Multi-turn | **0.660** | **0.660** | 3.77s |
| Joint / Retrieval K=8 | Single-turn | 0.960 | 0.960 | **1.67s** |
| Joint / Retrieval K=8 | Multi-turn | 0.620 | 0.620 | **2.72s** |

#### Key Findings

**Numeric Router achieved higher Tool Selection Accuracy.**

- Single-turn: Numeric 100% vs Joint 96% (+4 percentage points).
- Multi-turn: Numeric 66% vs Joint 62% (+4 percentage points).
- Combined average: Numeric 83% vs Joint 79%.

However, the Joint Router with Retrieval K=8 showed lower p95 latency on both test scenarios. In multi-turn evaluation, p95 latency was 2.72s compared with 3.77s for Numeric LLM Only.

Because the tested methods use different candidate selection strategies, these differences reflect the complete routing pipelines rather than the output formats alone.

**Interpretation:** Numeric prediction provided the highest accuracy in the tested configurations, while Joint Routing offered a latency trade-off. Generating Tool IDs and Queries together did not demonstrate an accuracy advantage.

The actual retrieval quality of generated Search Queries was not measured.

---

### 4. ReAct-style Routing

| Scenario | Baseline Accuracy | ReAct Accuracy | Baseline p95 | ReAct p95 |
|---|---:|---:|---:|---:|
| Single-turn | 1.000 | 1.000 | 1.83s | 2.13s |
| Multi-turn | 0.660 | 0.660 | 3.77s | 3.41s |

#### Key Findings

- ReAct did not improve aggregate routing accuracy.
- Single-turn p95 latency increased by 0.30s.
- Multi-turn p95 latency decreased by 0.36s, despite unchanged accuracy.
- The current aggregate metrics do not reveal how frequently the recovery mechanism activated or whether it corrected individual routing errors.

**Interpretation:** The current ReAct-style recovery mechanism has not demonstrated a measurable accuracy benefit.

Improving failure detection, collecting explicit Tool execution feedback, and measuring successful corrections may be more valuable than adding unconditional reasoning steps.

---

### 5. Multi-tool and Combination Evaluation

| Experiment | Single-turn F1 | Multi-turn F1 | Single p95 | Multi p95 |
|---|---:|---:|---:|---:|
| Parallel Multi-tool | **0.995** | 0.567 | 1.77s | 1.87s |
| Sequential Combination | 0.738 | 0.556 | 1.65s | 2.03s |

#### Key Findings

**Parallel Multi-tool Selection**

- Single-turn F1 reached 99.5%.
- Multi-turn F1 dropped to 56.7%.
- The 42.8 percentage-point difference indicates a substantial performance gap when conversational context is introduced.

**Sequential Tool Combination**

- Single-turn F1 reached 73.8%.
- Multi-turn F1 decreased to 55.6%.
- Performance was lower than Parallel Multi-tool Selection in both scenarios.

**Interpretation:** Selecting multiple Tools for a single request was highly effective in this benchmark, while predicting a sequence of Tool calls was more difficult.

Multi-turn scenarios remain challenging for both approaches. Sequence prediction likely requires additional work on dependencies, state representation, and intermediate-result handling.

Parallel and Sequential F1 values correspond to different tasks, so the differences should not be interpreted as a direct ranking of general-purpose agent performance.

---

### 6. Overall Analysis

| Experiment | Single-turn | Multi-turn | Observation |
|---|---:|---:|---|
| Numeric Router | 100% Acc. | 66% Acc. | Highest tested routing accuracy |
| Joint Router | 96% Acc. | 62% Acc. | Lower p95 latency |
| ReAct Router | 100% Acc. | 66% Acc. | No aggregate accuracy improvement |
| Parallel Multi-tool | 99.5% F1 | 56.7% F1 | Strong single-turn performance |
| Sequential Combination | 73.8% F1 | 55.6% F1 | Tool sequencing remains challenging |

### 7. Conclusions

**1. Numeric-only Tool Selection is the strongest tested baseline.**

Direct Tool ID prediction achieved 100% single-turn accuracy and 66% multi-turn accuracy on the 100-sample test subsets.

**2. Retrieval + Ranking did not consistently improve accuracy.**

The strongest retrieval-based validation configuration achieved 84% average accuracy, compared with 86% for Numeric LLM Only. Candidate retrieval may still become useful as Tool catalog size increases.

**3. Multi-turn Routing is the primary accuracy bottleneck.**

Numeric routing accuracy dropped by 34 percentage points between single-turn and multi-turn evaluations. Similar degradation appeared in Joint Routing, Parallel Multi-tool Selection, and Sequential Combination.

**4. ReAct-style recovery requires better failure signals.**

The recovery mechanism produced no aggregate accuracy improvement. Future versions should distinguish routing failures, invalid arguments, missing information, and unsuccessful Tool execution.

**5. Tool dependencies introduce additional complexity.**

Sequential Combination achieved 73.8% F1 on single-turn and 55.6% F1 on multi-turn scenarios, suggesting room for improvement in planning and ordered Tool selection.

### 8. Limitations and Future Work

- **Sample size:** Validation contains only 25 samples per scenario. One additional correct prediction changes validation accuracy by 4 percentage points.
- **Tool catalog:** LLM Only uses a restricted candidate set with distractors, rather than a full production-scale Tool registry.
- **Retrieval:** BM25 retrieval was evaluated; Cross-Encoder or embedding-based retrieval was not compared.
- **Multi-turn:** Conversation history is included, but real Tool execution state is not fully simulated.
- **ReAct:** Aggregate results do not establish actual recovery success or correction frequency.
- **Query generation:** Generated Search Query relevance and retrieval effectiveness were not evaluated.
- **Latency:** p95 latency depends on network conditions, API load, and concurrency. Repeated trials are needed for stable comparisons.
- **Generalization:** The 100% single-turn Numeric result should be verified using larger, independently sampled and more challenging candidate sets, including checks for unintended answer leakage.

Recommended next experiments are multi-turn history selection, contrastive hard-negative Tool candidates, Cross-Encoder Tool re-ranking, and execution-feedback-driven Re-routing.

### Final Takeaway

**GPT-6 Luna demonstrated strong single-turn Tool Selection performance using numeric Tool IDs, reaching 100% accuracy on the evaluated test subset. Multi-turn routing and sequential Tool combinations remain the main areas for improvement.**

Under the evaluated conditions, Numeric LLM Only is the preferred accuracy-oriented baseline. Further experiments are required to determine whether this advantage persists across larger Tool catalogs and full stateful agent execution.
