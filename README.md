# GPT-6 Luna Tool Router Benchmark — BFCL V3 자동 다운로드

BFCL V3 데이터셋을 Hugging Face에서 필요한 파일만 자동 다운로드하고, OpenAI Python SDK의 `chat.completions.create()`로 `gpt-6-luna`를 호출합니다. **EM(Exact Match) 지표는 없습니다.**

## 1. 시작

```bash
unzip tool_router_bench_luna_auto.zip -d tool_router_bench_luna_auto
cd tool_router_bench_luna_auto
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='sk-...'
python test_smoke.py
```

## 2. 전체 실험 (자동 데이터 다운로드 + API 평가)

```bash
python bench.py \
  --model gpt-6-luna \
  --effort none \
  --data-dir ./bfcl_data \
  --val 25 --test 100 \
  --topks 3,5,8,12 \
  --distractors 24 \
  --concurrency 4 \
  --out results_luna
```

처음 실행하면 HF의 [`gorilla-llm/Berkeley-Function-Calling-Leaderboard`](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)에서 **22개 원본 파일**을 다운로드합니다. 데이터는 `bfcl_data/`에 캐시되며 다음 실행부터는 재사용합니다. 네트워크 정책상 HF 직접 접속이 안 된다면 필요한 22개 파일을 수동으로 배치한 후 `--no-download`로 실행할 수 있습니다.

Hugging Face rate limit에 걸리면 `HF_TOKEN` 환경 변수를 사용할 수 있습니다. 원본 파일을 다운로드하는 데는 `OPENAI_API_KEY`가 필요하지 않습니다.

### 데이터 다운로드만 실행

```bash
python bench.py --download-only --data-dir ./bfcl_data
python bench.py --check-data --data-dir ./bfcl_data
python bench.py --list-urls
```

### 저비용 샘플 테스트

```bash
python bench.py --val 5 --test 5 --topks 3,5 \
  --concurrency 2 --out results_luna_smoke
```

## 3. 평가 순서

| 단계 | 비교 | 지표 |
|---|---|---|
| 1 | 숫자 Tool ID만 생성, Single/Multi-turn | Accuracy, Precision, Recall, F1 |
| 2 | Tool ID + Query joint 생성, Single/Multi-turn | Accuracy, F1, Query 생성률 |
| 1차 튜닝 | LLM-only vs BM25 Top-K + LLM rank, K=3/5/8/12 | Validation Accuracy, Candidate Recall, Latency |
| 3 | 선택된 전략으로 ReAct-style Conditional Re-routing | Accuracy, Recovery/Regression Rate |
| 4 | Multi-tool 집합 선택 | Mean Instance Precision/Recall/F1 |
| 5 | Combination(여러 Tool과 순서·중복) 선택 | Precision/Recall/F1, Ordered LCS Recall |

단일턴·멀티턴 각 테스트 그룹에서 100개를 추출하며 Validation은 25개씩 별도로 추출합니다. 정답 후보가 충분하지 않으면 데이터 수를 임의로 채우지 않고 오류로 알려줍니다. `--test`와 `--val`을 줄여 재실행할 수 있습니다. 샘플은 대화 ID 단위로 Validation과 Test를 나눕니다.

**방식 선택**: 1~2의 각 후보(LLM-only + BM25 Top-K)를 Validation 단일/멀티턴 Accuracy의 평균으로 비교하고, 동점일 때 평균 Latency를 비교합니다. Test 단계에서는 별도로 튜닝하지 않습니다. 3~5번은 Validation 승자만 실행합니다.

**지표**:
- Routing Accuracy: 하나의 정답 Tool 이름과 예측 이름이 같은 비율 (1/2/3 단계).
- Mean Instance Precision/Recall/F1: 정답 및 예측 Tool 이름의 집합을 비교해 인스턴스별 계산 후 평균. Multi-tool은 순서를 반영하지 않습니다.
- Candidate Recall: 후보 목록에 포함된 정답 Tool 비율.
- Ordered LCS Recall: Combination 호출 순서를 고려한 최장 공통 부분 수열 길이 / 정답 시퀀스 길이.
- Latency: 전체 평균, 중간값, p95. API token 사용량과 LLM 호출 수도 기록합니다.

## 4. 출력

- `summary.md`: 단계별 지표 표
- `summary.csv`, `summary.json`: 전체 측정치, 승자, 데이터 소스
- `predictions.jsonl`: 샘플별 정답/예측, Latency, token, error, 생성 쿼리, 캐시 키(재실행 시 API 호출 재사용)

기존 결과와 섞이지 않도록 새로운 `--out` 폴더를 사용하세요.

## 5. 주요 옵션

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--model` | `gpt-6-luna` | OpenAI 모델 |
| `--effort` | `none` | `reasoning_effort` (none/low/medium/high/xhigh/max) |
| `--data-dir` | `./bfcl_data` | 데이터 캐시 디렉터리 |
| `--revision` | `main` | HF 브랜치/커밋. 재현성을 위해 SHA 권장 |
| `--download-only` | off | 다운로드만 수행 |
| `--no-download` | off | 다운로드 중단, 로컬 파일만 사용 |
| `--force-download` | off | 데이터 다시 받기 |
| `--hf-workers` | 6 | 데이터 다운로드 동시 연결 수 |
| `--concurrency` | 4 | OpenAI 동시 호출 수 |
| `--timeout` | 60s | API 요청 타임아웃 |
| `--api-retries` | 2 | SDK 재시도 횟수 |
| `--temperature` | 미지정 | 필요한 경우 `--effort none`과만 사용 |

## 6. 데이터 해석상 한계

- BFCL 공식 평가 스코어가 아닙니다. **Tool 이름 선택만** 확인하며 arguments 및 실제 Tool 실행 결과는 점수에 포함되지 않습니다.
- Joint Query는 BFCL에 별도 Query 정답이 없어 비어 있지 않은지 여부만 측정합니다. Query의 검색 성능을 측정하지 않습니다.
- ReAct-style 단계는 실제 Tool을 실행하지 않습니다. BM25 메타데이터 관련성을 관찰 신호로 사용해 조건부 재선택하는 실험입니다.
- Multi-turn은 이전 **사용자 발화만** 전송합니다. Tool 결과, 상태 변경, `initial_config`를 시뮬레이션하지 않습니다.
- `--distractors`로 무관한 Tool 후보를 추가해 평가 난도를 높입니다. BFCL 원본 후보 목록이 아니라 확장된 후보 목록에서 측정됩니다.
- 입력 데이터는 공개 BFCL이지만, 평가 프롬프트는 OpenAI API로 전송됩니다. 사내 비공개 데이터로 확장할 때 별도 승인/보안 정책을 확인하세요.
- 결과를 재현하려면 BFCL의 커밋 SHA를 `--revision`에 고정하고 요청 concurrency/effort를 동일하게 유지하세요.
