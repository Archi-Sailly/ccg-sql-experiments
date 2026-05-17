# CLAUDE.md — Claude Code 작업 컨텍스트

> 이 문서는 Claude Code가 본 repo에서 작업할 때 항상 먼저 읽는 컨텍스트 파일입니다.
> 모든 모듈 작성 시 이 가이드를 따라야 합니다.

---

## 프로젝트 개요

**CCG-SQL** (Construct-Conditioned Generation Model) — 한양대 석사논문(윤정석, 2026) 실험 코드.

소형 언어 모델(Qwen3-8B)을 위한 Text-to-SQL 생성 모델로, 단일 함수 f_θ로 정의됩니다:

```
y* = f_θ(q, S) = argmax_k [ R_exec_pred(y_k) + λ·R_construct_pred(y_k, c_pred) ]
```

5개 component (Layer 1~5) 모두 단일 잠재변수 `target_construct (c_pred)`로 조건화됩니다.

---

## 핵심 학술적 주장 (코드 작성 시 항상 의식할 것)

1. **단일 모델 f_θ** — 파이프라인 조립이 아닌 통합 모델 정의. 따라서 코드도 단일 진입점(`f_theta(q, S)`)을 두는 게 좋음.
2. **Reward-Consistent Inference** — 학습 J(θ)의 R_total이 추론 score s_k에 동일 형태로 출현. 학습 reward 함수와 inference reranking 함수는 **반드시 같은 함수**를 호출해야 함.
3. **다층 conditioning** — c_pred가 Layer 3 (검색), Layer 4 (정책 컨텍스트), Layer 5 (재순위)에 모두 명시적 인자로 진입.

---

## 코드 작성 원칙

### Python 스타일
- **Python 3.10+** (Colab 표준)
- **Type hints 필수** — 모든 함수에 `def foo(x: int, y: str) -> bool:`
- **Docstring 필수** — Google style
- **black + ruff** 통과 — line length 100
- **테스트 필수** — `tests/test_<module>.py` 작성

### 모듈 구조
- 한 함수는 한 가지 일만 (Single Responsibility)
- 부작용 있는 함수와 순수 함수 분리
- I/O와 계산 로직 분리

### 의존성
- HuggingFace 생태계 우선 (transformers, peft, trl, datasets)
- 가능한 vLLM 사용 (rollout 가속)
- Sentence-Transformers · scikit-learn · sqlglot 사용

### 비-금기
- ❌ `from typing import *` — explicit import만
- ❌ 전역 변수 사용 — config는 YAML 파일에서 로드
- ❌ `print()` 디버그 — `logging` 모듈 또는 W&B 사용
- ❌ Magic number — 모든 상수는 config 또는 명명 상수

---

## Phase별 작업 분담

| Phase | 모듈 | 핵심 함수 |
|---|---|---|
| 1 Data | `src/data/` | `classify_construct(sql) -> str` |
| 2 Predictor | `src/predictor/` | `predict_construct(q) -> np.ndarray[5]` |
| 3 Retrieval | `src/retrieval/` | `retrieve(q, c_pred, K=8) -> list[dict]` |
| 4 Reward | `src/reward/` | `reward_exec(y, sql_g, db) -> float` <br> `reward_construct(y, c_gold) -> float` |
| 5 SFT | `src/sft/` | `train_sft(model, data, config) -> Model` |
| 6 GSPO | `src/gspo/` | `train_gspo(sft_model, data, reward_fns, config) -> Model` |
| 7 Reranker | `src/reranker/` | `sample_and_select(model, q, S, K=8) -> str` |
| 8 Eval | `src/eval/` | `eval_bird_dev(pipeline) -> dict` |
| 9 Experiments | `src/experiments/` | `run_ablation(config) -> dict` |
| 10 Viz | `src/viz/` | `generate_chart(results, name) -> Path` |

---

## 환경 변수 / 시크릿

다음 환경 변수가 설정되어야 함 (Colab의 경우 secrets에 저장):

- `HF_TOKEN` — HuggingFace 토큰 (Qwen3 다운로드)
- `WANDB_API_KEY` — W&B 토큰 (로깅)
- `WANDB_PROJECT=ccg-sql` — 프로젝트명 고정

---

## 학습 환경 가정

- **GPU**: A100 80GB (Colab Pro+, A100 40GB도 대응 가능)
- **메모리 한계 시**: LoRA r=16 → r=8, batch=2 → batch=1, gradient_accumulation 늘리기
- **세션 24h 제한**: 모든 학습은 resume 가능해야 함 (`--resume` 플래그)

---

## 체크리스트 — 모듈 작성 완료 기준

새 모듈을 작성할 때마다 다음을 모두 충족해야 마침:

- [ ] Type hints 완비
- [ ] Docstring (Google style)
- [ ] unit test (정상 케이스 + 에지 케이스)
- [ ] `make test` 통과
- [ ] `black src/ tests/` 적용
- [ ] `ruff check src/ tests/` 무경고
- [ ] 사용 예시가 함수 docstring 또는 notebook에 있음
- [ ] 로깅: 진행상황은 `logging.info`, 에러는 `logging.error`
- [ ] 결과는 JSON으로 `results/` 폴더에 저장

---

## 자주 쓰는 패턴

### YAML config 로드
```python
import yaml
from pathlib import Path

def load_config(path: Path = Path("configs/default.yaml")) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
```

### Resumable 학습 (체크포인트)
```python
from peft import PeftModel

if Path(ckpt_dir).exists() and args.resume:
    model = PeftModel.from_pretrained(base, ckpt_dir)
    logger.info(f"Resumed from {ckpt_dir}")
else:
    model = init_lora(base, config)
```

### W&B 로깅
```python
import wandb
wandb.init(project="ccg-sql", name=f"phase_{phase}_{run_name}", config=config)
wandb.log({"reward": r, "kl": kl, "step": step})
```

---

## 자주 발생하는 함정

1. **TRL GSPO 옵션 이름**: `importance_sampling="sequence"` (NOT `"sequence_level"`)
2. **Dr.GRPO 패치**: `dr_grpo_loss=True` 필수 (advantage 정규화 버그 수정)
3. **vLLM rollout**: GPU 1장만 있으면 학습 모델과 vLLM이 GPU를 공유해야 — `vllm.LLM(gpu_memory_utilization=0.4)`로 학습 모델에 60% 양보
4. **sqlglot 파싱 실패**: try-except 필수 — 깨진 SQL은 `c='plain'` 또는 `None` 반환
5. **SQLite 실행 timeout**: `signal.alarm()` 또는 `subprocess.run(timeout=30)` 필수 — 데드락 방지

---

## 발견 시 알려주세요

본 가이드에 누락된 컨텍스트나 패턴이 있으면 이 파일에 추가해주세요. Claude Code는 다음 세션에서 이 파일을 다시 읽습니다.
