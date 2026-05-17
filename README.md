# CCG-SQL Experiments

Implementation of **CCG-SQL** (Construct-Conditioned Generation Model) — a single-function f_θ Text-to-SQL model with Construct-Aware Retrieval and Reward-Consistent Inference.

> 윤정석 · 한양대학교 공학대학원 컴퓨터공학과 119기 석사논문 실험 코드

---

## Quick Start

### 환경 설정 (Colab Pro+ 권장)

```bash
# 1. Repository clone
git clone https://github.com/<your-id>/ccg-sql-experiments.git
cd ccg-sql-experiments

# 2. 의존성 설치
pip install -r requirements.txt

# 3. HuggingFace + W&B 로그인
huggingface-cli login   # Qwen3-8B 다운로드용
wandb login              # 학습 메트릭 로깅용

# 4. BIRD 벤치마크 다운로드
make download-bird
```

### 학습·평가 흐름

```bash
make label-data       # Phase 1: sqlglot AST → 5-class labels
make train-predictor  # Phase 2: Sentence-BERT + LogReg
make build-retriever  # Phase 3: Construct-Aware Retrieval index
make train-sft        # Phase 5: LoRA SFT baseline
make train-gspo       # Phase 6: GSPO RL training (4-6 days)
make eval-full        # Phase 7-8: Reranker + BIRD Dev EX
make ablation         # Phase 9: All ablation experiments
```

---

## 모델 정의 — 한 줄 요약

```
y* = f_θ(q, S) = argmax_k [ R_exec_pred(y_k) + λ·R_construct_pred(y_k, c_pred) ]
```

학습 가능 파라미터:
- `φ_BERT` — Sentence-BERT (Construct Predictor encoder)
- `φ_LR` — Logistic Regression (Construct Predictor classifier)
- `φ_LoRA` — Qwen3-8B LoRA adapter (Policy)
- `α` — Retrieval cm weight
- `λ` — Reward balance

학습 목적:
```
J(θ) = E_(x,Y)[ Σ_k clip(r_seq,k, 1±ε) · A_k ] − β · KL(π_θ ‖ π_ref)
```

---

## Directory Structure

```
ccg-sql-experiments/
├── src/
│   ├── data/         # Phase 1 — BIRD 로딩, sqlglot AST 5-class 라벨링
│   ├── predictor/    # Phase 2 — Sentence-BERT + LogReg (Construct Predictor)
│   ├── retrieval/    # Phase 3 — 2-stage CA-Retrieval
│   ├── reward/       # Phase 4 — R_exec, R_construct, R_total
│   ├── sft/          # Phase 5 — LoRA SFT baseline
│   ├── gspo/         # Phase 6 — GSPO RL training
│   ├── reranker/     # Phase 7 — K=8 sampling + R_total scoring
│   ├── eval/         # Phase 8 — BIRD Dev EX 평가 자동화
│   ├── experiments/  # Phase 9 — Ablation, hyperparameter grid
│   └── viz/          # Phase 10 — 차트·표 생성
├── tests/            # unit tests
├── notebooks/        # Colab notebook entries
├── configs/          # YAML configs (hyperparameters)
├── scripts/          # CLI utilities
├── data/             # BIRD dataset (gitignore)
├── checkpoints/      # LoRA adapter checkpoints (gitignore)
├── results/          # JSON 결과 파일들
└── logs/             # W&B 백업
```

---

## Target Metrics (검증할 수치)

| 단계 | BIRD Dev EX | 비고 |
|---|---|---|
| Base (Qwen3-8B) | 52.3% | baseline |
| + SFT | 64.7% | LoRA SFT |
| + GSPO | 71.2% | RL training |
| + CCG-SQL 전체 | **76.5%** | Reranker 포함 ★ |
| Oracle Predictor | 78.2% | 시스템 상한 |

| Construct | 향상폭 (Base 대비) |
|---|---|
| plain | +14.1pp |
| HAVING | +22.9pp |
| EXISTS | +23.5pp |
| Window | +27.7pp |
| CTE | +29.3pp |

| 안정성 | GSPO | GRPO |
|---|---|---|
| KL divergence | **0.04** | 0.18 (4.5배 안정) |

**허용 오차**: ±2pp 학술적 변동 범위. 가상값과 ±3pp 이상 차이 시 논문 본문 수치 갱신.

---

## Citation

```bibtex
@mastersthesis{yoon2026ccgsql,
  title={구문 조건화 생성 모델: 시퀀스 단위 정책 학습과 구문 인식 검색 기반 소형 언어 모델 Text-to-SQL},
  author={윤정석},
  school={한양대학교 공학대학원 컴퓨터공학과},
  year={2026}
}
```

## License

MIT
