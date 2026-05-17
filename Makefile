.PHONY: help install download-bird label-data train-predictor build-retriever \
        train-sft train-gspo eval-full ablation viz test clean

PYTHON := python
CONFIG := configs/default.yaml

help:
	@echo "CCG-SQL Experiments — Makefile commands"
	@echo ""
	@echo "  install            의존성 설치"
	@echo "  download-bird      BIRD 벤치마크 다운로드"
	@echo ""
	@echo "  Phase 1 — Data:"
	@echo "  label-data         sqlglot AST 5-class 라벨링"
	@echo ""
	@echo "  Phase 2 — Predictor:"
	@echo "  train-predictor    Sentence-BERT + LogReg 학습"
	@echo ""
	@echo "  Phase 3 — Retrieval:"
	@echo "  build-retriever    2-stage retrieval 인덱스 구축"
	@echo ""
	@echo "  Phase 5-6 — Training:"
	@echo "  train-sft          LoRA SFT baseline (10-15h)"
	@echo "  train-gspo         GSPO RL 학습 (4-6 days)"
	@echo ""
	@echo "  Phase 7-8 — Evaluation:"
	@echo "  eval-full          전체 파이프라인 평가 (BIRD Dev)"
	@echo ""
	@echo "  Phase 9 — Ablation:"
	@echo "  ablation           모든 ablation 자동 실행"
	@echo ""
	@echo "  Phase 10 — Output:"
	@echo "  viz                차트 + 표 생성"
	@echo ""
	@echo "  test               unit tests"
	@echo "  clean              임시 파일 정리"

install:
	pip install -r requirements.txt
	@echo "✓ 의존성 설치 완료"

download-bird:
	bash scripts/download_bird.sh
	@echo "✓ BIRD 다운로드 완료 — data/bird/ 확인"

# Phase 1
label-data:
	$(PYTHON) -m src.data.label_constructs --config $(CONFIG)

# Phase 2
train-predictor:
	$(PYTHON) -m src.predictor.train --config $(CONFIG)

# Phase 3
build-retriever:
	$(PYTHON) -m src.retrieval.build_index --config $(CONFIG)

# Phase 5
train-sft:
	$(PYTHON) -m src.sft.train --config $(CONFIG)

# Phase 6 — GSPO RL (resumable)
train-gspo:
	$(PYTHON) -m src.gspo.train --config $(CONFIG) --resume

# Phase 7-8
eval-full:
	$(PYTHON) -m src.eval.bird_eval --config $(CONFIG)

# Phase 9
ablation:
	$(PYTHON) -m src.experiments.ablation_runner --config $(CONFIG)

# Phase 10
viz:
	$(PYTHON) -m src.viz.generate_charts --config $(CONFIG)

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
