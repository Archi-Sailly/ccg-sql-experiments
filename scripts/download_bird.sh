#!/usr/bin/env bash
# BIRD 벤치마크 다운로드 스크립트
# 참고: https://bird-bench.github.io

set -e

DATA_DIR="${DATA_DIR:-./data/bird}"
mkdir -p "$DATA_DIR"

echo "==> BIRD train + dev 다운로드"

# Train set (9,428 questions + databases)
if [ ! -f "$DATA_DIR/train.zip" ]; then
    echo "  -> train.zip 다운로드 중..."
    wget -O "$DATA_DIR/train.zip" \
        https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
fi

# Dev set (1,534 questions)
if [ ! -f "$DATA_DIR/dev.zip" ]; then
    echo "  -> dev.zip 다운로드 중..."
    wget -O "$DATA_DIR/dev.zip" \
        https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
fi

# Unzip
echo "==> 압축 해제"
cd "$DATA_DIR"
[ ! -d "train" ] && unzip -q train.zip
[ ! -d "dev" ] && unzip -q dev.zip
cd -

# 검증
TRAIN_JSON=$(find "$DATA_DIR/train" -name "train.json" | head -1)
DEV_JSON=$(find "$DATA_DIR/dev" -name "dev.json" | head -1)

if [ -f "$TRAIN_JSON" ]; then
    TRAIN_N=$(python -c "import json; print(len(json.load(open('$TRAIN_JSON'))))")
    echo "  ✓ train: $TRAIN_N 질의 (기대값 9,428)"
fi
if [ -f "$DEV_JSON" ]; then
    DEV_N=$(python -c "import json; print(len(json.load(open('$DEV_JSON'))))")
    echo "  ✓ dev: $DEV_N 질의 (기대값 1,534)"
fi

echo ""
echo "==> 완료. 데이터 위치: $DATA_DIR"
echo "    다음 단계: make label-data"
