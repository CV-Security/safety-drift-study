#!/bin/bash
# scripts/run_experiments.sh
# Runs all 6 experiments sequentially overnight

echo "🔬 Starting all 6 experiments..."
echo "================================"

cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

run_experiment() {
    MODEL=$1
    DATASET=$2
    NAME="${MODEL}_${DATASET}"
    echo ""
    echo "▶ Running: $NAME"
    python3 main.py --model $MODEL --dataset $DATASET \
        > logs/${NAME}.txt 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ $NAME complete"
    else
        echo "❌ $NAME FAILED — check logs/${NAME}.txt"
    fi
}

# All 6 experiments
run_experiment qwen0.5b  dialogsum
run_experiment qwen0.5b  xsum
run_experiment qwen1.5b  dialogsum
run_experiment qwen1.5b  xsum
run_experiment smollm1.7b dialogsum
run_experiment smollm1.7b xsum

echo ""
echo "================================"
echo "✅ All experiments complete!"
echo "Results saved in results/ folder"