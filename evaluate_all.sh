#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./evaluate_all.sh <path_to_model.pt>"
    exit 1
fi

MODEL="$1"
LAYOUTS=("classic" "empty" "spiral" "spiral_harder")

echo "=========================================="
echo "Evaluating Model: $MODEL"
echo "=========================================="

for layout in "${LAYOUTS[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "Testing on Layout: $layout"
    echo "------------------------------------------"
    python play_cv.py --layout "$layout" --model "$MODEL" --episodes 100 --headless
done

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
