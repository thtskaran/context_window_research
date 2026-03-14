#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
#  Mixed Filler Ratio Experiment — Full Pipeline
#
#  Tests ecological validity: what agreement:correction ratio
#  triggers the behavioral ratchet?
#
#  Per model: 115 probes × 7 conditions = 805 calls (~$1.85)
#  All 6 models: 4,830 experiment calls + 4,830 judge calls
#  Estimated cost: ~$12 experiment + ~$29 judge ≈ $41 total
# ═══════════════════════════════════════════════════════════════

KEYS_FILE="keys.txt"
WORKERS=25
JUDGE_MODEL="anthropic/claude-sonnet-4-6"
JUDGE_WORKERS=30

# All 6 models from the original experiment
MODELS=(
    "google/gemma-3n-e4b-it"
    "qwen/qwen-2.5-7b-instruct"
    "mistralai/mixtral-8x7b-instruct"
    "mistralai/mistral-small-24b-instruct-2501"
    "deepseek/deepseek-chat-v3.1"
    "qwen/qwen-2.5-72b-instruct"
)

MAX_TOKENS=(
    8192      # Gemma 3N
    32768     # Qwen 7B
    32768     # Mixtral 8x7B
    32768     # Mistral 24B
    65536     # DeepSeek V3.1
    32768     # Qwen 72B
)

echo "═══════════════════════════════════════════════════════════"
echo " Mixed Filler Ratio Experiment"
echo " 6 models × 115 probes × 7 conditions = 4,830 calls"
echo " + Sonnet 4.6 judge pass"
echo "═══════════════════════════════════════════════════════════"

mkdir -p results figures

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MAX_TOK="${MAX_TOKENS[$i]}"
    MODEL_SLUG="${MODEL//\//_}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶ ${MODEL} (ctx=${MAX_TOK})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    RESULT_FILE="results/${MODEL_SLUG}_mixed_results.jsonl"
    JUDGED_FILE="results/${MODEL_SLUG}_mixed_judged.jsonl"

    # Clean previous
    rm -f "$RESULT_FILE" "$JUDGED_FILE"

    # Step 1: Run experiment
    echo "  ▶ Running mixed filler experiment..."
    python run_mixed_filler.py \
        --mode api \
        --model "$MODEL" \
        --keys-file "$KEYS_FILE" \
        --max-tokens "$MAX_TOK" \
        --output-dir results \
        --workers "$WORKERS"

    echo "  ▶ Results: $(wc -l < "$RESULT_FILE") lines"

    # Step 2: Judge with Sonnet 4.6
    echo "  ▶ Running Sonnet 4.6 judge..."
    python llm_judge.py \
        --input "$RESULT_FILE" \
        --output "$JUDGED_FILE" \
        --judge-model "$JUDGE_MODEL" \
        --keys-file "$KEYS_FILE" \
        --workers "$JUDGE_WORKERS"

    echo "  ▶ Judged: $(wc -l < "$JUDGED_FILE") lines"
done

# Step 3: Analyze all models
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ▶ Analyzing results across all models..."
echo "═══════════════════════════════════════════════════════════"
python analyze_mixed_filler.py \
    --results-dir results/ \
    --output figures/mixed_filler_report.json \
    --verbose

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " DONE!"
echo " Results:  results/*_mixed_judged.jsonl"
echo " Report:   figures/mixed_filler_report.json"
echo "═══════════════════════════════════════════════════════════"
