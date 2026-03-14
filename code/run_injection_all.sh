#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
#  Correction Injection Mitigation — Full Pipeline
#
#  Tests whether injecting correction exchanges after agreement
#  filler can reset the behavioral ratchet.
#
#  Per model: 115 probes × 6 conditions = 690 calls (~$1.60)
#  All 6 models: 4,140 experiment calls + 4,140 judge calls
#  Estimated cost: ~$10 experiment + ~$25 judge ≈ $35 total
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
echo " Correction Injection Mitigation Experiment"
echo " 6 models × 115 probes × 6 conditions = 4,140 calls"
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

    RESULT_FILE="results/${MODEL_SLUG}_injection_results.jsonl"
    JUDGED_FILE="results/${MODEL_SLUG}_injection_judged.jsonl"

    # Clean previous
    rm -f "$RESULT_FILE" "$JUDGED_FILE"

    # Step 1: Run experiment
    echo "  ▶ Running injection experiment..."
    python run_correction_injection.py \
        --mode api \
        --model "$MODEL" \
        --keys-file "$KEYS_FILE" \
        --max-tokens "$MAX_TOK" \
        --output-dir results \
        --workers "$WORKERS"

    echo "  ▶ Results: $(wc -l < "$RESULT_FILE") lines"

    # Step 2: Judge with Sonnet 4.6
    # The llm_judge.py works on any JSONL with probe_id, probe_domain, response fields.
    # It reads probes.json for claim/truth, so it's compatible with injection results.
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
python analyze_injection.py \
    --results-dir results/ \
    --output figures/injection_report.json \
    --verbose

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " DONE!"
echo " Results:  results/*_injection_judged.jsonl"
echo " Report:   figures/injection_report.json"
echo "═══════════════════════════════════════════════════════════"
