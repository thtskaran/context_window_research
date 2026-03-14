#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
#  Fine-Grained 0–10% Context Fill — Full Pipeline
#
#  Zooms into Qwen 7B's step function between 0% and 10% context
#  fill to determine phase transition vs sampling noise.
#
#  Single model: Qwen 2.5 7B Instruct
#  115 probes × 11 levels × 3 fillers = 3,795 calls
#  + Sonnet 4.6 judge pass (3,795 calls)
#  Estimated cost: ~$2 experiment + ~$8 judge ≈ $10 total
# ═══════════════════════════════════════════════════════════════

KEYS_FILE="keys.txt"
WORKERS=25
JUDGE_MODEL="anthropic/claude-sonnet-4-6"
JUDGE_WORKERS=30

MODEL="qwen/qwen-2.5-7b-instruct"
MAX_TOK=32768
MODEL_SLUG="${MODEL//\//_}"

echo "═══════════════════════════════════════════════════════════"
echo " Fine-Grained 0–10% Experiment (Qwen 7B)"
echo " 115 probes × 11 levels × 3 fillers = 3,795 calls"
echo " + Sonnet 4.6 judge pass"
echo "═══════════════════════════════════════════════════════════"

mkdir -p results figures

RESULT_FILE="results/${MODEL_SLUG}_finegrained_results.jsonl"
JUDGED_FILE="results/${MODEL_SLUG}_finegrained_judged.jsonl"

# Clean previous
rm -f "$RESULT_FILE" "$JUDGED_FILE"

# Step 1: Run experiment
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ▶ Running fine-grained experiment..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_finegrained.py \
    --model "$MODEL" \
    --keys-file "$KEYS_FILE" \
    --max-tokens "$MAX_TOK" \
    --output-dir results \
    --workers "$WORKERS"

echo "  ▶ Results: $(wc -l < "$RESULT_FILE") lines"

# Step 2: Judge with Sonnet 4.6
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ▶ Running Sonnet 4.6 judge..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python llm_judge.py \
    --input "$RESULT_FILE" \
    --output "$JUDGED_FILE" \
    --judge-model "$JUDGE_MODEL" \
    --keys-file "$KEYS_FILE" \
    --workers "$JUDGE_WORKERS"

echo "  ▶ Judged: $(wc -l < "$JUDGED_FILE") lines"

# Step 3: Analyze
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ▶ Analyzing results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python analyze_finegrained.py \
    --results-dir results/ \
    --output figures/finegrained_report.json \
    --verbose

# Step 4: Generate diagrams
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ▶ Generating diagrams..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python generate_finegrained_diagrams.py

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " DONE!"
echo " Results:   results/${MODEL_SLUG}_finegrained_judged.jsonl"
echo " Report:    figures/finegrained_report.json"
echo " Diagrams:  figures/finegrained_*.png"
echo "═══════════════════════════════════════════════════════════"
