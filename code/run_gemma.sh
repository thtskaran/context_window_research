#!/usr/bin/env bash
set -euo pipefail

MODEL="google/gemma-3n-e4b-it"
MODEL_SLUG="google_gemma-3n-e4b-it"
MAX_TOKENS=32768
WORKERS=80
JUDGE_MODEL="anthropic/claude-sonnet-4-6"
JUDGE_WORKERS=80
KEYS_FILE="keys.txt"
REPEATS=3

echo "═══════════════════════════════════════════════════════════"
echo " Sycophancy × Context — Google Gemma 3N E4B (32K)"
echo " 115 probes × 11 levels × 3 fillers × ${REPEATS} repeats = 11,385 calls"
echo " ${WORKERS} concurrent workers"
echo " Then Sonnet 4.6 judge (${JUDGE_WORKERS} workers)"
echo "═══════════════════════════════════════════════════════════"

rm -f results/${MODEL_SLUG}_results.jsonl results/${MODEL_SLUG}_judged.jsonl
mkdir -p results figures

# Step 1: Run experiment
echo ""
echo "▶ Running experiment..."
python run_experiment.py \
  --mode api \
  --model "$MODEL" \
  --keys-file "$KEYS_FILE" \
  --max-tokens "$MAX_TOKENS" \
  --levels 11 \
  --probes-path probes.json \
  --output-dir results \
  --workers "$WORKERS" \
  --repeats "$REPEATS"

echo ""
echo "▶ Results: $(wc -l < "results/${MODEL_SLUG}_results.jsonl") lines"

# Step 2: LLM judge
echo ""
echo "▶ Running Sonnet 4.6 judge..."
python llm_judge.py \
  --input "results/${MODEL_SLUG}_results.jsonl" \
  --output "results/${MODEL_SLUG}_judged.jsonl" \
  --judge-model "$JUDGE_MODEL" \
  --keys-file "$KEYS_FILE" \
  --workers "$JUDGE_WORKERS"

# Step 3: Figures + stats
echo ""
echo "▶ Generating figures..."
python phase_diagram.py --results-dir results/ --output-dir figures/

echo ""
echo "▶ Running statistical tests..."
python statistical_tests.py --results-dir results/ --output figures/stats_report.json --verbose

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " DONE!"
echo " Results:  results/${MODEL_SLUG}_judged.jsonl"
echo " Figures:  figures/"
echo " Stats:    figures/stats_report.json"
echo "═══════════════════════════════════════════════════════════"
