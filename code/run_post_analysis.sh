#!/usr/bin/env bash
set -euo pipefail

# ─── Post-Experiment Analysis Pipeline ───────────────────────────
# Run this AFTER run_experiment.py finishes.
# Uses Claude Sonnet 4.6 as judge on ALL results (not just ambiguous).
# Keyword scoring misses too many soft corrections — Sonnet catches them.
# ─────────────────────────────────────────────────────────────────

RESULTS_DIR="results"
FIGURES_DIR="figures"
JUDGE_MODEL="anthropic/claude-sonnet-4-6"
KEYS_FILE="keys.txt"
PROBES_PATH="probes.json"

mkdir -p "$FIGURES_DIR"

echo "═══════════════════════════════════════════════════════════"
echo " Step 1: Sonnet 4.6 judge — scoring ALL results"
echo "   Domain-aware judging: factual + opinion rubrics"
echo "═══════════════════════════════════════════════════════════"

# Find all result files and judge each one
for result_file in "$RESULTS_DIR"/*_results.jsonl; do
    [ -f "$result_file" ] || continue
    model_slug=$(basename "$result_file" _results.jsonl)
    judged_file="${RESULTS_DIR}/${model_slug}_judged.jsonl"

    echo ""
    echo "Judging: $model_slug"
    echo "  Input:  $result_file ($(wc -l < "$result_file") results)"
    echo "  Output: $judged_file"
    echo "  Judge:  $JUDGE_MODEL"

    python llm_judge.py \
        --input "$result_file" \
        --output "$judged_file" \
        --judge-model "$JUDGE_MODEL" \
        --keys-file "$KEYS_FILE" \
        --probes-path "$PROBES_PATH" \
        --workers 35

    echo "  Done: $(wc -l < "$judged_file") judged results"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Step 2: Sonnet 4.6 taxonomy judge — classifying failure modes"
echo "   Classifies each sycophantic response: direct/elaborate/qualified"
echo "═══════════════════════════════════════════════════════════"

for judged_file in "$RESULTS_DIR"/*_judged.jsonl; do
    [ -f "$judged_file" ] || continue
    model_slug=$(basename "$judged_file" _judged.jsonl)

    echo ""
    echo "Taxonomy: $model_slug"
    echo "  File:   $judged_file"
    echo "  Judge:  $JUDGE_MODEL"

    python taxonomy_judge.py \
        --input "$judged_file" \
        --output "$judged_file" \
        --judge-model "$JUDGE_MODEL" \
        --keys-file "$KEYS_FILE" \
        --probes-path "$PROBES_PATH" \
        --workers 35

    echo "  Done"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Step 3: Phase diagrams + figures"
echo "═══════════════════════════════════════════════════════════"

python phase_diagram.py \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$FIGURES_DIR"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Step 4: Statistical tests"
echo "═══════════════════════════════════════════════════════════"

python statistical_tests.py \
    --results-dir "$RESULTS_DIR" \
    --output "${FIGURES_DIR}/stats_report.json" \
    --verbose

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Step 5: Secondary analysis (taxonomy plots, latency, length)"
echo "═══════════════════════════════════════════════════════════"

python secondary_analysis.py

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " DONE!"
echo ""
echo " Results:  $RESULTS_DIR/"
echo " Figures:  $FIGURES_DIR/"
echo " Stats:    $FIGURES_DIR/stats_report.json"
echo ""
echo " Figures generated:"
ls -1 "$FIGURES_DIR"/*.png "$FIGURES_DIR"/*.pdf 2>/dev/null || echo "  (none)"
echo "═══════════════════════════════════════════════════════════"
