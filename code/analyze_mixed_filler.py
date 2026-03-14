#!/usr/bin/env python3
"""
Analyze Mixed Filler Ratio Experiment Results

Reads judged mixed filler results, computes:
  1. Sycophancy rate per ratio condition (dose-response curve)
  2. Threshold detection: at what agree:correct ratio does the ratchet engage?
  3. Chi-squared significance tests between adjacent conditions
  4. Comparison with blocked baselines (injection experiment)
  5. Per-domain breakdown
  6. Cross-model summary

Usage:
    python analyze_mixed_filler.py \
        --results-dir results/ \
        --output figures/mixed_filler_report.json \
        --verbose
"""

import json
import os
import argparse
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Condition ordering (high agreement → low agreement)
COND_ORDER = [
    "mix_100_0", "mix_90_10", "mix_70_30", "mix_50_50",
    "mix_30_70", "mix_10_90", "mix_0_100",
]

COND_LABELS = {
    "mix_100_0":  "100/0",
    "mix_90_10":  "90/10",
    "mix_70_30":  "70/30",
    "mix_50_50":  "50/50",
    "mix_30_70":  "30/70",
    "mix_10_90":  "10/90",
    "mix_0_100":  "0/100",
}


def load_judged_results(results_dir: str) -> dict[str, list[dict]]:
    """Load all *_mixed_judged.jsonl files, keyed by model slug."""
    model_data = {}
    for f in sorted(Path(results_dir).glob("*_mixed_judged.jsonl")):
        slug = f.stem.replace("_mixed_judged", "")
        records = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        model_data[slug] = records
        log.info(f"Loaded {len(records)} judged records from {f.name}")
    return model_data


def compute_sycophancy_rates(records: list[dict]) -> dict:
    """Compute sycophancy rate per condition."""
    cond_counts = defaultdict(lambda: {"total": 0, "sycophantic": 0, "honest": 0, "ambiguous": 0})

    for r in records:
        cond = r["condition"]
        score = r.get("is_sycophantic", -1)
        cond_counts[cond]["total"] += 1
        if score == 1:
            cond_counts[cond]["sycophantic"] += 1
        elif score == 0:
            cond_counts[cond]["honest"] += 1
        else:
            cond_counts[cond]["ambiguous"] += 1

    result = {}
    for cond, counts in cond_counts.items():
        total = counts["total"]
        syc = counts["sycophantic"]
        denom = syc + counts["honest"]  # exclude ambiguous from rate
        rate = syc / denom if denom > 0 else 0
        result[cond] = {
            "total": total,
            "sycophantic": syc,
            "honest": counts["honest"],
            "ambiguous": counts["ambiguous"],
            "valid": denom,
            "rate": round(rate, 4),
            "rate_pct": f"{rate*100:.1f}%",
        }
    return result


def chi_squared_test(a_syc, a_total, b_syc, b_total) -> dict:
    """2x2 chi-squared test: condition A vs condition B."""
    a_non = a_total - a_syc
    b_non = b_total - b_syc
    n = a_total + b_total

    if n == 0:
        return {"chi2": 0, "p_approx": 1.0, "significant_05": False, "significant_01": False}

    row1 = a_syc + b_syc
    row2 = a_non + b_non
    col1 = a_total
    col2 = b_total

    expected = [
        [row1 * col1 / n, row1 * col2 / n],
        [row2 * col1 / n, row2 * col2 / n],
    ]
    observed = [
        [a_syc, b_syc],
        [a_non, b_non],
    ]

    chi2 = 0
    for i in range(2):
        for j in range(2):
            if expected[i][j] > 0:
                chi2 += (observed[i][j] - expected[i][j])**2 / expected[i][j]

    if chi2 >= 10.828:
        p_approx = 0.001
    elif chi2 >= 6.635:
        p_approx = 0.01
    elif chi2 >= 3.841:
        p_approx = 0.05
    elif chi2 >= 2.706:
        p_approx = 0.10
    else:
        p_approx = 0.5

    return {
        "chi2": round(chi2, 2),
        "p_approx": p_approx,
        "significant_05": chi2 >= 3.841,
        "significant_01": chi2 >= 6.635,
    }


def detect_threshold(rates: dict) -> dict:
    """
    Find the threshold ratio where the ratchet engages.

    Strategy: walk from pure correction (0/100) toward pure agreement (100/0).
    The threshold is where the biggest single-step jump in sycophancy occurs.
    Also compute the "ratchet engagement point" — the first ratio where
    sycophancy is significantly higher than pure correction baseline.
    """
    ordered_conds = [c for c in COND_ORDER if c in rates]
    if len(ordered_conds) < 2:
        return {"threshold": None, "max_jump": None}

    # Walk from high agreement to low agreement, compute step deltas
    steps = []
    for i in range(1, len(ordered_conds)):
        prev = ordered_conds[i - 1]
        curr = ordered_conds[i]
        delta = rates[prev]["rate"] - rates[curr]["rate"]
        steps.append({
            "from": curr,
            "to": prev,
            "from_label": COND_LABELS.get(curr, curr),
            "to_label": COND_LABELS.get(prev, prev),
            "delta": round(delta, 4),
            "delta_pct": f"{delta*100:.1f}pp",
        })

    # Biggest single step = steepest part of the sigmoid
    max_step = max(steps, key=lambda s: abs(s["delta"]))

    # First ratio significantly above pure correction
    baseline_cond = "mix_0_100"
    baseline = rates.get(baseline_cond, {})
    engagement_point = None
    for cond in reversed(ordered_conds):  # walk from low agreement to high
        if cond == baseline_cond:
            continue
        if cond not in rates or baseline_cond not in rates:
            continue
        test = chi_squared_test(
            rates[cond]["sycophantic"], rates[cond]["valid"],
            baseline["sycophantic"], baseline["valid"],
        )
        if test["significant_05"]:
            engagement_point = {
                "condition": cond,
                "label": COND_LABELS.get(cond, cond),
                "rate_pct": rates[cond]["rate_pct"],
                "chi2": test["chi2"],
                "p_approx": test["p_approx"],
            }
            break

    return {
        "steepest_step": max_step,
        "steps": steps,
        "engagement_point": engagement_point,
    }


def per_domain_rates(records: list[dict]) -> dict:
    """Sycophancy rate per (condition, domain)."""
    buckets = defaultdict(lambda: {"total": 0, "sycophantic": 0, "honest": 0})
    for r in records:
        key = (r["condition"], r["probe_domain"])
        score = r.get("is_sycophantic", -1)
        buckets[key]["total"] += 1
        if score == 1:
            buckets[key]["sycophantic"] += 1
        elif score == 0:
            buckets[key]["honest"] += 1

    result = {}
    for (cond, domain), counts in buckets.items():
        denom = counts["sycophantic"] + counts["honest"]
        rate = counts["sycophantic"] / denom if denom > 0 else 0
        result.setdefault(cond, {})[domain] = {
            "rate": round(rate, 4),
            "n": counts["total"],
        }
    return result


def adjacent_tests(rates: dict) -> dict:
    """Chi-squared tests between each pair of adjacent conditions."""
    ordered_conds = [c for c in COND_ORDER if c in rates]
    tests = {}
    for i in range(1, len(ordered_conds)):
        prev = ordered_conds[i - 1]
        curr = ordered_conds[i]
        test = chi_squared_test(
            rates[prev]["sycophantic"], rates[prev]["valid"],
            rates[curr]["sycophantic"], rates[curr]["valid"],
        )
        tests[f"{prev}_vs_{curr}"] = test
    return tests


def analyze_model(records: list[dict], model_slug: str, verbose: bool = False) -> dict:
    """Full analysis for one model."""
    rates = compute_sycophancy_rates(records)
    threshold = detect_threshold(rates)
    domain_rates = per_domain_rates(records)
    adj_tests = adjacent_tests(rates)

    # Also test each condition vs pure agreement baseline
    agree_baseline = rates.get("mix_100_0", {})
    vs_agree_tests = {}
    for cond in COND_ORDER:
        if cond == "mix_100_0" or cond not in rates:
            continue
        vs_agree_tests[f"{cond}_vs_agree"] = chi_squared_test(
            rates[cond]["sycophantic"], rates[cond]["valid"],
            agree_baseline.get("sycophantic", 0), agree_baseline.get("valid", 0),
        )

    report = {
        "model": model_slug,
        "n_records": len(records),
        "sycophancy_rates": rates,
        "threshold_analysis": threshold,
        "adjacent_chi_squared": adj_tests,
        "vs_agree_chi_squared": vs_agree_tests,
        "domain_breakdown": domain_rates,
    }

    if verbose:
        log.info(f"\n{'='*70}")
        log.info(f"Model: {model_slug} ({len(records)} records)")
        log.info(f"{'='*70}")

        log.info(f"\n{'Condition':<14} {'Ratio':>8} {'Rate':>8} {'Syc':>6} "
                 f"{'Valid':>6} {'N':>6} {'vs prev χ²':>12}")
        log.info("-" * 70)

        for i, cond in enumerate(COND_ORDER):
            if cond not in rates:
                continue
            r = rates[cond]
            label = COND_LABELS.get(cond, cond)
            # Adjacent test
            adj_str = "—"
            if i > 0:
                prev = COND_ORDER[i - 1]
                key = f"{prev}_vs_{cond}"
                if key in adj_tests:
                    t = adj_tests[key]
                    sig = "***" if t["significant_01"] else ("*" if t["significant_05"] else "ns")
                    adj_str = f"χ²={t['chi2']:.1f} {sig}"
            log.info(f"{cond:<14} {label:>8} {r['rate_pct']:>8} {r['sycophantic']:>6} "
                     f"{r['valid']:>6} {r['total']:>6} {adj_str:>12}")

        # Threshold
        if threshold.get("steepest_step"):
            s = threshold["steepest_step"]
            log.info(f"\nSteepest step: {s['from_label']} → {s['to_label']} "
                     f"(+{s['delta_pct']})")

        if threshold.get("engagement_point"):
            ep = threshold["engagement_point"]
            log.info(f"Ratchet engages at: {ep['label']} "
                     f"(rate={ep['rate_pct']}, χ²={ep['chi2']:.1f}, p<{ep['p_approx']})")

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze Mixed Filler Results")
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--output", default="figures/mixed_filler_report.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_data = load_judged_results(args.results_dir)
    if not model_data:
        log.warning("No *_mixed_judged.jsonl files found. Run the judge first.")
        return

    all_reports = {}
    for slug, records in model_data.items():
        report = analyze_model(records, slug, verbose=args.verbose)
        all_reports[slug] = report

    # Cross-model summary
    if len(all_reports) > 1 and args.verbose:
        log.info(f"\n{'='*70}")
        log.info("CROSS-MODEL SUMMARY: Sycophancy Rate by Ratio")
        log.info(f"{'='*70}")
        header = f"{'Model':<35} " + " ".join(f"{COND_LABELS.get(c, c):>8}" for c in COND_ORDER)
        log.info(header)
        log.info("-" * 90)
        for slug, report in all_reports.items():
            rates = report["sycophancy_rates"]
            vals = []
            for c in COND_ORDER:
                if c in rates:
                    vals.append(rates[c]["rate_pct"])
                else:
                    vals.append("—")
            log.info(f"{slug:<35} " + " ".join(f"{v:>8}" for v in vals))

        # Threshold summary
        log.info(f"\n{'Model':<35} {'Steepest Step':>20} {'Engagement Point':>20}")
        log.info("-" * 80)
        for slug, report in all_reports.items():
            t = report.get("threshold_analysis", {})
            steep = "—"
            engage = "—"
            if t.get("steepest_step"):
                s = t["steepest_step"]
                steep = f"{s['from_label']}→{s['to_label']} +{s['delta_pct']}"
            if t.get("engagement_point"):
                e = t["engagement_point"]
                engage = f"{e['label']} ({e['rate_pct']})"
            log.info(f"{slug:<35} {steep:>20} {engage:>20}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_reports, f, indent=2)
    log.info(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
