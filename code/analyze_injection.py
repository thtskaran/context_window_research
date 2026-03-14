#!/usr/bin/env python3
"""
Analyze Correction Injection Experiment Results

Reads judged injection results, computes:
  1. Sycophancy rate per condition (dose-response curve)
  2. Reset fraction: how far each injection dose moves from agree_only toward correct_only
  3. Chi-squared significance tests (each injection vs agree_only)
  4. Taxonomy breakdown per condition (if taxonomy judge data available)
  5. Per-model comparison table

Usage:
    python analyze_injection.py \
        --results-dir results/ \
        --output figures/injection_report.json \
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


def load_judged_results(results_dir: str) -> dict[str, list[dict]]:
    """Load all *_injection_judged.jsonl files, keyed by model slug."""
    model_data = {}
    for f in sorted(Path(results_dir).glob("*_injection_judged.jsonl")):
        slug = f.stem.replace("_injection_judged", "")
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
        # llm_judge.py sets is_sycophantic: 1=sycophantic, 0=honest, -1=ambiguous
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
        rate = syc / total if total > 0 else 0
        result[cond] = {
            "total": total,
            "sycophantic": syc,
            "honest": counts["honest"],
            "ambiguous": counts["ambiguous"],
            "rate": round(rate, 4),
            "rate_pct": f"{rate*100:.1f}%",
        }
    return result


def compute_reset_fraction(rates: dict) -> dict:
    """
    Reset fraction = (agree_rate - inject_rate) / (agree_rate - correct_rate)

    1.0 = full reset (injection drops sycophancy to correction baseline)
    0.0 = no reset (injection has no effect)
    >1.0 = overcorrection
    <0.0 = injection makes it worse
    """
    agree_rate = rates.get("agree_only", {}).get("rate", 0)
    correct_rate = rates.get("correct_only", {}).get("rate", 0)
    gap = agree_rate - correct_rate

    fractions = {}
    for cond, data in rates.items():
        if cond in ("agree_only", "correct_only"):
            continue
        if gap > 0.001:  # avoid division by zero
            frac = (agree_rate - data["rate"]) / gap
            fractions[cond] = round(frac, 3)
        else:
            fractions[cond] = None  # gap too small to measure

    return fractions


def chi_squared_test(a_syc, a_total, b_syc, b_total) -> dict:
    """2x2 chi-squared test: condition A vs condition B."""
    a_non = a_total - a_syc
    b_non = b_total - b_syc
    n = a_total + b_total

    if n == 0:
        return {"chi2": 0, "p": 1.0, "significant": False}

    # Expected values
    row1 = a_syc + b_syc       # total sycophantic
    row2 = a_non + b_non       # total non-sycophantic
    col1 = a_total             # condition A total
    col2 = b_total             # condition B total

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

    # p-value from chi-squared with 1 df (use survival function approximation)
    # For simplicity, use a lookup table for common thresholds
    if chi2 >= 10.828:
        p_approx = 0.001
    elif chi2 >= 6.635:
        p_approx = 0.01
    elif chi2 >= 3.841:
        p_approx = 0.05
    elif chi2 >= 2.706:
        p_approx = 0.10
    else:
        p_approx = 0.5  # not significant

    return {
        "chi2": round(chi2, 2),
        "p_approx": p_approx,
        "significant_05": chi2 >= 3.841,
        "significant_01": chi2 >= 6.635,
    }


def per_domain_rates(records: list[dict]) -> dict:
    """Sycophancy rate per (condition, domain)."""
    buckets = defaultdict(lambda: {"total": 0, "sycophantic": 0})
    for r in records:
        key = (r["condition"], r["probe_domain"])
        score = r.get("is_sycophantic", -1)
        buckets[key]["total"] += 1
        if score == 1:
            buckets[key]["sycophantic"] += 1

    result = {}
    for (cond, domain), counts in buckets.items():
        rate = counts["sycophantic"] / counts["total"] if counts["total"] > 0 else 0
        result.setdefault(cond, {})[domain] = {
            "rate": round(rate, 4),
            "n": counts["total"],
        }
    return result


def analyze_model(records: list[dict], model_slug: str, verbose: bool = False) -> dict:
    """Full analysis for one model."""
    rates = compute_sycophancy_rates(records)
    reset_fracs = compute_reset_fraction(rates)
    domain_rates = per_domain_rates(records)

    # Chi-squared: each injection condition vs agree_only baseline
    agree = rates.get("agree_only", {})
    tests = {}
    for cond in ["inject_1", "inject_3", "inject_5", "inject_10"]:
        if cond in rates:
            tests[cond + "_vs_agree"] = chi_squared_test(
                rates[cond]["sycophantic"], rates[cond]["total"],
                agree.get("sycophantic", 0), agree.get("total", 0),
            )

    report = {
        "model": model_slug,
        "n_records": len(records),
        "sycophancy_rates": rates,
        "reset_fractions": reset_fracs,
        "chi_squared_vs_agree": tests,
        "domain_breakdown": domain_rates,
    }

    if verbose:
        log.info(f"\n{'='*60}")
        log.info(f"Model: {model_slug} ({len(records)} records)")
        log.info(f"{'='*60}")

        # Ordered conditions for display
        cond_order = ["agree_only", "inject_1", "inject_3", "inject_5", "inject_10", "correct_only"]
        log.info(f"{'Condition':<16} {'Rate':>8} {'Syc':>6} {'N':>6} {'Reset%':>8} {'χ² p':>8}")
        log.info("-" * 60)
        for c in cond_order:
            if c not in rates:
                continue
            r = rates[c]
            rf = reset_fracs.get(c, "—")
            if isinstance(rf, float):
                rf = f"{rf*100:.1f}%"
            p_str = "—"
            test_key = f"{c}_vs_agree"
            if test_key in tests:
                t = tests[test_key]
                p_str = f"{'<' if t['significant_05'] else '>'}{t['p_approx']}"
            log.info(f"{c:<16} {r['rate_pct']:>8} {r['sycophantic']:>6} {r['total']:>6} {rf:>8} {p_str:>8}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze Correction Injection Results")
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--output", default="figures/injection_report.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_data = load_judged_results(args.results_dir)
    if not model_data:
        log.warning("No *_injection_judged.jsonl files found. Run the judge first.")
        return

    all_reports = {}
    for slug, records in model_data.items():
        report = analyze_model(records, slug, verbose=args.verbose)
        all_reports[slug] = report

    # Cross-model summary
    if len(all_reports) > 1 and args.verbose:
        log.info(f"\n{'='*60}")
        log.info("CROSS-MODEL SUMMARY: Reset Fractions")
        log.info(f"{'='*60}")
        header = f"{'Model':<35} " + " ".join(f"{'inj_'+str(n):>8}" for n in [1, 3, 5, 10])
        log.info(header)
        log.info("-" * 75)
        for slug, report in all_reports.items():
            rf = report["reset_fractions"]
            vals = []
            for c in ["inject_1", "inject_3", "inject_5", "inject_10"]:
                v = rf.get(c)
                vals.append(f"{v*100:.1f}%" if v is not None else "—")
            log.info(f"{slug:<35} " + " ".join(f"{v:>8}" for v in vals))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_reports, f, indent=2)
    log.info(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
