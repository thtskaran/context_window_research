#!/usr/bin/env python3
"""
Analyze Fine-Grained 0–10% Context Fill Results

Reads judged fine-grained results for Qwen 7B, computes:
  1. Sycophancy rate per (context_pct, filler_type) — the dose-response curve
  2. Phase transition detection: is there a single sharp step, or a smooth gradient?
  3. Chi-squared tests between adjacent 1% steps
  4. Changepoint analysis: where does the biggest single-step jump occur?
  5. Comparison with original 0% vs 10% data points
  6. Per-domain breakdown at each level
  7. Filler-type interaction: does the step function differ by filler type?

Usage:
    python analyze_finegrained.py \
        --results-dir results/ \
        --output figures/finegrained_report.json \
        --verbose
"""

import json
import os
import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FILLER_TYPES = ["neutral", "agreement", "correction"]
FINE_PCTS = list(range(11))  # 0 to 10


def load_judged_results(results_dir: str) -> list[dict]:
    """Load *_finegrained_judged.jsonl files."""
    records = []
    for f in sorted(Path(results_dir).glob("*_finegrained_judged.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        log.info(f"Loaded {len(records)} judged records from {f.name}")
    return records


def compute_rates(records: list[dict]) -> dict:
    """
    Compute sycophancy rates per (context_pct, filler_type).
    Returns nested dict: rates[pct][filler] = {rate, sycophantic, honest, ambiguous, valid, total}
    """
    buckets = defaultdict(lambda: {"total": 0, "sycophantic": 0, "honest": 0, "ambiguous": 0})

    for r in records:
        pct = r.get("context_pct", int(round(r.get("context_level", 0) * 100)))
        filler = r.get("filler_type", "unknown")
        score = r.get("is_sycophantic", -1)

        key = (pct, filler)
        buckets[key]["total"] += 1
        if score == 1:
            buckets[key]["sycophantic"] += 1
        elif score == 0:
            buckets[key]["honest"] += 1
        else:
            buckets[key]["ambiguous"] += 1

    rates = {}
    for (pct, filler), counts in buckets.items():
        denom = counts["sycophantic"] + counts["honest"]
        rate = counts["sycophantic"] / denom if denom > 0 else 0
        rates.setdefault(pct, {})[filler] = {
            "rate": round(rate, 4),
            "rate_pct": f"{rate*100:.1f}%",
            "sycophantic": counts["sycophantic"],
            "honest": counts["honest"],
            "ambiguous": counts["ambiguous"],
            "valid": denom,
            "total": counts["total"],
        }

    return rates


def chi_squared_test(a_syc, a_total, b_syc, b_total) -> dict:
    """2x2 chi-squared test between two proportions."""
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


def adjacent_step_tests(rates: dict, filler: str) -> list[dict]:
    """Chi-squared between each pair of adjacent 1% steps for one filler type."""
    steps = []
    for pct in range(1, 11):
        prev = rates.get(pct - 1, {}).get(filler, {})
        curr = rates.get(pct, {}).get(filler, {})
        if not prev or not curr:
            continue

        test = chi_squared_test(
            prev["sycophantic"], prev["valid"],
            curr["sycophantic"], curr["valid"],
        )
        delta = curr["rate"] - prev["rate"]
        steps.append({
            "from_pct": pct - 1,
            "to_pct": pct,
            "from_rate": prev["rate_pct"],
            "to_rate": curr["rate_pct"],
            "delta": round(delta, 4),
            "delta_pp": f"{delta*100:+.1f}pp",
            "chi2": test["chi2"],
            "p_approx": test["p_approx"],
            "significant_05": test["significant_05"],
            "significant_01": test["significant_01"],
        })
    return steps


def detect_changepoint(rates: dict, filler: str) -> dict:
    """
    Find the single biggest step between adjacent levels.
    Also detect if there's a clear phase transition (one dominant step)
    vs a smooth gradient (multiple small steps).
    """
    steps = adjacent_step_tests(rates, filler)
    if not steps:
        return {"pattern": "insufficient_data"}

    # Sort by absolute delta
    sorted_steps = sorted(steps, key=lambda s: abs(s["delta"]), reverse=True)
    biggest = sorted_steps[0]

    # Total range
    rate_0 = rates.get(0, {}).get(filler, {}).get("rate", 0)
    rate_10 = rates.get(10, {}).get(filler, {}).get("rate", 0)
    total_range = rate_10 - rate_0

    # Phase transition test: does the biggest step account for >50% of the total range?
    if total_range != 0:
        biggest_fraction = abs(biggest["delta"]) / abs(total_range)
    else:
        biggest_fraction = 0

    # Count how many steps are significant
    sig_steps = [s for s in steps if s["significant_05"]]

    # Classification
    if biggest_fraction > 0.50 and biggest["significant_05"]:
        pattern = "phase_transition"
        description = (f"Phase transition at {biggest['from_pct']}%→{biggest['to_pct']}%: "
                       f"biggest step explains {biggest_fraction:.0%} of total range")
    elif len(sig_steps) >= 3:
        pattern = "smooth_gradient"
        description = f"Smooth gradient: {len(sig_steps)} significant steps across 0–10%"
    elif len(sig_steps) == 0:
        pattern = "flat"
        description = f"No significant variation across 0–10% (total range: {total_range*100:.1f}pp)"
    else:
        pattern = "noisy_step"
        description = (f"Noisy step: biggest at {biggest['from_pct']}%→{biggest['to_pct']}% "
                       f"({biggest_fraction:.0%} of range), {len(sig_steps)} sig steps")

    return {
        "pattern": pattern,
        "description": description,
        "biggest_step": biggest,
        "biggest_fraction": round(biggest_fraction, 3),
        "total_range_pp": f"{total_range*100:.1f}pp",
        "n_significant_steps": len(sig_steps),
        "all_steps": steps,
    }


def wilson_ci(successes, total, z=1.96):
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0, 0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (max(0, round(center - spread, 4)), min(1, round(center + spread, 4)))


def compute_ci_bands(rates: dict) -> dict:
    """Compute 95% Wilson confidence intervals for each (pct, filler)."""
    ci_data = {}
    for pct in FINE_PCTS:
        for filler in FILLER_TYPES:
            info = rates.get(pct, {}).get(filler, {})
            if not info:
                continue
            lo, hi = wilson_ci(info["sycophantic"], info["valid"])
            ci_data.setdefault(pct, {})[filler] = {
                "rate": info["rate"],
                "ci_lo": lo,
                "ci_hi": hi,
                "n": info["valid"],
            }
    return ci_data


def per_domain_rates(records: list[dict]) -> dict:
    """Sycophancy rate per (context_pct, domain)."""
    buckets = defaultdict(lambda: {"total": 0, "sycophantic": 0, "honest": 0})
    for r in records:
        pct = r.get("context_pct", int(round(r.get("context_level", 0) * 100)))
        domain = r.get("probe_domain", "unknown")
        score = r.get("is_sycophantic", -1)
        key = (pct, domain)
        buckets[key]["total"] += 1
        if score == 1:
            buckets[key]["sycophantic"] += 1
        elif score == 0:
            buckets[key]["honest"] += 1

    result = {}
    for (pct, domain), counts in buckets.items():
        denom = counts["sycophantic"] + counts["honest"]
        rate = counts["sycophantic"] / denom if denom > 0 else 0
        result.setdefault(pct, {})[domain] = {
            "rate": round(rate, 4),
            "n": counts["total"],
        }
    return result


def filler_interaction_test(rates: dict) -> dict:
    """
    Test whether the step function differs by filler type.
    For each filler type, compute the total 0%→10% delta and compare.
    """
    deltas = {}
    for filler in FILLER_TYPES:
        r0 = rates.get(0, {}).get(filler, {}).get("rate", 0)
        r10 = rates.get(10, {}).get(filler, {}).get("rate", 0)
        deltas[filler] = {
            "rate_0pct": round(r0, 4),
            "rate_10pct": round(r10, 4),
            "delta": round(r10 - r0, 4),
            "delta_pp": f"{(r10 - r0)*100:+.1f}pp",
        }

    # Compare agreement vs correction delta magnitudes
    agree_delta = abs(deltas.get("agreement", {}).get("delta", 0))
    correct_delta = abs(deltas.get("correction", {}).get("delta", 0))
    neutral_delta = abs(deltas.get("neutral", {}).get("delta", 0))

    return {
        "per_filler": deltas,
        "agreement_delta_pp": f"{agree_delta*100:.1f}pp",
        "correction_delta_pp": f"{correct_delta*100:.1f}pp",
        "neutral_delta_pp": f"{neutral_delta*100:.1f}pp",
        "filler_dependent": agree_delta > 2 * correct_delta or correct_delta > 2 * agree_delta,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Fine-Grained 0–10%% Results")
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--output", default="figures/finegrained_report.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    records = load_judged_results(args.results_dir)
    if not records:
        log.warning("No *_finegrained_judged.jsonl files found. Run the judge first.")
        return

    rates = compute_rates(records)
    ci_bands = compute_ci_bands(rates)
    domain_rates = per_domain_rates(records)
    filler_interaction = filler_interaction_test(rates)

    # Changepoint analysis per filler type
    changepoints = {}
    for filler in FILLER_TYPES:
        changepoints[filler] = detect_changepoint(rates, filler)

    # 0% vs 10% significance test per filler
    zero_vs_ten = {}
    for filler in FILLER_TYPES:
        r0 = rates.get(0, {}).get(filler, {})
        r10 = rates.get(10, {}).get(filler, {})
        if r0 and r10:
            test = chi_squared_test(
                r0["sycophantic"], r0["valid"],
                r10["sycophantic"], r10["valid"],
            )
            zero_vs_ten[filler] = test

    report = {
        "model": "qwen/qwen-2.5-7b-instruct",
        "n_records": len(records),
        "sycophancy_rates": {str(k): v for k, v in rates.items()},
        "confidence_intervals": {str(k): v for k, v in ci_bands.items()},
        "changepoint_analysis": changepoints,
        "zero_vs_ten_tests": zero_vs_ten,
        "filler_interaction": filler_interaction,
        "domain_breakdown": {str(k): v for k, v in domain_rates.items()},
    }

    if args.verbose:
        log.info(f"\n{'='*70}")
        log.info(f"FINE-GRAINED 0–10% ANALYSIS (Qwen 7B)")
        log.info(f"{'='*70}")
        log.info(f"Total records: {len(records)}")

        for filler in FILLER_TYPES:
            log.info(f"\n{'─'*50}")
            log.info(f"Filler: {filler}")
            log.info(f"{'─'*50}")
            log.info(f"{'Pct':>4}  {'Rate':>8}  {'Syc':>5}  {'Valid':>5}  "
                     f"{'95% CI':>16}  {'Δ prev':>10}  {'χ² adj':>10}")
            log.info("-" * 70)

            steps = adjacent_step_tests(rates, filler)
            step_map = {s["to_pct"]: s for s in steps}

            for pct in FINE_PCTS:
                info = rates.get(pct, {}).get(filler, {})
                ci = ci_bands.get(pct, {}).get(filler, {})
                if not info:
                    log.info(f"{pct:>3}%  —")
                    continue

                ci_str = f"[{ci.get('ci_lo', 0):.3f}, {ci.get('ci_hi', 0):.3f}]"
                delta_str = "—"
                chi_str = "—"
                if pct in step_map:
                    s = step_map[pct]
                    delta_str = s["delta_pp"]
                    sig = "***" if s["significant_01"] else ("*" if s["significant_05"] else "ns")
                    chi_str = f"{s['chi2']:.1f} {sig}"

                log.info(f"{pct:>3}%  {info['rate_pct']:>8}  {info['sycophantic']:>5}  "
                         f"{info['valid']:>5}  {ci_str:>16}  {delta_str:>10}  {chi_str:>10}")

            # Changepoint summary
            cp = changepoints.get(filler, {})
            if cp.get("pattern"):
                log.info(f"\nPattern: {cp['description']}")

        # Filler interaction
        log.info(f"\n{'='*50}")
        log.info("FILLER INTERACTION (0%→10% deltas):")
        log.info(f"{'='*50}")
        for filler in FILLER_TYPES:
            d = filler_interaction["per_filler"].get(filler, {})
            log.info(f"  {filler:<12}: {d.get('rate_0pct', 0)*100:.1f}% → "
                     f"{d.get('rate_10pct', 0)*100:.1f}% ({d.get('delta_pp', '?')})")

        # 0% vs 10% tests
        log.info(f"\n0% vs 10% significance:")
        for filler in FILLER_TYPES:
            t = zero_vs_ten.get(filler, {})
            sig = "***" if t.get("significant_01") else ("*" if t.get("significant_05") else "ns")
            log.info(f"  {filler:<12}: χ²={t.get('chi2', 0):.1f} {sig}")

    # Save report
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
