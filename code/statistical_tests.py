#!/usr/bin/env python3
"""
Statistical Tests for the Sycophancy-Context Experiment

Runs the full battery of statistical analyses:
  1. Spearman correlation: context level vs sycophancy (per model)
  2. Mann-Whitney U: sycophancy at 0% vs 80%+ context (per model)
  3. Chi-squared: sycophancy rate across filler types (per model)
  4. Mixed-effects logistic regression: sycophancy ~ context + filler + (1|probe)
  5. Effect size: Cohen's h for 0% vs max context
  6. Trend detection: monotonic vs threshold vs flat

Usage:
    python statistical_tests.py --results-dir results/ --output stats_report.json

    # Also saves a human-readable report:
    python statistical_tests.py --results-dir results/ --output stats_report.json --verbose
"""

import json
import os
import argparse
import logging
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.generic)):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
        return super().default(obj)


# ─── Data loading (shared with phase_diagram.py) ──────────────────

EXCLUDE_PROBES = set()  # no excluded probes


def load_results(results_dir: str, exclude_probes: set[str] = None) -> dict:
    if exclude_probes is None:
        exclude_probes = EXCLUDE_PROBES

    all_results = defaultdict(list)
    for fpath in Path(results_dir).glob("*_results.jsonl"):
        model = fpath.stem.replace("_results", "")
        with open(fpath) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("probe_id") not in exclude_probes:
                        all_results[model].append(r)
    # Prefer judged files
    for fpath in Path(results_dir).glob("*_judged.jsonl"):
        model = fpath.stem.replace("_judged", "")
        judged = []
        with open(fpath) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("probe_id") not in exclude_probes:
                        judged.append(r)
        if judged:
            all_results[model] = judged
    if exclude_probes:
        log.info(f"Excluded probes: {exclude_probes}")
    return dict(all_results)


# ─── Test 1: Spearman correlation ─────────────────────────────────

def test_spearman(results: list[dict]) -> dict:
    """Spearman rank correlation between context_level and is_sycophantic."""
    from scipy.stats import spearmanr

    valid = [(r["context_level"], r["is_sycophantic"])
             for r in results if r["is_sycophantic"] in (0, 1)]
    if len(valid) < 10:
        return {"rho": None, "p_value": None, "n": len(valid), "error": "insufficient data"}

    levels, scores = zip(*valid)
    rho, p = spearmanr(levels, scores)
    return {
        "rho": round(float(rho), 4),
        "p_value": float(p),
        "n": len(valid),
        "significant": p < 0.05,
        "interpretation": (
            "positive correlation (sycophancy increases with context)" if rho > 0 and p < 0.05
            else "negative correlation (sycophancy decreases with context)" if rho < 0 and p < 0.05
            else "no significant correlation"
        ),
    }


# ─── Test 2: Mann-Whitney U ───────────────────────────────────────

def test_mann_whitney(results: list[dict], low_threshold: float = 0.1,
                      high_threshold: float = 0.8) -> dict:
    """
    Mann-Whitney U: compare sycophancy scores at low context (≤ threshold)
    vs high context (≥ threshold).
    """
    from scipy.stats import mannwhitneyu

    low = [r["is_sycophantic"] for r in results
           if r["is_sycophantic"] in (0, 1) and r["context_level"] <= low_threshold]
    high = [r["is_sycophantic"] for r in results
            if r["is_sycophantic"] in (0, 1) and r["context_level"] >= high_threshold]

    if len(low) < 5 or len(high) < 5:
        return {"U": None, "p_value": None, "error": "insufficient data",
                "n_low": len(low), "n_high": len(high)}

    U, p = mannwhitneyu(high, low, alternative="greater")
    return {
        "U": float(U),
        "p_value": float(p),
        "n_low": len(low),
        "n_high": len(high),
        "mean_low": round(np.mean(low), 4),
        "mean_high": round(np.mean(high), 4),
        "significant": p < 0.05,
        "interpretation": (
            f"sycophancy significantly higher at ≥{high_threshold*100:.0f}% context "
            f"vs ≤{low_threshold*100:.0f}% (p={p:.4f})" if p < 0.05
            else f"no significant difference (p={p:.4f})"
        ),
    }


# ─── Test 3: Chi-squared across filler types ──────────────────────

def test_chi_squared_filler(results: list[dict]) -> dict:
    """Chi-squared test: is sycophancy rate different across filler types?"""
    from scipy.stats import chi2_contingency

    contingency = {}
    for ftype in ["neutral", "agreement", "correction"]:
        ftype_results = [r for r in results
                         if r.get("filler_type") == ftype and r["is_sycophantic"] in (0, 1)]
        if ftype_results:
            syc = sum(1 for r in ftype_results if r["is_sycophantic"] == 1)
            honest = len(ftype_results) - syc
            contingency[ftype] = (syc, honest)

    if len(contingency) < 2:
        return {"chi2": None, "error": "need at least 2 filler types with data"}

    table = np.array(list(contingency.values()))
    try:
        chi2, p, dof, expected = chi2_contingency(table)
    except ValueError as e:
        return {"chi2": None, "error": str(e)}

    rates = {k: v[0] / sum(v) for k, v in contingency.items()}

    return {
        "chi2": round(float(chi2), 4),
        "p_value": float(p),
        "dof": int(dof),
        "significant": p < 0.05,
        "rates_by_filler": {k: round(v, 4) for k, v in rates.items()},
        "interpretation": (
            "filler type significantly affects sycophancy rate" if p < 0.05
            else "no significant effect of filler type"
        ),
    }


# ─── Test 4: Mixed-effects logistic regression ────────────────────

def test_mixed_effects(results: list[dict]) -> dict:
    """
    Mixed-effects logistic regression (GLMM):
        sycophancy ~ context_level + filler_type + (1|probe_id)

    Uses statsmodels BinomialBayesMixedGLM for proper binary outcome modeling.
    Falls back to standard logit if GLMM fails to converge.
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        import statsmodels.genmod.bayes_mixed_glm as bm
    except ImportError:
        return {"error": "statsmodels and pandas required: pip install statsmodels pandas"}

    valid = [r for r in results if r["is_sycophantic"] in (0, 1)]
    if len(valid) < 30:
        return {"error": "insufficient data for mixed-effects model"}

    df = pd.DataFrame(valid)
    df["syc"] = df["is_sycophantic"].astype(float)
    df["ctx"] = df["context_level"].astype(float)

    # Encode filler type as dummy variables (neutral = reference)
    if "filler_type" in df.columns:
        df["filler_agreement"] = (df["filler_type"] == "agreement").astype(float)
        df["filler_correction"] = (df["filler_type"] == "correction").astype(float)
    else:
        df["filler_agreement"] = 0.0
        df["filler_correction"] = 0.0

    try:
        # Primary: Bayesian mixed-effects logistic regression (GLMM)
        # This is the correct model for a binary DV with random intercepts
        random_effects = {"probe_id": "0 + C(probe_id)"}
        model = bm.BinomialBayesMixedGLM.from_formula(
            "syc ~ ctx + filler_agreement + filler_correction",
            random_effects,
            data=df,
        )
        result = model.fit_vb()

        params = result.params
        # Extract fixed effects (first 4 params: intercept, ctx, agreement, correction)
        fe_names = result.model.exog_names
        fe_params = {name: float(params[i]) for i, name in enumerate(fe_names)}

        return {
            "method": "binomial_bayesian_mixed_glm",
            "context_coef": round(fe_params.get("ctx", 0), 4),
            "agreement_coef": round(fe_params.get("filler_agreement", 0), 4),
            "correction_coef": round(fe_params.get("filler_correction", 0), 4),
            "intercept": round(fe_params.get("Intercept", 0), 4),
            "n_observations": len(df),
            "n_groups": df["probe_id"].nunique(),
            "note": "Coefficients are log-odds. Positive context_coef = sycophancy increases with context.",
        }
    except Exception as e:
        log.warning(f"Bayesian GLMM failed ({e}), trying GEE logistic")

        try:
            # Fallback 1: GEE with logit link — proper for binary outcome with clustering
            import statsmodels.api as sm
            from statsmodels.genmod.generalized_estimating_equations import GEE
            from statsmodels.genmod.families import Binomial
            from statsmodels.genmod.cov_struct import Exchangeable

            df_sorted = df.sort_values("probe_id").reset_index(drop=True)
            exog = df_sorted[["ctx", "filler_agreement", "filler_correction"]]
            exog = sm.add_constant(exog)
            model = GEE(
                df_sorted["syc"], exog,
                groups=df_sorted["probe_id"],
                family=Binomial(),
                cov_struct=Exchangeable(),
            )
            result = model.fit()

            return {
                "method": "gee_logistic",
                "context_coef": round(float(result.params.get("ctx", 0)), 4),
                "context_pvalue": float(result.pvalues.get("ctx", 1)),
                "agreement_coef": round(float(result.params.get("filler_agreement", 0)), 4),
                "agreement_pvalue": float(result.pvalues.get("filler_agreement", 1)),
                "correction_coef": round(float(result.params.get("filler_correction", 0)), 4),
                "correction_pvalue": float(result.pvalues.get("filler_correction", 1)),
                "n_observations": len(df_sorted),
                "n_groups": df_sorted["probe_id"].nunique(),
                "note": "GEE with exchangeable correlation, logit link. Coefficients are log-odds.",
            }
        except Exception as e2:
            log.warning(f"GEE failed ({e2}), falling back to standard logit")

            # Fallback 2: plain logistic regression (no random effects)
            try:
                model = smf.logit("syc ~ ctx + filler_agreement + filler_correction", data=df)
                result = model.fit(disp=0)
                return {
                    "method": "logistic_regression_fallback",
                    "context_coef": round(float(result.params.get("ctx", 0)), 4),
                    "context_pvalue": float(result.pvalues.get("ctx", 1)),
                    "agreement_coef": round(float(result.params.get("filler_agreement", 0)), 4),
                    "correction_coef": round(float(result.params.get("filler_correction", 0)), 4),
                    "pseudo_r2": round(float(result.prsquared), 4),
                    "n_observations": len(df),
                    "note": "Random effects failed; using fixed-effects logit. Coefficients are log-odds.",
                }
            except Exception as e3:
                return {"error": f"All models failed: GLMM({e}), GEE({e2}), logit({e3})"}


# ─── Test 5: Cohen's h effect size ────────────────────────────────

def test_effect_size(results: list[dict]) -> dict:
    """
    Cohen's h: standardized effect size for difference between two proportions.
    Compares sycophancy rate at 0% context vs max context.
    """
    by_level = defaultdict(list)
    for r in results:
        if r["is_sycophantic"] in (0, 1):
            by_level[r["context_level"]].append(r["is_sycophantic"])

    levels = sorted(by_level.keys())
    if len(levels) < 2:
        return {"error": "need at least 2 context levels"}

    p1_data = by_level[levels[0]]
    p2_data = by_level[levels[-1]]

    if not p1_data or not p2_data:
        return {"error": "empty data at extreme levels"}

    p1 = np.mean(p1_data)  # low context
    p2 = np.mean(p2_data)  # high context

    # Cohen's h = 2 * arcsin(sqrt(p2)) - 2 * arcsin(sqrt(p1))
    h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))

    magnitude = (
        "negligible" if abs(h) < 0.2 else
        "small" if abs(h) < 0.5 else
        "medium" if abs(h) < 0.8 else
        "large"
    )

    return {
        "cohens_h": round(float(h), 4),
        "p_low": round(float(p1), 4),
        "p_high": round(float(p2), 4),
        "context_low": levels[0],
        "context_high": levels[-1],
        "magnitude": magnitude,
        "interpretation": (
            f"{'Large' if abs(h) >= 0.8 else 'Medium' if abs(h) >= 0.5 else 'Small' if abs(h) >= 0.2 else 'Negligible'} "
            f"effect (h={h:.3f}): sycophancy {'increases' if h > 0 else 'decreases'} from "
            f"{p1*100:.1f}% at {levels[0]*100:.0f}% context to "
            f"{p2*100:.1f}% at {levels[-1]*100:.0f}% context"
        ),
    }


# ─── Test 6: Trend detection ──────────────────────────────────────

def test_trend(results: list[dict]) -> dict:
    """
    Classify the observed trend:
      - Monotonic increase: consistent upward trend
      - Threshold effect: flat then sharp increase at some point
      - Flat: no meaningful change
      - Non-monotonic: rises then falls, or irregular
    """
    by_level = defaultdict(list)
    for r in results:
        if r["is_sycophantic"] in (0, 1):
            by_level[r["context_level"]].append(r["is_sycophantic"])

    levels = sorted(by_level.keys())
    if len(levels) < 3:
        return {"trend": "insufficient_data", "n_levels": len(levels)}

    rates = [np.mean(by_level[l]) for l in levels]

    # Monotonicity check: count direction changes
    diffs = [rates[i+1] - rates[i] for i in range(len(rates)-1)]
    increasing = sum(1 for d in diffs if d > 0.01)
    decreasing = sum(1 for d in diffs if d < -0.01)
    flat_segments = sum(1 for d in diffs if abs(d) <= 0.01)

    total_change = rates[-1] - rates[0]

    # Threshold detection: find max single-step increase
    max_jump_idx = int(np.argmax(diffs))
    max_jump = diffs[max_jump_idx]
    avg_other_jumps = np.mean([d for i, d in enumerate(diffs) if i != max_jump_idx]) if len(diffs) > 1 else 0

    if abs(total_change) < 0.05:
        trend = "flat"
        description = f"No meaningful change ({total_change*100:+.1f}pp across range)"
    elif increasing >= len(diffs) * 0.7 and decreasing <= 1:
        if max_jump > 3 * abs(avg_other_jumps) and abs(avg_other_jumps) > 0.001:
            trend = "threshold"
            threshold_level = levels[max_jump_idx + 1]
            description = (f"Threshold effect at ~{threshold_level*100:.0f}% context "
                          f"(jump of {max_jump*100:.1f}pp)")
        else:
            trend = "monotonic_increase"
            description = f"Consistent increase ({total_change*100:+.1f}pp across range)"
    elif decreasing >= len(diffs) * 0.7 and increasing <= 1:
        trend = "monotonic_decrease"
        description = f"Consistent decrease ({total_change*100:+.1f}pp across range)"
    else:
        trend = "non_monotonic"
        description = f"Irregular pattern (↑{increasing}, ↓{decreasing}, →{flat_segments})"

    return {
        "trend": trend,
        "description": description,
        "total_change_pp": round(total_change * 100, 2),
        "rates_by_level": {f"{l*100:.0f}%": round(r, 4) for l, r in zip(levels, rates)},
        "n_levels": len(levels),
    }


# ─── Run all tests ────────────────────────────────────────────────

def run_all_tests(all_results: dict) -> dict:
    """Run full test battery for each model."""
    report = {}

    for model, results in sorted(all_results.items()):
        log.info(f"Testing: {model}")
        model_report = {
            "n_total": len(results),
            "n_valid": sum(1 for r in results if r["is_sycophantic"] in (0, 1)),
            "n_ambiguous": sum(1 for r in results if r["is_sycophantic"] == -1),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model_report["spearman"] = test_spearman(results)
            model_report["mann_whitney"] = test_mann_whitney(results)
            model_report["chi_squared_filler"] = test_chi_squared_filler(results)
            model_report["mixed_effects"] = test_mixed_effects(results)
            model_report["effect_size"] = test_effect_size(results)
            model_report["trend"] = test_trend(results)

        report[model] = model_report

    return report


def print_report(report: dict):
    """Human-readable summary of statistical tests."""
    print("\n" + "=" * 80)
    print("STATISTICAL TEST REPORT")
    print("=" * 80)

    for model, tests in report.items():
        print(f"\n{'━' * 70}")
        print(f"Model: {model.replace('_', '/')}")
        print(f"  Data: {tests['n_valid']} valid / {tests['n_total']} total "
              f"({tests['n_ambiguous']} ambiguous)")
        print()

        # Trend
        t = tests["trend"]
        print(f"  TREND: {t['trend'].upper()}")
        print(f"    {t['description']}")
        print()

        # Spearman
        s = tests["spearman"]
        if s.get("rho") is not None:
            sig = "***" if s["p_value"] < 0.001 else "**" if s["p_value"] < 0.01 else "*" if s["p_value"] < 0.05 else "ns"
            print(f"  Spearman ρ = {s['rho']:.3f} (p = {s['p_value']:.4f}) {sig}")
            print(f"    → {s['interpretation']}")
        else:
            print(f"  Spearman: {s.get('error', 'N/A')}")

        # Mann-Whitney
        m = tests["mann_whitney"]
        if m.get("U") is not None:
            sig = "***" if m["p_value"] < 0.001 else "**" if m["p_value"] < 0.01 else "*" if m["p_value"] < 0.05 else "ns"
            print(f"  Mann-Whitney U = {m['U']:.0f} (p = {m['p_value']:.4f}) {sig}")
            print(f"    Low context: {m['mean_low']*100:.1f}% (n={m['n_low']}), "
                  f"High context: {m['mean_high']*100:.1f}% (n={m['n_high']})")
        else:
            print(f"  Mann-Whitney: {m.get('error', 'N/A')}")

        # Effect size
        e = tests["effect_size"]
        if "cohens_h" in e:
            print(f"  Cohen's h = {e['cohens_h']:.3f} ({e['magnitude']})")
        else:
            print(f"  Effect size: {e.get('error', 'N/A')}")

        # Chi-squared
        c = tests["chi_squared_filler"]
        if c.get("chi2") is not None:
            sig = "***" if c["p_value"] < 0.001 else "**" if c["p_value"] < 0.01 else "*" if c["p_value"] < 0.05 else "ns"
            print(f"  Chi-squared (filler types) = {c['chi2']:.2f} (p = {c['p_value']:.4f}) {sig}")
            if "rates_by_filler" in c:
                for ft, rate in c["rates_by_filler"].items():
                    print(f"    {ft}: {rate*100:.1f}%")
        else:
            print(f"  Chi-squared: {c.get('error', 'N/A')}")

        # Mixed effects
        me = tests["mixed_effects"]
        if "context_coef" in me:
            print(f"  Mixed-effects ({me.get('method', 'unknown')}):")
            print(f"    context β = {me['context_coef']:.4f} (p = {me['context_pvalue']:.4f})")
            if "agreement_coef" in me:
                print(f"    agreement β = {me['agreement_coef']:.4f}")
            if "correction_coef" in me:
                print(f"    correction β = {me['correction_coef']:.4f}")
        else:
            print(f"  Mixed-effects: {me.get('error', 'N/A')}")

    print("\n" + "=" * 80)
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant")
    print("=" * 80)


# ─── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Statistical Tests for Sycophancy Experiment")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="stats_report.json")
    parser.add_argument("--verbose", action="store_true", help="Print human-readable report")
    args = parser.parse_args()

    all_results = load_results(args.results_dir)
    if not all_results:
        log.error(f"No results in {args.results_dir}")
        return

    # Filter out models with too few results (e.g. incomplete runs)
    MIN_RESULTS = 100
    filtered = {}
    for model, results in all_results.items():
        if len(results) >= MIN_RESULTS:
            filtered[model] = results
        else:
            log.warning(f"Skipping {model}: only {len(results)} results (min {MIN_RESULTS})")
    all_results = filtered
    if not all_results:
        log.error("No models with sufficient data")
        return

    report = run_all_tests(all_results)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    log.info(f"Report saved to {args.output}")

    if args.verbose:
        print_report(report)
    else:
        # Always print summary
        for model, tests in report.items():
            trend = tests["trend"]["trend"]
            rho = tests["spearman"].get("rho", "N/A")
            p = tests["spearman"].get("p_value", "N/A")
            log.info(f"{model}: trend={trend}, ρ={rho}, p={p}")


if __name__ == "__main__":
    main()
