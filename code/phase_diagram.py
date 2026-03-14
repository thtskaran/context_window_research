#!/usr/bin/env python3
"""
Phase Diagram: Sycophancy Rate vs Context Window Utilization

Produces the central figure of the experiment — sycophancy rate plotted as a
function of context window utilization (%), one curve per model, with
bootstrapped 95% confidence intervals.

Also produces:
  - Per-domain breakdown (factual vs math)
  - Per-filler-type breakdown (neutral vs agreement vs correction)
  - Heatmap (model × context level)

Usage:
    python phase_diagram.py \
        --results-dir results/ \
        --output-dir figures/ \
        --models Qwen_Qwen2.5-7B-Instruct google_gemini-2.0-flash-001

    # Or auto-detect all result files:
    python phase_diagram.py --results-dir results/ --output-dir figures/
"""

import json
import os
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Plotting style ──────────────────────────────────────────────

# Defer matplotlib import to allow headless usage
def setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt, mticker


# ─── Data loading ──────────────────────────────────────────────────

# Probes to exclude from analysis (none)
EXCLUDE_PROBES = set()


def load_results(results_dir: str, model_filter: list[str] = None,
                 exclude_probes: set[str] = None) -> dict:
    """
    Load all JSONL result files from results_dir.
    Returns {model_name: [result_dicts]}
    """
    if exclude_probes is None:
        exclude_probes = EXCLUDE_PROBES

    all_results = defaultdict(list)

    for fpath in Path(results_dir).glob("*_results.jsonl"):
        model_slug = fpath.stem.replace("_results", "")
        if model_filter and model_slug not in model_filter:
            continue

        with open(fpath) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("probe_id") not in exclude_probes:
                        all_results[model_slug].append(r)

    # Also load judged files (override if present)
    for fpath in Path(results_dir).glob("*_judged.jsonl"):
        model_slug = fpath.stem.replace("_judged", "")
        if model_filter and model_slug not in model_filter:
            continue

        judged = []
        with open(fpath) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("probe_id") not in exclude_probes:
                        judged.append(r)
        if judged:
            all_results[model_slug] = judged
            log.info(f"Using judged results for {model_slug}")

    for model, results in all_results.items():
        log.info(f"Loaded {len(results)} results for {model}")
    if exclude_probes:
        log.info(f"Excluded probes: {exclude_probes}")

    return dict(all_results)


def compute_sycophancy_rate(results: list[dict], bootstrap_n: int = 1000) -> dict:
    """
    Compute sycophancy rate per context level with bootstrapped 95% CI.
    Returns {context_level: {"rate": float, "ci_low": float, "ci_high": float, "n": int}}
    """
    by_level = defaultdict(list)
    for r in results:
        if r["is_sycophantic"] in (0, 1):  # exclude ambiguous
            by_level[r["context_level"]].append(r["is_sycophantic"])

    stats = {}
    for level in sorted(by_level.keys()):
        scores = np.array(by_level[level])
        n = len(scores)
        rate = scores.mean()

        # Bootstrap 95% CI
        if n >= 5:
            boot_means = []
            for _ in range(bootstrap_n):
                sample = np.random.choice(scores, size=n, replace=True)
                boot_means.append(sample.mean())
            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)
        else:
            ci_low = ci_high = rate

        stats[level] = {"rate": rate, "ci_low": ci_low, "ci_high": ci_high, "n": n}

    return stats


# ─── Plot 1: Main phase diagram ───────────────────────────────────

def plot_phase_diagram(all_results: dict, output_dir: str):
    """
    Main figure: sycophancy rate (%) vs context utilization (%), one curve per model.
    """
    plt, mticker = setup_plotting()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(all_results), 8)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, (model, results) in enumerate(sorted(all_results.items())):
        stats = compute_sycophancy_rate(results)
        if not stats:
            continue

        levels = sorted(stats.keys())
        rates = [stats[l]["rate"] * 100 for l in levels]
        ci_low = [stats[l]["ci_low"] * 100 for l in levels]
        ci_high = [stats[l]["ci_high"] * 100 for l in levels]
        xs = [l * 100 for l in levels]

        # Clean model name for legend
        label = model.replace("_", "/")

        ax.plot(xs, rates, marker=markers[i % len(markers)],
                color=colors[i], linewidth=2, markersize=6, label=label)
        ax.fill_between(xs, ci_low, ci_high, alpha=0.15, color=colors[i])

    ax.set_xlabel("Context Utilization (% of 32K tokens)", fontsize=13)
    ax.set_ylabel("Sycophancy Rate (%)", fontsize=13)
    ax.set_title("Sycophancy as a Function of Context Length", fontsize=15, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, None)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    fig.tight_layout()
    out_path = os.path.join(output_dir, "phase_diagram.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")

    # Also save as PDF for paper
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i, (model, results) in enumerate(sorted(all_results.items())):
        stats = compute_sycophancy_rate(results)
        if not stats:
            continue
        levels = sorted(stats.keys())
        rates = [stats[l]["rate"] * 100 for l in levels]
        ci_low = [stats[l]["ci_low"] * 100 for l in levels]
        ci_high = [stats[l]["ci_high"] * 100 for l in levels]
        xs = [l * 100 for l in levels]
        label = model.replace("_", "/")
        ax2.plot(xs, rates, marker=markers[i % len(markers)],
                 color=colors[i], linewidth=2, markersize=6, label=label)
        ax2.fill_between(xs, ci_low, ci_high, alpha=0.15, color=colors[i])
    ax2.set_xlabel("Context Utilization (% of 32K tokens)", fontsize=13)
    ax2.set_ylabel("Sycophancy Rate (%)", fontsize=13)
    ax2.set_title("Sycophancy as a Function of Context Length", fontsize=15, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax2.set_xlim(-2, 102)
    ax2.set_ylim(-2, None)
    fig2.tight_layout()
    pdf_path = os.path.join(output_dir, "phase_diagram.pdf")
    fig2.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig2)
    log.info(f"Saved: {pdf_path}")


# ─── Plot 2: Per-domain breakdown ─────────────────────────────────

def plot_domain_breakdown(all_results: dict, output_dir: str):
    """
    For each model, show sycophancy rate by domain across context levels.
    """
    plt, mticker = setup_plotting()

    domain_styles = [
        ("factual", "-o"),
        ("math", "--s"),
        ("science", "-.^"),
        ("logic", ":D"),
        ("cs", "-v"),
        ("opinion", "-P"),
    ]

    for model, results in sorted(all_results.items()):
        fig, ax = plt.subplots(figsize=(9, 5))

        for domain, style in domain_styles:
            domain_results = [r for r in results if r.get("probe_domain") == domain]
            if not domain_results:
                continue
            stats = compute_sycophancy_rate(domain_results)
            levels = sorted(stats.keys())
            rates = [stats[l]["rate"] * 100 for l in levels]
            xs = [l * 100 for l in levels]
            ax.plot(xs, rates, style, linewidth=2, markersize=5, label=domain.capitalize())

        ax.set_xlabel("Context Utilization (% of 32K tokens)")
        ax.set_ylabel("Sycophancy Rate (%)")
        ax.set_title(f"Sycophancy by Domain — {model.replace('_', '/')}", fontweight="bold")
        ax.legend()
        ax.set_xlim(-2, 102)

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"domain_breakdown_{model}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved: {out_path}")


# ─── Plot 3: Filler type comparison ───────────────────────────────

def plot_filler_comparison(all_results: dict, output_dir: str):
    """
    For each model, show sycophancy rate by filler type across context levels.
    This is the key figure for distinguishing LENGTH vs PATTERN effects.
    """
    plt, mticker = setup_plotting()

    filler_styles = {
        "neutral": ("-o", "#2196F3"),
        "agreement": ("--^", "#F44336"),
        "correction": ("-.s", "#4CAF50"),
    }

    for model, results in sorted(all_results.items()):
        fig, ax = plt.subplots(figsize=(10, 6))

        for ftype, (style, color) in filler_styles.items():
            ftype_results = [r for r in results if r.get("filler_type") == ftype]
            if not ftype_results:
                continue
            stats = compute_sycophancy_rate(ftype_results)
            levels = sorted(stats.keys())
            rates = [stats[l]["rate"] * 100 for l in levels]
            ci_low = [stats[l]["ci_low"] * 100 for l in levels]
            ci_high = [stats[l]["ci_high"] * 100 for l in levels]
            xs = [l * 100 for l in levels]

            ax.plot(xs, rates, style, color=color, linewidth=2, markersize=6,
                    label=f"{ftype.capitalize()} filler")
            ax.fill_between(xs, ci_low, ci_high, alpha=0.1, color=color)

        ax.set_xlabel("Context Utilization (% of 32K tokens)", fontsize=12)
        ax.set_ylabel("Sycophancy Rate (%)", fontsize=12)
        ax.set_title(f"Filler Type Comparison — {model.replace('_', '/')}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim(-2, 102)

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"filler_comparison_{model}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved: {out_path}")


# ─── Plot 4: Heatmap (model × context level) ──────────────────────

def plot_heatmap(all_results: dict, output_dir: str):
    """
    Heatmap showing sycophancy rate for each (model, context_level) pair.
    """
    plt, _ = setup_plotting()

    models = sorted(all_results.keys())
    if not models:
        return

    # Collect all context levels
    all_levels = set()
    for results in all_results.values():
        for r in results:
            all_levels.add(r["context_level"])
    levels = sorted(all_levels)

    # Build matrix
    matrix = np.full((len(models), len(levels)), np.nan)
    for i, model in enumerate(models):
        stats = compute_sycophancy_rate(all_results[model])
        for j, level in enumerate(levels):
            if level in stats:
                matrix[i, j] = stats[level]["rate"] * 100

    fig, ax = plt.subplots(figsize=(12, max(3, len(models) * 0.8)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{l*100:.0f}%" for l in levels], fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace("_", "/") for m in models], fontsize=9)
    ax.set_xlabel("Context Utilization (% of 32K tokens)")
    ax.set_title("Sycophancy Rate Heatmap (% sycophantic)", fontweight="bold")

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(levels)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Sycophancy Rate (%)")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")


# ─── Summary table ────────────────────────────────────────────────

def print_summary(all_results: dict):
    """Print a text summary table of key metrics."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    for model, results in sorted(all_results.items()):
        valid = [r for r in results if r["is_sycophantic"] in (0, 1)]
        syc = [r for r in valid if r["is_sycophantic"] == 1]
        ambig = [r for r in results if r["is_sycophantic"] == -1]

        print(f"\n{'─' * 60}")
        print(f"Model: {model.replace('_', '/')}")
        print(f"  Total responses: {len(results)} (valid: {len(valid)}, ambiguous: {len(ambig)})")
        print(f"  Overall sycophancy rate: {len(syc)/max(len(valid),1)*100:.1f}%")

        # Rate at lowest and highest context
        stats = compute_sycophancy_rate(results)
        levels = sorted(stats.keys())
        if len(levels) >= 2:
            low = stats[levels[0]]
            high = stats[levels[-1]]
            print(f"  At {levels[0]*100:.0f}% context: {low['rate']*100:.1f}% "
                  f"(n={low['n']}, 95% CI: [{low['ci_low']*100:.1f}%, {low['ci_high']*100:.1f}%])")
            print(f"  At {levels[-1]*100:.0f}% context: {high['rate']*100:.1f}% "
                  f"(n={high['n']}, 95% CI: [{high['ci_low']*100:.1f}%, {high['ci_high']*100:.1f}%])")
            delta = high['rate'] - low['rate']
            print(f"  Delta (high - low): {delta*100:+.1f} percentage points")

        # By filler type
        for ftype in ["neutral", "agreement", "correction"]:
            ftype_results = [r for r in valid if r.get("filler_type") == ftype]
            if ftype_results:
                ftype_syc = [r for r in ftype_results if r["is_sycophantic"] == 1]
                print(f"  {ftype.capitalize()} filler rate: "
                      f"{len(ftype_syc)/len(ftype_results)*100:.1f}% (n={len(ftype_results)})")

    print("\n" + "=" * 80)


# ─── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase Diagram: Sycophancy vs Context Length")
    parser.add_argument("--results-dir", default="results", help="Directory with JSONL results")
    parser.add_argument("--output-dir", default="figures", help="Output directory for figures")
    parser.add_argument("--models", nargs="*", default=None, help="Model slugs to include")
    parser.add_argument("--no-summary", action="store_true", help="Skip text summary")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = load_results(args.results_dir, args.models)

    if not all_results:
        log.error(f"No results found in {args.results_dir}")
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
        log.error("No models with sufficient data after filtering")
        return

    if not args.no_summary:
        print_summary(all_results)

    plot_phase_diagram(all_results, args.output_dir)
    plot_domain_breakdown(all_results, args.output_dir)
    plot_filler_comparison(all_results, args.output_dir)
    plot_heatmap(all_results, args.output_dir)

    log.info(f"All figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
