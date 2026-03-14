#!/usr/bin/env python3
"""
Secondary Analysis: Response Taxonomy, Latency, and Length

1. TAXONOMY — reads LLM-judged failure modes from judged JSONL files:
   - Direct validation: "You're right!", short affirmation, no nuance
   - Elaborate justification: Builds structured arguments supporting the false claim
   - Qualified agreement: Hedged, contains caveats, but still net validates

   Taxonomy labels are produced by taxonomy_judge.py (Sonnet 4.6).
   This script only reads and analyzes them — zero API calls.

2. LATENCY — are sycophantic responses faster or slower than honest ones?

3. LENGTH — are sycophantic responses shorter or longer?

All analysis runs on existing data. Zero API calls.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

MODEL_SHORT = {
    "qwen/qwen-2.5-7b-instruct": "Qwen 7B",
    "qwen/qwen-2.5-72b-instruct": "Qwen 72B",
    "mistralai/mistral-small-24b-instruct-2501": "Mistral 24B",
    "deepseek/deepseek-chat-v3.1": "DeepSeek V3.1",
    "google/gemma-3n-e4b-it": "Gemma 3N",
    "mistralai/mixtral-8x7b-instruct": "Mixtral 8x7B",
}

# Model ordering by size for consistent plotting
MODEL_ORDER = [
    "google/gemma-3n-e4b-it",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "mistralai/mistral-small-24b-instruct-2501",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen-2.5-72b-instruct",
]

VALID_MODES = {"direct", "elaborate", "qualified"}


# ── Data Loading ────────────────────────────────────────────────

def load_all_judged():
    rows = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith("_judged.jsonl"):
            continue
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            for line in f:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df[df["is_sycophantic"] >= 0].copy()
    df["response_len"] = df["response"].fillna("").str.len()
    df["response_words"] = df["response"].fillna("").str.split().str.len()
    return df


# ── 1. RESPONSE TAXONOMY ───────────────────────────────────────

def taxonomy_analysis(df):
    """Read taxonomy labels from judged data. Labels come from taxonomy_judge.py."""
    syc = df[df["is_sycophantic"] == 1].copy()

    if "failure_mode" not in syc.columns:
        print("  WARNING: no failure_mode field found in judged data.")
        print("  Run taxonomy_judge.py first to classify sycophantic responses.")
        print("  Skipping taxonomy analysis.")
        return None

    # Count how many have valid labels vs missing/unknown
    has_label = syc["failure_mode"].isin(VALID_MODES)
    n_valid = has_label.sum()
    n_missing = len(syc) - n_valid
    if n_missing > 0:
        print(f"  {n_missing} sycophantic responses missing taxonomy label "
              f"({n_missing/len(syc):.1%}) — excluded from taxonomy plots")
    syc = syc[has_label].copy()

    return syc


def plot_taxonomy_stacked(syc_df):
    """Stacked bar: failure mode proportions per model."""
    models = [m for m in MODEL_ORDER if m in syc_df["model"].unique()]
    modes = ["direct", "elaborate", "qualified"]
    colors = {"direct": "#F44336", "elaborate": "#FF9800", "qualified": "#FFC107"}
    labels = {"direct": "Direct validation", "elaborate": "Elaborate justification",
              "qualified": "Qualified agreement"}

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.6
    bottoms = np.zeros(len(models))

    for mode in modes:
        proportions = []
        for m in models:
            mdf = syc_df[syc_df["model"] == m]
            total = len(mdf)
            count = len(mdf[mdf["failure_mode"] == mode])
            proportions.append(count / total * 100 if total > 0 else 0)
        bars = ax.bar(x, proportions, width, bottom=bottoms, label=labels[mode],
                      color=colors[mode], alpha=0.85)
        # Label bars with count
        for i, (p, b) in enumerate(zip(proportions, bottoms)):
            if p > 5:
                mdf = syc_df[syc_df["model"] == models[i]]
                count = len(mdf[mdf["failure_mode"] == mode])
                ax.text(x[i], b + p/2, f"{p:.0f}%\n({count})",
                        ha="center", va="center", fontsize=8, fontweight="bold")
        bottoms += proportions

    short_names = [MODEL_SHORT.get(m, m.split("/")[-1]) for m in models]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Proportion of sycophantic responses (%)", fontsize=11)
    ax.set_title("How Models Cave: Sycophantic Failure Mode Taxonomy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "taxonomy_stacked.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_taxonomy_by_context(syc_df):
    """Line plot: do failure modes shift with context level?"""
    models = [m for m in MODEL_ORDER if m in syc_df["model"].unique()]
    modes = ["direct", "elaborate", "qualified"]
    colors = {"direct": "#F44336", "elaborate": "#FF9800", "qualified": "#FFC107"}
    labels = {"direct": "Direct", "elaborate": "Elaborate", "qualified": "Qualified"}

    # Only plot models with enough sycophantic responses
    plot_models = [m for m in models
                   if len(syc_df[syc_df["model"] == m]) > 100]

    ncols = 3
    nrows = (len(plot_models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, model in enumerate(plot_models):
        ax = axes[idx]
        mdf = syc_df[syc_df["model"] == model]
        short = MODEL_SHORT.get(model, model.split("/")[-1])

        for mode in modes:
            by_level = mdf.groupby("context_level").apply(
                lambda g: (g["failure_mode"] == mode).mean() * 100
            ).reset_index(name="pct")
            ax.plot(by_level["context_level"] * 100, by_level["pct"],
                    marker="o", markersize=4, label=labels[mode],
                    color=colors[mode], alpha=0.8)

        ax.set_title(short, fontsize=11, fontweight="bold")
        ax.set_xlabel("Context fill (%)", fontsize=9)
        ax.set_ylabel("Mode share (%)", fontsize=9)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(len(plot_models), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Failure Mode Composition Across Context Fill",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "taxonomy_by_context.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 2. LATENCY ANALYSIS ────────────────────────────────────────

def latency_analysis(df):
    """Compare latency between sycophantic and honest responses."""
    print("\n" + "─" * 60)
    print("LATENCY ANALYSIS")
    print("─" * 60)

    results = {}
    for model in MODEL_ORDER:
        if model not in df["model"].unique():
            continue
        mdf = df[df["model"] == model]
        short = MODEL_SHORT.get(model, model.split("/")[-1])

        syc_lat = mdf[mdf["is_sycophantic"] == 1]["latency_ms"]
        hon_lat = mdf[mdf["is_sycophantic"] == 0]["latency_ms"]

        if len(syc_lat) < 10 or len(hon_lat) < 10:
            continue

        # Mann-Whitney U test
        u_stat, u_p = stats.mannwhitneyu(syc_lat, hon_lat, alternative="two-sided")
        # Effect size: rank-biserial correlation
        n1, n2 = len(syc_lat), len(hon_lat)
        r_rb = 1 - (2 * u_stat) / (n1 * n2)

        results[short] = {
            "syc_median_ms": float(np.median(syc_lat)),
            "hon_median_ms": float(np.median(hon_lat)),
            "diff_ms": float(np.median(syc_lat) - np.median(hon_lat)),
            "diff_pct": float((np.median(syc_lat) - np.median(hon_lat)) / np.median(hon_lat) * 100),
            "mann_whitney_p": float(u_p),
            "rank_biserial_r": float(r_rb),
            "n_syc": int(n1),
            "n_hon": int(n2),
        }

        sig = "***" if u_p < 0.001 else "**" if u_p < 0.01 else "*" if u_p < 0.05 else "ns"
        direction = "SLOWER" if results[short]["diff_ms"] > 0 else "FASTER"
        print(f"  {short:20s} Syc={np.median(syc_lat):.0f}ms  Hon={np.median(hon_lat):.0f}ms  "
              f"diff={results[short]['diff_ms']:+.0f}ms ({results[short]['diff_pct']:+.1f}%)  "
              f"p={u_p:.4f} {sig}  Syc is {direction}")

    return results


def plot_latency(df):
    """Box plot: latency distribution by sycophantic vs honest."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        mdf = df[df["model"] == model]
        short = MODEL_SHORT.get(model, model.split("/")[-1])

        syc_lat = mdf[mdf["is_sycophantic"] == 1]["latency_ms"]
        hon_lat = mdf[mdf["is_sycophantic"] == 0]["latency_ms"]

        # Cap at 99th percentile for visualization
        cap = np.percentile(mdf["latency_ms"].dropna(), 99)
        syc_lat = syc_lat.clip(upper=cap)
        hon_lat = hon_lat.clip(upper=cap)

        flier_props = dict(marker='o', markersize=2, alpha=0.15, markerfacecolor='#888888',
                           markeredgecolor='none')
        bp = ax.boxplot([hon_lat, syc_lat], tick_labels=["Honest", "Sycophantic"],
                       patch_artist=True, widths=0.6,
                       medianprops=dict(color="black", linewidth=2),
                       flierprops=flier_props)
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#F44336")
        bp["boxes"][1].set_alpha(0.6)

        ax.set_title(short, fontsize=11, fontweight="bold")
        ax.set_ylabel("Latency (ms)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Annotate medians
        for i, (data, label) in enumerate([(hon_lat, "Honest"), (syc_lat, "Syc")]):
            med = np.median(data)
            ax.text(i+1, med, f" {med:.0f}ms", va="bottom", fontsize=8, fontweight="bold")

    plt.suptitle("Response Latency: Sycophantic vs Honest",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "latency_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── 3. RESPONSE LENGTH ANALYSIS ────────────────────────────────

def length_analysis(df):
    """Compare response length between sycophantic and honest responses."""
    print("\n" + "─" * 60)
    print("RESPONSE LENGTH ANALYSIS")
    print("─" * 60)

    results = {}
    for model in MODEL_ORDER:
        if model not in df["model"].unique():
            continue
        mdf = df[df["model"] == model]
        short = MODEL_SHORT.get(model, model.split("/")[-1])

        syc_words = mdf[mdf["is_sycophantic"] == 1]["response_words"]
        hon_words = mdf[mdf["is_sycophantic"] == 0]["response_words"]

        if len(syc_words) < 10 or len(hon_words) < 10:
            continue

        u_stat, u_p = stats.mannwhitneyu(syc_words, hon_words, alternative="two-sided")
        n1, n2 = len(syc_words), len(hon_words)
        r_rb = 1 - (2 * u_stat) / (n1 * n2)

        results[short] = {
            "syc_median_words": float(np.median(syc_words)),
            "hon_median_words": float(np.median(hon_words)),
            "diff_words": float(np.median(syc_words) - np.median(hon_words)),
            "diff_pct": float((np.median(syc_words) - np.median(hon_words)) / np.median(hon_words) * 100),
            "mann_whitney_p": float(u_p),
            "rank_biserial_r": float(r_rb),
        }

        sig = "***" if u_p < 0.001 else "**" if u_p < 0.01 else "*" if u_p < 0.05 else "ns"
        direction = "LONGER" if results[short]["diff_words"] > 0 else "SHORTER"
        print(f"  {short:20s} Syc={np.median(syc_words):.0f}w  Hon={np.median(hon_words):.0f}w  "
              f"diff={results[short]['diff_words']:+.0f}w ({results[short]['diff_pct']:+.1f}%)  "
              f"p={u_p:.4f} {sig}  Syc is {direction}")

    return results


def plot_length(df):
    """Box plot: response length by sycophantic vs honest."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        mdf = df[df["model"] == model]
        short = MODEL_SHORT.get(model, model.split("/")[-1])

        syc_w = mdf[mdf["is_sycophantic"] == 1]["response_words"]
        hon_w = mdf[mdf["is_sycophantic"] == 0]["response_words"]

        cap = np.percentile(mdf["response_words"].dropna(), 99)
        syc_w = syc_w.clip(upper=cap)
        hon_w = hon_w.clip(upper=cap)

        flier_props = dict(marker='o', markersize=2, alpha=0.15, markerfacecolor='#888888',
                           markeredgecolor='none')
        bp = ax.boxplot([hon_w, syc_w], tick_labels=["Honest", "Sycophantic"],
                       patch_artist=True, widths=0.6,
                       medianprops=dict(color="black", linewidth=2),
                       flierprops=flier_props)
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#F44336")
        bp["boxes"][1].set_alpha(0.6)

        ax.set_title(short, fontsize=11, fontweight="bold")
        ax.set_ylabel("Response length (words)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        for i, data in enumerate([hon_w, syc_w]):
            med = np.median(data)
            ax.text(i+1, med, f" {med:.0f}w", va="bottom", fontsize=8, fontweight="bold")

    plt.suptitle("Response Length: Sycophantic vs Honest",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "length_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("SECONDARY ANALYSIS")
    print("=" * 60)

    df = load_all_judged()
    print(f"\nLoaded {len(df)} valid results")
    print(f"  Sycophantic: {(df['is_sycophantic']==1).sum()}")
    print(f"  Honest: {(df['is_sycophantic']==0).sum()}")

    # ── 1. Taxonomy ──
    print("\n" + "─" * 60)
    print("RESPONSE TAXONOMY")
    print("─" * 60)

    syc_df = taxonomy_analysis(df)

    if syc_df is not None and len(syc_df) > 0:
        print(f"\nClassified {len(syc_df)} sycophantic responses")

        # Overall distribution
        total = len(syc_df)
        for mode in ["direct", "elaborate", "qualified"]:
            count = (syc_df["failure_mode"] == mode).sum()
            print(f"  {mode:12s}: {count:5d} ({count/total:.1%})")

        # Per-model
        print(f"\n  Per-model breakdown:")
        for model in MODEL_ORDER:
            if model not in syc_df["model"].unique():
                continue
            mdf = syc_df[syc_df["model"] == model]
            short = MODEL_SHORT.get(model, model.split("/")[-1])
            total_m = len(mdf)
            d = (mdf["failure_mode"] == "direct").sum()
            e = (mdf["failure_mode"] == "elaborate").sum()
            q = (mdf["failure_mode"] == "qualified").sum()
            print(f"    {short:20s} direct={d/total_m:.0%}({d})  "
                  f"elaborate={e/total_m:.0%}({e})  qualified={q/total_m:.0%}({q})")

        # Per-domain
        print(f"\n  Per-domain breakdown:")
        for domain in ["factual", "math", "science", "logic", "cs", "opinion"]:
            ddf = syc_df[syc_df["probe_domain"] == domain]
            if len(ddf) == 0:
                continue
            total_d = len(ddf)
            d = (ddf["failure_mode"] == "direct").sum()
            e = (ddf["failure_mode"] == "elaborate").sum()
            q = (ddf["failure_mode"] == "qualified").sum()
            print(f"    {domain:12s} direct={d/total_d:.0%}({d})  "
                  f"elaborate={e/total_d:.0%}({e})  qualified={q/total_d:.0%}({q})")

        plot_taxonomy_stacked(syc_df)
        plot_taxonomy_by_context(syc_df)

        taxonomy_report = {
            "total_sycophantic": len(syc_df),
            "overall": {mode: int((syc_df["failure_mode"] == mode).sum())
                       for mode in ["direct", "elaborate", "qualified"]},
        }
    else:
        print("\n  Taxonomy skipped — run taxonomy_judge.py first")
        taxonomy_report = {"total_sycophantic": 0, "overall": {}, "note": "taxonomy_judge.py not run yet"}

    # ── 2. Latency ──
    latency_results = latency_analysis(df)
    plot_latency(df)

    # ── 3. Length ──
    length_results = length_analysis(df)
    plot_length(df)

    # ── Save report ──
    report = {
        "taxonomy": taxonomy_report,
        "latency": latency_results,
        "length": length_results,
    }
    report_path = os.path.join(FIGURES_DIR, "secondary_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved: {report_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
