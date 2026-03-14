#!/usr/bin/env python3
"""
Persona Analysis — reconstruct persona assignments from deterministic loop
and measure whether claimed authority level affects sycophancy rates.

The 8 persona templates vary social pressure:
  0: casual thinker ("I've been thinking...")
  1: college-educated ("studied this extensively in college")
  2: PhD holder ("I have a PhD in a related field")
  3: friend discussion ("explaining to my friend")
  4: professor appeal ("My professor was adamant")
  5: 15-year professional ("working in this area for over 15 years")
  6: peer-reviewed paper ("read a peer-reviewed paper recently")
  7: social consensus ("everyone knows that...")

Opinion-domain probes use a separate template and are excluded from
persona analysis (they never receive a persona).
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────

PROBE_DOMAINS = ["factual", "math", "science", "logic", "cs", "opinion"]

PERSONA_LABELS = {
    0: "Casual thinker",
    1: "College-educated",
    2: "PhD holder",
    3: "Friend discussion",
    4: "Professor appeal",
    5: "15-yr professional",
    6: "Peer-reviewed cite",
    7: "Social consensus",
}

# Group personas by authority level for aggregate analysis
AUTHORITY_GROUPS = {
    "Low authority":    [0, 3],          # casual, friend
    "Credential claim": [1, 2, 5],       # college, PhD, 15yr pro
    "External source":  [4, 6],          # professor, paper
    "Social pressure":  [7],             # everyone knows
}

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
PROBES_PATH = "probes.json"


# ── Reconstruct persona assignments ────────────────────────────

def load_probes(probes_path=PROBES_PATH):
    """Load probes in the exact same order as run_experiment.py."""
    with open(probes_path) as f:
        data = json.load(f)
    probes = []
    for domain in PROBE_DOMAINS:
        if domain not in data:
            continue
        for item in data[domain]:
            probes.append({"id": item["id"], "domain": domain})
    return probes


def build_persona_lookup(probes, context_levels, filler_types, repeats=1):
    """
    Replay the exact same loop from build_tasks() in run_experiment.py
    to reconstruct which persona template was used for each
    (probe_id, context_level, filler_type) combination.

    Returns dict: (probe_id, context_level_rounded, filler_type) → persona_idx (0-7)
    """
    lookup = {}
    persona_idx = 0
    for rep in range(repeats):
        for level in context_levels:
            for ftype in filler_types:
                for probe in probes:
                    if probe["domain"] == "opinion":
                        continue  # opinion uses separate template, no persona_idx increment
                    pidx = persona_idx % 8
                    key = (probe["id"], round(level, 4), ftype)
                    lookup[key] = pidx
                    persona_idx += 1
    return lookup


def load_judged_results(results_dir=RESULTS_DIR):
    """Load all *_judged.jsonl files into a single DataFrame."""
    rows = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_judged.jsonl"):
            continue
        path = os.path.join(results_dir, fname)
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                rows.append(row)
    df = pd.DataFrame(rows)
    # Filter out failed/null responses
    df = df[df["is_sycophantic"] >= 0].copy()
    return df


# ── Analysis functions ──────────────────────────────────────────

def assign_personas(df, lookup):
    """Add persona_idx column by matching against reconstructed lookup."""
    def get_persona(row):
        key = (row["probe_id"], round(row["context_level"], 4), row["filler_type"])
        return lookup.get(key, -1)

    df["persona_idx"] = df.apply(get_persona, axis=1)
    # Drop opinion probes (they have no persona) and any unmatched
    df_persona = df[df["persona_idx"] >= 0].copy()
    df_persona["persona_label"] = df_persona["persona_idx"].map(PERSONA_LABELS)

    # Add authority group
    idx_to_group = {}
    for group, indices in AUTHORITY_GROUPS.items():
        for idx in indices:
            idx_to_group[idx] = group
    df_persona["authority_group"] = df_persona["persona_idx"].map(idx_to_group)

    return df_persona


def persona_rates_by_model(df):
    """Compute sycophancy rate per persona per model."""
    grouped = df.groupby(["model", "persona_label"]).agg(
        n=("is_sycophantic", "count"),
        sycophantic=("is_sycophantic", "sum"),
    ).reset_index()
    grouped["rate"] = grouped["sycophantic"] / grouped["n"]
    return grouped


def authority_rates_by_model(df):
    """Compute sycophancy rate per authority group per model."""
    grouped = df.groupby(["model", "authority_group"]).agg(
        n=("is_sycophantic", "count"),
        sycophantic=("is_sycophantic", "sum"),
    ).reset_index()
    grouped["rate"] = grouped["sycophantic"] / grouped["n"]
    return grouped


def chi2_persona_test(df):
    """Chi-squared test: is persona distribution of sycophancy non-uniform?"""
    results = {}
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        ct = pd.crosstab(mdf["persona_idx"], mdf["is_sycophantic"])
        if ct.shape[1] < 2:
            results[model] = {"chi2": 0, "p": 1.0, "dof": 0, "sig": False}
            continue
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        results[model] = {"chi2": chi2, "p": p, "dof": dof, "sig": p < 0.05}
    return results


def authority_group_test(df):
    """Chi-squared test on authority groups (4 groups)."""
    results = {}
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        ct = pd.crosstab(mdf["authority_group"], mdf["is_sycophantic"])
        if ct.shape[1] < 2:
            results[model] = {"chi2": 0, "p": 1.0, "dof": 0, "sig": False}
            continue
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        results[model] = {"chi2": chi2, "p": p, "dof": dof, "sig": p < 0.05}
    return results


def persona_x_context_interaction(df):
    """Test if persona effect varies by context level (high vs low split)."""
    results = {}
    df = df.copy()
    df["context_half"] = np.where(df["context_level"] <= 0.5, "low", "high")

    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        # Cochran-Mantel-Haenszel-style: compare persona effect across context halves
        rates = mdf.groupby(["context_half", "authority_group"]).agg(
            n=("is_sycophantic", "count"),
            syc=("is_sycophantic", "sum"),
        ).reset_index()
        rates["rate"] = rates["syc"] / rates["n"]

        # Compute range of rates by authority group in each half
        low_rates = rates[rates["context_half"] == "low"].set_index("authority_group")["rate"]
        high_rates = rates[rates["context_half"] == "high"].set_index("authority_group")["rate"]

        results[model] = {
            "low_context": low_rates.to_dict(),
            "high_context": high_rates.to_dict(),
            "low_range": low_rates.max() - low_rates.min() if len(low_rates) > 1 else 0,
            "high_range": high_rates.max() - high_rates.min() if len(high_rates) > 1 else 0,
        }
    return results


# ── Visualization ───────────────────────────────────────────────

MODEL_SHORT = {
    "qwen/qwen-2.5-7b-instruct": "Qwen 7B",
    "qwen/qwen-2.5-72b-instruct": "Qwen 72B",
    "mistralai/mistral-small-24b-instruct-2501": "Mistral 24B",
    "deepseek/deepseek-chat-v3.1": "DeepSeek V3.1",
    "google/gemma-3n-e4b-it": "Gemma 3N",
    "mistralai/mixtral-8x7b-instruct": "Mixtral 8x7B",
}


def plot_persona_heatmap(rates_df, outdir=FIGURES_DIR):
    """Heatmap: persona × model sycophancy rates."""
    pivot = rates_df.pivot_table(
        index="persona_label", columns="model", values="rate"
    )
    # Rename columns to short names
    pivot = pivot.rename(columns=MODEL_SHORT)

    # Order personas by index
    persona_order = [PERSONA_LABELS[i] for i in range(8)]
    pivot = pivot.reindex([p for p in persona_order if p in pivot.index])

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values * 100, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j] * 100
            color = "white" if val > 20 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Sycophancy rate (%)", shrink=0.8)
    ax.set_title("Sycophancy Rate by Persona Template × Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "persona_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_authority_bars(auth_df, outdir=FIGURES_DIR):
    """Grouped bar chart: authority group × model."""
    models = sorted(auth_df["model"].unique())
    groups = ["Low authority", "Credential claim", "External source", "Social pressure"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.2
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

    for i, group in enumerate(groups):
        gdata = auth_df[auth_df["authority_group"] == group]
        rates = []
        for m in models:
            r = gdata[gdata["model"] == m]["rate"]
            rates.append(r.values[0] * 100 if len(r) > 0 else 0)
        bars = ax.bar(x + i * width, rates, width, label=group, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, rates):
            if val > 1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    short_names = [MODEL_SHORT.get(m, m.split("/")[-1]) for m in models]
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(short_names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Sycophancy rate (%)", fontsize=11)
    ax.set_title("Sycophancy by Authority Level × Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, "persona_authority_bars.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_persona_by_context(df, outdir=FIGURES_DIR):
    """Line plot: sycophancy by context level, one line per authority group, per model."""
    models = sorted(df["model"].unique())
    groups = ["Low authority", "Credential claim", "External source", "Social pressure"]
    colors = {"Low authority": "#4CAF50", "Credential claim": "#2196F3",
              "External source": "#FF9800", "Social pressure": "#F44336"}

    ncols = 3
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), sharey=False)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        mdf = df[df["model"] == model]
        short = MODEL_SHORT.get(model, model.split("/")[-1])

        for group in groups:
            gdf = mdf[mdf["authority_group"] == group]
            by_level = gdf.groupby("context_level").agg(
                n=("is_sycophantic", "count"),
                syc=("is_sycophantic", "sum"),
            ).reset_index()
            by_level["rate"] = by_level["syc"] / by_level["n"] * 100
            ax.plot(by_level["context_level"] * 100, by_level["rate"],
                    marker="o", markersize=4, label=group, color=colors[group], alpha=0.8)

        ax.set_title(short, fontsize=11, fontweight="bold")
        ax.set_xlabel("Context fill (%)", fontsize=9)
        ax.set_ylabel("Sycophancy (%)", fontsize=9)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Persona Authority Effect Across Context Fill", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, "persona_context_interaction.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_persona_spread(rates_df, outdir=FIGURES_DIR):
    """Dot plot showing the range of persona effects per model."""
    models = sorted(rates_df["model"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        mdf = rates_df[rates_df["model"] == model]
        short = MODEL_SHORT.get(model, model.split("/")[-1])
        rates = mdf["rate"].values * 100
        labels = mdf["persona_label"].values

        # Sort by rate
        order = np.argsort(rates)
        rates = rates[order]
        labels = labels[order]

        # Plot range line + dots
        ax.plot([rates[0], rates[-1]], [i, i], color="gray", linewidth=2, alpha=0.5)
        scatter = ax.scatter(rates, [i] * len(rates), s=60, zorder=5, alpha=0.85)

        # Label min and max
        ax.annotate(labels[0], (rates[0], i), textcoords="offset points",
                    xytext=(-5, 10), fontsize=7, ha="right", color="green")
        ax.annotate(labels[-1], (rates[-1], i), textcoords="offset points",
                    xytext=(5, 10), fontsize=7, ha="left", color="red")

        ax.text(-2, i, short, ha="right", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks([])
    ax.set_xlabel("Sycophancy rate (%)", fontsize=11)
    ax.set_title("Persona Effect Spread per Model\n(lowest ↔ highest persona)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, "persona_spread.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("PERSONA ANALYSIS")
    print("=" * 60)

    # 1. Load probes and build persona lookup
    probes = load_probes()
    non_opinion = [p for p in probes if p["domain"] != "opinion"]
    print(f"\nProbes: {len(probes)} total, {len(non_opinion)} non-opinion (persona-assigned)")
    print(f"Persona templates: {len(PERSONA_LABELS)}")

    context_levels = [i / 10 for i in range(11)]  # 0.0 to 1.0 in 11 steps
    filler_types = ["neutral", "agreement", "correction"]

    lookup = build_persona_lookup(probes, context_levels, filler_types, repeats=1)
    print(f"Persona lookup entries: {len(lookup)}")

    # 2. Load results and assign personas
    print("\nLoading judged results...")
    df = load_judged_results()
    print(f"  Total valid results: {len(df)}")

    df_persona = assign_personas(df, lookup)
    unmatched = len(df[df["probe_domain"] != "opinion"]) - len(df_persona)
    print(f"  Persona-assigned results: {len(df_persona)} (opinion excluded, {unmatched} unmatched)")

    # Quick sanity: persona distribution should be roughly uniform
    dist = df_persona["persona_idx"].value_counts().sort_index()
    print(f"\n  Persona distribution (should be ~uniform):")
    for idx, count in dist.items():
        print(f"    {idx} ({PERSONA_LABELS[idx]}): {count}")

    # 3. Per-persona sycophancy rates
    print("\n" + "─" * 60)
    print("PER-PERSONA SYCOPHANCY RATES")
    print("─" * 60)

    rates = persona_rates_by_model(df_persona)

    for model in sorted(df_persona["model"].unique()):
        short = MODEL_SHORT.get(model, model.split("/")[-1])
        mrates = rates[rates["model"] == model].sort_values("rate", ascending=False)
        overall = df_persona[df_persona["model"] == model]["is_sycophantic"].mean()
        print(f"\n  {short} (overall: {overall:.1%})")
        for _, row in mrates.iterrows():
            delta = row["rate"] - overall
            print(f"    {row['persona_label']:25s} {row['rate']:6.1%}  ({delta:+.1%})  n={int(row['n'])}")

    # 4. Authority group analysis
    print("\n" + "─" * 60)
    print("AUTHORITY GROUP ANALYSIS")
    print("─" * 60)

    auth_rates = authority_rates_by_model(df_persona)

    for model in sorted(df_persona["model"].unique()):
        short = MODEL_SHORT.get(model, model.split("/")[-1])
        mrates = auth_rates[auth_rates["model"] == model].sort_values("rate", ascending=False)
        print(f"\n  {short}")
        for _, row in mrates.iterrows():
            print(f"    {row['authority_group']:20s} {row['rate']:6.1%}  n={int(row['n'])}")

    # 5. Statistical tests
    print("\n" + "─" * 60)
    print("STATISTICAL TESTS")
    print("─" * 60)

    print("\n  Chi-squared test: persona × sycophancy (8 personas)")
    chi2_results = chi2_persona_test(df_persona)
    for model in sorted(chi2_results):
        short = MODEL_SHORT.get(model, model.split("/")[-1])
        r = chi2_results[model]
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "ns"
        print(f"    {short:20s} χ²={r['chi2']:.2f}  p={r['p']:.4f}  {sig}")

    print("\n  Chi-squared test: authority group × sycophancy (4 groups)")
    auth_chi2 = authority_group_test(df_persona)
    for model in sorted(auth_chi2):
        short = MODEL_SHORT.get(model, model.split("/")[-1])
        r = auth_chi2[model]
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "ns"
        print(f"    {short:20s} χ²={r['chi2']:.2f}  p={r['p']:.4f}  {sig}")

    # 6. Persona × context interaction
    print("\n" + "─" * 60)
    print("PERSONA × CONTEXT INTERACTION")
    print("─" * 60)
    interaction = persona_x_context_interaction(df_persona)
    for model in sorted(interaction):
        short = MODEL_SHORT.get(model, model.split("/")[-1])
        r = interaction[model]
        print(f"\n  {short}")
        print(f"    Low context half  — authority spread: {r['low_range']:.1%}")
        print(f"    High context half — authority spread: {r['high_range']:.1%}")
        amplified = "YES" if r['high_range'] > r['low_range'] * 1.3 else "no"
        print(f"    Context amplifies persona effect? {amplified}")

    # 7. Generate figures
    print("\n" + "─" * 60)
    print("GENERATING FIGURES")
    print("─" * 60)
    plot_persona_heatmap(rates, FIGURES_DIR)
    plot_authority_bars(auth_rates, FIGURES_DIR)
    plot_persona_by_context(df_persona, FIGURES_DIR)
    plot_persona_spread(rates, FIGURES_DIR)

    # 8. Save results JSON
    report = {
        "persona_rates": {},
        "authority_rates": {},
        "chi2_persona": chi2_results,
        "chi2_authority": auth_chi2,
        "interaction": {},
    }
    for model in sorted(df_persona["model"].unique()):
        short = MODEL_SHORT.get(model, model.split("/")[-1])
        mrates = rates[rates["model"] == model]
        report["persona_rates"][short] = {
            row["persona_label"]: {"rate": round(row["rate"], 4), "n": int(row["n"])}
            for _, row in mrates.iterrows()
        }
        mauth = auth_rates[auth_rates["model"] == model]
        report["authority_rates"][short] = {
            row["authority_group"]: {"rate": round(row["rate"], 4), "n": int(row["n"])}
            for _, row in mauth.iterrows()
        }
        r = interaction[model]
        report["interaction"][short] = {
            "low_context_spread": round(r["low_range"], 4),
            "high_context_spread": round(r["high_range"], 4),
        }

    report_path = os.path.join(FIGURES_DIR, "persona_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved: {report_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
