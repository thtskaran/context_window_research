#!/usr/bin/env python3
"""
Generate publication-quality diagrams for the fine-grained 0–10% experiment.

Produces:
  1. finegrained_step_function.png — Main figure: sycophancy rate vs context %
     for all 3 filler types, with 95% CI bands and changepoint annotation
  2. finegrained_filler_comparison.png — Side-by-side panels for each filler
     type with adjacent-step significance markers
  3. finegrained_domain_heatmap.png — Domain × context_pct heatmap (agreement filler)

Reads from: figures/finegrained_report.json
"""

import json
import os
import sys

# Set backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_report(path="figures/finegrained_report.json"):
    with open(path) as f:
        return json.load(f)


def fig1_step_function(report, outdir="figures"):
    """Main figure: all 3 filler types on one plot with CI bands."""
    rates = report["sycophancy_rates"]
    ci_data = report.get("confidence_intervals", {})

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"agreement": "#e74c3c", "neutral": "#95a5a6", "correction": "#27ae60"}
    labels = {"agreement": "Agreement Filler", "neutral": "Neutral Filler",
              "correction": "Correction Filler"}
    markers = {"agreement": "o", "neutral": "s", "correction": "^"}

    pcts = list(range(11))

    for filler in ["agreement", "neutral", "correction"]:
        y_vals = []
        ci_lo = []
        ci_hi = []
        for pct in pcts:
            info = rates.get(str(pct), {}).get(filler, {})
            ci = ci_data.get(str(pct), {}).get(filler, {})
            rate = info.get("rate", 0) if info else 0
            y_vals.append(rate * 100)
            ci_lo.append(ci.get("ci_lo", rate) * 100 if ci else rate * 100)
            ci_hi.append(ci.get("ci_hi", rate) * 100 if ci else rate * 100)

        y_arr = np.array(y_vals)
        lo_arr = np.array(ci_lo)
        hi_arr = np.array(ci_hi)

        ax.plot(pcts, y_arr, color=colors[filler], marker=markers[filler],
                linewidth=2.5, markersize=8, label=labels[filler], zorder=3)
        ax.fill_between(pcts, lo_arr, hi_arr, color=colors[filler],
                        alpha=0.15, zorder=1)

    # Annotate changepoint for agreement filler if phase transition detected
    cp = report.get("changepoint_analysis", {}).get("agreement", {})
    if cp.get("biggest_step"):
        bs = cp["biggest_step"]
        from_pct = bs["from_pct"]
        to_pct = bs["to_pct"]
        # Get the rate at from and to
        r_from = rates.get(str(from_pct), {}).get("agreement", {}).get("rate", 0) * 100
        r_to = rates.get(str(to_pct), {}).get("agreement", {}).get("rate", 0) * 100
        mid_y = (r_from + r_to) / 2
        mid_x = (from_pct + to_pct) / 2

        ax.annotate(
            f"Biggest step: {bs['delta_pp']}\n({from_pct}% → {to_pct}%)",
            xy=(mid_x, mid_y), xytext=(mid_x + 2.5, mid_y + 5),
            fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#e74c3c", alpha=0.9),
        )

    # Pattern annotation
    pattern = cp.get("pattern", "unknown")
    pattern_nice = {
        "phase_transition": "Phase Transition Detected",
        "smooth_gradient": "Smooth Gradient (No Phase Transition)",
        "flat": "Flat (No Effect)",
        "noisy_step": "Noisy Step Function",
    }.get(pattern, pattern)

    ax.text(0.98, 0.02, f"Pattern: {pattern_nice}",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

    ax.set_xlabel("Context Window Fill (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sycophancy Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Qwen 7B: Fine-Grained 0–10% Context Fill\n"
                 "Is the Step Function a Phase Transition?",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(pcts)
    ax.set_xticklabels([f"{p}%" for p in pcts])
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.3, 10.3)

    fig.tight_layout()
    out = os.path.join(outdir, "finegrained_step_function.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig2_filler_panels(report, outdir="figures"):
    """Three panels: one per filler type with significance markers."""
    rates = report["sycophancy_rates"]
    changepoints = report.get("changepoint_analysis", {})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    colors = {"agreement": "#e74c3c", "neutral": "#95a5a6", "correction": "#27ae60"}
    pcts = list(range(11))

    for ax, filler in zip(axes, ["agreement", "neutral", "correction"]):
        y_vals = []
        for pct in pcts:
            info = rates.get(str(pct), {}).get(filler, {})
            y_vals.append(info.get("rate", 0) * 100 if info else 0)

        ax.bar(pcts, y_vals, color=colors[filler], alpha=0.7, edgecolor="white",
               linewidth=0.5)
        ax.plot(pcts, y_vals, color=colors[filler], marker="o", linewidth=2,
                markersize=6, zorder=3)

        # Add significance markers between adjacent bars
        cp = changepoints.get(filler, {})
        if cp.get("all_steps"):
            for step in cp["all_steps"]:
                if step.get("significant_01"):
                    mid_x = (step["from_pct"] + step["to_pct"]) / 2
                    max_y = max(y_vals[step["from_pct"]], y_vals[step["to_pct"]])
                    ax.text(mid_x, max_y + 1.5, "***", ha="center", fontsize=8,
                            fontweight="bold", color="black")
                elif step.get("significant_05"):
                    mid_x = (step["from_pct"] + step["to_pct"]) / 2
                    max_y = max(y_vals[step["from_pct"]], y_vals[step["to_pct"]])
                    ax.text(mid_x, max_y + 1.5, "*", ha="center", fontsize=8,
                            fontweight="bold", color="black")

        ax.set_xlabel("Context Fill (%)", fontsize=10)
        ax.set_xticks(pcts)
        ax.set_xticklabels([f"{p}" for p in pcts], fontsize=8)
        ax.set_title(f"{filler.capitalize()} Filler", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        # Pattern label
        pattern = cp.get("pattern", "?")
        ax.text(0.98, 0.98, pattern.replace("_", " ").title(),
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.8))

    axes[0].set_ylabel("Sycophancy Rate (%)", fontsize=10, fontweight="bold")

    fig.suptitle("Qwen 7B: Step Function by Filler Type (* p<.05, *** p<.01)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(outdir, "finegrained_filler_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig3_domain_heatmap(report, outdir="figures"):
    """Heatmap: domain × context_pct for agreement filler only."""
    rates = report["sycophancy_rates"]
    pcts = list(range(11))

    # Get agreement filler rates per pct — then we need domain breakdown
    # Use domain_breakdown from the report
    domain_data = report.get("domain_breakdown", {})

    # Collect all domains
    all_domains = set()
    for pct_str, domains in domain_data.items():
        all_domains.update(domains.keys())
    domains_sorted = sorted(all_domains)

    if not domains_sorted:
        print("No domain data available, skipping heatmap")
        return

    # Build matrix
    matrix = np.zeros((len(domains_sorted), len(pcts)))
    for j, pct in enumerate(pcts):
        pct_data = domain_data.get(str(pct), {})
        for i, domain in enumerate(domains_sorted):
            info = pct_data.get(domain, {})
            matrix[i, j] = info.get("rate", 0) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0)

    # Annotate cells
    for i in range(len(domains_sorted)):
        for j in range(len(pcts)):
            val = matrix[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)

    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels([f"{p}%" for p in pcts])
    ax.set_yticks(range(len(domains_sorted)))
    ax.set_yticklabels([d.capitalize() for d in domains_sorted])
    ax.set_xlabel("Context Fill (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probe Domain", fontsize=11, fontweight="bold")
    ax.set_title("Qwen 7B: Sycophancy by Domain × Context Fill (All Fillers)",
                 fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Sycophancy Rate (%)", fontsize=10)

    fig.tight_layout()
    out = os.path.join(outdir, "finegrained_domain_heatmap.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    report = load_report()
    os.makedirs("figures", exist_ok=True)
    fig1_step_function(report)
    fig2_filler_panels(report)
    fig3_domain_heatmap(report)
    print("\nAll diagrams generated!")


if __name__ == "__main__":
    main()
