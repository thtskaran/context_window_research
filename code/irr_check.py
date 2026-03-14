#!/usr/bin/env python3
"""
Inter-Rater Reliability Check

Samples a stratified subset from each model's judged results,
re-judges with a cheap second model (e.g., Haiku), and computes
Cohen's κ, percent agreement, and confusion matrix vs the primary
Sonnet 4.6 judge.

Usage:
    python irr_check.py --keys-file keys.txt --workers 40
    python irr_check.py --keys-file keys.txt --second-judge google/gemma-3-4b-it --workers 40
"""

import json
import os
import sys
import time
import asyncio
import argparse
import logging
import random
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Reuse judge infrastructure
from llm_judge import KeyPool, judge_single, load_probes_map

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
PROBES_PATH = "probes.json"
SAMPLE_PER_MODEL = 200  # 200 per model × 6 models = 1,200 total


def stratified_sample(judged_path: str, n: int = SAMPLE_PER_MODEL, seed: int = 42) -> list[dict]:
    """
    Stratified sample from a judged JSONL file.
    Ensures proportional representation of sycophantic vs honest labels.
    Excludes ambiguous results (is_sycophantic == -1).
    """
    rng = random.Random(seed)

    sycophantic = []
    honest = []
    with open(judged_path) as f:
        for line in f:
            r = json.loads(line)
            if r["is_sycophantic"] == 1:
                sycophantic.append(r)
            elif r["is_sycophantic"] == 0:
                honest.append(r)
            # skip ambiguous

    total_valid = len(sycophantic) + len(honest)
    if total_valid == 0:
        return []

    # Proportional allocation
    syc_ratio = len(sycophantic) / total_valid
    n_syc = max(1, round(n * syc_ratio))
    n_hon = n - n_syc

    # Cap at available
    n_syc = min(n_syc, len(sycophantic))
    n_hon = min(n_hon, len(honest))

    sample = rng.sample(sycophantic, n_syc) + rng.sample(honest, n_hon)
    rng.shuffle(sample)
    return sample


async def rejudge_sample(
    sample: list[dict],
    probes_map: dict,
    key_pool: KeyPool,
    judge_model: str,
    base_url: str,
    workers: int = 40,
) -> list[dict]:
    """Re-judge all sampled results with the second judge model."""
    import httpx

    sem = asyncio.Semaphore(workers)
    done = 0
    total = len(sample)

    async with httpx.AsyncClient() as client:

        async def process(r: dict):
            nonlocal done
            async with sem:
                probe_info = probes_map.get(r["probe_id"])
                if not probe_info:
                    r["second_judge_score"] = -1
                    r["second_judge_conf"] = 0.0
                    return

                domain = probe_info.get("domain", r.get("probe_domain", "factual"))
                score, conf = await judge_single(
                    client, base_url, key_pool, judge_model,
                    claim=probe_info["claim"],
                    truth=probe_info["truth"],
                    response=r["response"],
                    domain=domain,
                )
                r["second_judge_score"] = score
                r["second_judge_conf"] = conf
                r["second_judge_model"] = judge_model

                done += 1
                if done % 50 == 0 or done == total:
                    log.info(f"  Re-judged {done}/{total}")

        coros = [process(r) for r in sample]
        await asyncio.gather(*coros)

    return sample


def compute_cohens_kappa(labels_a: list[int], labels_b: list[int]) -> dict:
    """
    Compute Cohen's κ for two raters on binary labels.
    Only considers items where both raters gave 0 or 1 (excludes -1 ambiguous).
    """
    # Filter to pairs where both are 0 or 1
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a in (0, 1) and b in (0, 1)]
    if not pairs:
        return {"kappa": 0.0, "agreement": 0.0, "n": 0}

    a_arr = np.array([p[0] for p in pairs])
    b_arr = np.array([p[1] for p in pairs])
    n = len(pairs)

    # Observed agreement
    agree = np.sum(a_arr == b_arr)
    p_o = agree / n

    # Expected agreement by chance
    a1 = np.sum(a_arr == 1) / n
    a0 = np.sum(a_arr == 0) / n
    b1 = np.sum(b_arr == 1) / n
    b0 = np.sum(b_arr == 0) / n
    p_e = a1 * b1 + a0 * b0

    # Cohen's κ
    if p_e == 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1 - p_e)

    # Confusion matrix
    tp = int(np.sum((a_arr == 1) & (b_arr == 1)))
    tn = int(np.sum((a_arr == 0) & (b_arr == 0)))
    fp = int(np.sum((a_arr == 0) & (b_arr == 1)))  # Sonnet=honest, second=syc
    fn = int(np.sum((a_arr == 1) & (b_arr == 0)))  # Sonnet=syc, second=honest

    return {
        "kappa": round(kappa, 4),
        "agreement": round(p_o, 4),
        "n": n,
        "n_excluded_ambiguous": len(labels_a) - n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sonnet_syc_rate": round(np.mean(a_arr), 4),
        "second_syc_rate": round(np.mean(b_arr), 4),
    }


def interpret_kappa(k: float) -> str:
    if k < 0:     return "poor"
    if k < 0.20:  return "slight"
    if k < 0.40:  return "fair"
    if k < 0.60:  return "moderate"
    if k < 0.80:  return "substantial"
    return "almost perfect"


MODEL_SHORT = {
    "qwen/qwen-2.5-7b-instruct": "Qwen 7B",
    "qwen/qwen-2.5-72b-instruct": "Qwen 72B",
    "mistralai/mistral-small-24b-instruct-2501": "Mistral 24B",
    "deepseek/deepseek-chat-v3.1": "DeepSeek V3.1",
    "google/gemma-3n-e4b-it": "Gemma 3N",
    "mistralai/mixtral-8x7b-instruct": "Mixtral 8x7B",
}


def main():
    parser = argparse.ArgumentParser(description="Inter-Rater Reliability Check")
    parser.add_argument("--keys-file", default="keys.txt")
    parser.add_argument("--second-judge", default="anthropic/claude-3.5-haiku",
                        help="Cheap second judge model")
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--sample-per-model", type=int, default=SAMPLE_PER_MODEL)
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    key_pool = KeyPool.from_file(args.keys_file)
    probes_map = load_probes_map(PROBES_PATH)

    print("=" * 60)
    print("INTER-RATER RELIABILITY CHECK")
    print(f"Primary judge: anthropic/claude-sonnet-4-6")
    print(f"Second judge:  {args.second_judge}")
    print(f"Sample size:   {args.sample_per_model} per model")
    print("=" * 60)

    # 1. Sample from each model
    all_samples = []
    judged_files = sorted([
        f for f in os.listdir(RESULTS_DIR) if f.endswith("_judged.jsonl")
    ])

    print(f"\nFound {len(judged_files)} judged files")
    for fname in judged_files:
        path = os.path.join(RESULTS_DIR, fname)
        sample = stratified_sample(path, n=args.sample_per_model, seed=args.seed)
        print(f"  {fname}: sampled {len(sample)} trials")
        all_samples.extend(sample)

    print(f"\nTotal sample: {len(all_samples)} trials")

    # Check label distribution
    n_syc = sum(1 for r in all_samples if r["is_sycophantic"] == 1)
    n_hon = sum(1 for r in all_samples if r["is_sycophantic"] == 0)
    print(f"  Sonnet labels: {n_syc} sycophantic, {n_hon} honest")

    # 2. Re-judge with cheap model
    print(f"\nRe-judging with {args.second_judge}...")
    t0 = time.time()
    all_samples = asyncio.run(rejudge_sample(
        all_samples, probes_map, key_pool, args.second_judge, args.base_url,
        workers=args.workers,
    ))
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # 3. Compute metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Overall
    sonnet_labels = [r["is_sycophantic"] for r in all_samples]
    second_labels = [r["second_judge_score"] for r in all_samples]
    overall = compute_cohens_kappa(sonnet_labels, second_labels)

    print(f"\n  OVERALL (n={overall['n']})")
    print(f"    Cohen's κ = {overall['kappa']:.3f} ({interpret_kappa(overall['kappa'])})")
    print(f"    Agreement = {overall['agreement']:.1%}")
    print(f"    Sonnet syc rate: {overall['sonnet_syc_rate']:.1%}")
    print(f"    Second syc rate: {overall['second_syc_rate']:.1%}")
    print(f"    Ambiguous excluded: {overall['n_excluded_ambiguous']}")
    print(f"\n    Confusion matrix (rows=Sonnet, cols=Second):")
    print(f"                   Second=Honest  Second=Syc")
    print(f"      Sonnet=Honest    {overall['tn']:5d}          {overall['fp']:5d}")
    print(f"      Sonnet=Syc       {overall['fn']:5d}          {overall['tp']:5d}")

    # Per-model
    per_model = defaultdict(lambda: {"sonnet": [], "second": []})
    for r in all_samples:
        model = r["model"]
        per_model[model]["sonnet"].append(r["is_sycophantic"])
        per_model[model]["second"].append(r["second_judge_score"])

    print(f"\n  PER-MODEL BREAKDOWN:")
    model_results = {}
    for model in sorted(per_model):
        short = MODEL_SHORT.get(model, model.split("/")[-1])
        m = compute_cohens_kappa(per_model[model]["sonnet"], per_model[model]["second"])
        model_results[short] = m
        interp = interpret_kappa(m['kappa'])
        print(f"    {short:20s} κ={m['kappa']:.3f} ({interp:16s})  "
              f"agree={m['agreement']:.1%}  n={m['n']}  "
              f"[Sonnet {m['sonnet_syc_rate']:.1%} vs Second {m['second_syc_rate']:.1%}]")

    # Per-domain
    per_domain = defaultdict(lambda: {"sonnet": [], "second": []})
    for r in all_samples:
        domain = r.get("probe_domain", "unknown")
        per_domain[domain]["sonnet"].append(r["is_sycophantic"])
        per_domain[domain]["second"].append(r["second_judge_score"])

    print(f"\n  PER-DOMAIN BREAKDOWN:")
    domain_results = {}
    for domain in sorted(per_domain):
        d = compute_cohens_kappa(per_domain[domain]["sonnet"], per_domain[domain]["second"])
        domain_results[domain] = d
        interp = interpret_kappa(d['kappa'])
        print(f"    {domain:12s} κ={d['kappa']:.3f} ({interp:16s})  "
              f"agree={d['agreement']:.1%}  n={d['n']}")

    # Disagreement analysis
    print(f"\n  DISAGREEMENT ANALYSIS:")
    disagree = [r for r in all_samples
                if r["is_sycophantic"] in (0, 1)
                and r["second_judge_score"] in (0, 1)
                and r["is_sycophantic"] != r["second_judge_score"]]
    print(f"    Total disagreements: {len(disagree)}")
    if disagree:
        # Which direction?
        sonnet_syc_second_hon = sum(1 for r in disagree if r["is_sycophantic"] == 1)
        sonnet_hon_second_syc = sum(1 for r in disagree if r["is_sycophantic"] == 0)
        print(f"    Sonnet=Syc, Second=Honest: {sonnet_syc_second_hon} (Sonnet stricter)")
        print(f"    Sonnet=Honest, Second=Syc: {sonnet_hon_second_syc} (Second stricter)")

        # Show a few examples
        print(f"\n    Sample disagreements (up to 5):")
        for r in disagree[:5]:
            short = MODEL_SHORT.get(r['model'], r['model'].split('/')[-1])
            resp_preview = (r.get('response') or '')[:120].replace('\n', ' ')
            print(f"      [{short}] {r['probe_id']} domain={r['probe_domain']} "
                  f"Sonnet={r['is_sycophantic']} Second={r['second_judge_score']}")
            print(f"        \"{resp_preview}...\"")

    # 4. Save report
    report = {
        "primary_judge": "anthropic/claude-sonnet-4-6",
        "second_judge": args.second_judge,
        "sample_per_model": args.sample_per_model,
        "total_sample": len(all_samples),
        "overall": overall,
        "per_model": model_results,
        "per_domain": domain_results,
        "total_disagreements": len(disagree),
    }

    os.makedirs(FIGURES_DIR, exist_ok=True)
    report_path = os.path.join(FIGURES_DIR, "irr_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved: {report_path}")

    # Save raw sample for inspection
    sample_path = os.path.join(RESULTS_DIR, "irr_sample.jsonl")
    with open(sample_path, "w") as f:
        for r in all_samples:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved: {sample_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
