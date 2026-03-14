#!/usr/bin/env python3
"""
Correction Injection Mitigation Experiment

Tests whether injecting correction exchanges after agreement filler
can "reset" the behavioral ratchet.

Design:
  - Fixed context level: 50% fill
  - Conditions:
      agree_only:     50% agreement filler → probe  (baseline, high sycophancy)
      correct_only:   50% correction filler → probe  (baseline, low sycophancy)
      inject_1:       ~48.5% agreement + 1 correction exchange → probe
      inject_3:       ~45.5% agreement + 3 correction exchanges → probe
      inject_5:       ~42.5% agreement + 5 correction exchanges → probe
      inject_10:      ~35% agreement + 10 correction exchanges → probe
  - All conditions keep total filler at 50% context to control for length.
  - 115 probes × 8 personas × 6 models × 6 conditions = 33,120 calls
    (but we run one model at a time like the original experiment)

Usage:
    python run_correction_injection.py \
      --mode api \
      --model qwen/qwen-2.5-7b-instruct \
      --keys-file keys.txt \
      --max-tokens 32768 \
      --workers 25
"""

import json
import os
import time
import asyncio
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# Import shared components from the main experiment
from run_experiment import (
    Probe, load_probes, load_persona_templates,
    generate_agreement_filler, generate_correction_filler,
    KeyPool, AsyncAPIRunner,
)


# ─── Data structures ────────────────────────────────────────────

@dataclass
class InjectionResult:
    model: str
    probe_id: str
    probe_domain: str
    condition: str              # "agree_only", "correct_only", "inject_1", etc.
    injection_count: int        # 0, 1, 3, 5, 10 correction exchanges injected
    context_tokens: int         # actual token count of filler
    agree_tokens: int           # tokens from agreement portion
    correct_tokens: int         # tokens from correction portion
    response: str
    latency_ms: float


# ─── Hybrid filler generation ────────────────────────────────────

def generate_hybrid_filler(
    total_target_tokens: int,
    correction_exchanges: int,
) -> tuple[list[dict], int, int]:
    """
    Generate agreement filler followed by N correction exchanges,
    with total length ≈ total_target_tokens.

    Returns (messages, agree_token_est, correct_token_est).
    """
    if correction_exchanges == 0:
        msgs = generate_agreement_filler(total_target_tokens)
        est = int(sum(len((m["content"]).split()) for m in msgs) * 1.8)
        return msgs, est, 0

    # First, generate a full block of correction filler to measure per-exchange size.
    # Each correction exchange is ~2 messages (user + assistant).
    # From the templates: avg ~85 words/exchange → ~153 tokens/exchange
    correction_full = generate_correction_filler(100000)  # generate excess
    # Take exactly N exchanges (2 messages each)
    correction_msgs = correction_full[:correction_exchanges * 2]
    correct_token_est = int(sum(len(m["content"].split()) for m in correction_msgs) * 1.8)

    # Fill remaining budget with agreement
    agree_target = max(0, total_target_tokens - correct_token_est)
    if agree_target > 100:
        agreement_msgs = generate_agreement_filler(agree_target)
    else:
        agreement_msgs = []
    agree_token_est = int(sum(len(m["content"].split()) for m in agreement_msgs) * 1.8)

    # Combine: agreement first, then correction (recency matters)
    combined = agreement_msgs + correction_msgs
    return combined, agree_token_est, correct_token_est


# ─── Task builder ───────────────────────────────────────────────

CONDITIONS = [
    ("agree_only",   0),
    ("inject_1",     1),
    ("inject_3",     3),
    ("inject_5",     5),
    ("inject_10",   10),
    ("correct_only", -1),  # -1 = pure correction (handled separately)
]

def build_injection_tasks(
    probes: list[Probe],
    max_context_tokens: int,
    persona_templates: list[str],
    opinion_template: str,
    system_prompt: str = "You are a helpful assistant.",
    conditions: list[tuple[str, int]] | None = None,
) -> list[dict]:
    """Build all tasks for the correction injection experiment."""

    if conditions is None:
        conditions = CONDITIONS

    usable_tokens = max(0, max_context_tokens - 5120)
    target_tokens = int(usable_tokens * 0.5)  # Fixed at 50% context

    tasks = []
    persona_idx = 0

    for cond_name, inject_count in conditions:
        # Pre-generate filler for this condition (shared across probes)
        if cond_name == "correct_only":
            filler_msgs = generate_correction_filler(target_tokens)
            agree_est = 0
            correct_est = int(sum(len(m["content"].split()) for m in filler_msgs) * 1.8)
        elif inject_count == 0:
            filler_msgs = generate_agreement_filler(target_tokens)
            agree_est = int(sum(len(m["content"].split()) for m in filler_msgs) * 1.8)
            correct_est = 0
        else:
            filler_msgs, agree_est, correct_est = generate_hybrid_filler(
                target_tokens, inject_count
            )

        for probe in probes:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(filler_msgs)

            # Probe framing (same persona rotation as original experiment)
            if probe.domain == "opinion":
                probe_text = opinion_template.format(claim=probe.claim)
            else:
                template = persona_templates[persona_idx % len(persona_templates)]
                probe_text = template.format(claim=probe.claim)
                persona_idx += 1

            messages.append({"role": "user", "content": probe_text})

            tasks.append({
                "messages": messages,
                "probe": probe,
                "condition": cond_name,
                "inject_count": inject_count if inject_count >= 0 else 0,
                "agree_est": agree_est,
                "correct_est": correct_est,
            })

    random.shuffle(tasks)
    return tasks


# ─── Async experiment loop ──────────────────────────────────────

async def run_injection_async(
    runner: AsyncAPIRunner,
    tasks: list[dict],
    output_path: str,
    workers: int = 25,
):
    """Run all injection tasks with bounded concurrency."""
    import httpx

    sem = asyncio.Semaphore(workers)
    done = 0
    total = len(tasks)
    results = []
    file_lock = asyncio.Lock()

    async with httpx.AsyncClient() as client:

        async def process_task(task):
            nonlocal done
            async with sem:
                messages = task["messages"]
                probe = task["probe"]
                actual_tokens = runner.count_tokens(messages)

                t0 = time.time()
                try:
                    response = await runner.generate(messages, client)
                except Exception as e:
                    log.error(f"Failed {probe.id} cond={task['condition']}: {e}")
                    return None
                latency = (time.time() - t0) * 1000

                result = InjectionResult(
                    model=runner.model_name,
                    probe_id=probe.id,
                    probe_domain=probe.domain,
                    condition=task["condition"],
                    injection_count=task["inject_count"],
                    context_tokens=actual_tokens,
                    agree_tokens=task["agree_est"],
                    correct_tokens=task["correct_est"],
                    response=response,
                    latency_ms=latency,
                )

                async with file_lock:
                    with open(output_path, "a") as f:
                        f.write(json.dumps(asdict(result)) + "\n")

                done += 1
                if done % 50 == 0 or done == total:
                    log.info(f"[{done}/{total}] {probe.id} cond={task['condition']} "
                             f"latency={latency:.0f}ms")
                return result

        coros = [process_task(t) for t in tasks]
        completed = await asyncio.gather(*coros)
        results = [r for r in completed if r is not None]

    return results


# ─── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Correction Injection Mitigation Experiment"
    )
    parser.add_argument("--mode", choices=["api"], default="api")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--keys-file", default=None)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--probes", type=int, default=None, help="Limit probes (testing)")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--probes-path", default="probes.json")
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument(
        "--conditions", nargs="+",
        default=["agree_only", "inject_1", "inject_3", "inject_5", "inject_10", "correct_only"],
        help="Conditions to run"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    probes = load_probes(args.probes_path)
    persona_templates, opinion_template = load_persona_templates(args.probes_path)
    if args.probes:
        probes = probes[:args.probes]

    # Filter conditions
    cond_map = {name: count for name, count in CONDITIONS}
    conditions = [(c, cond_map[c]) for c in args.conditions if c in cond_map]

    model_slug = args.model.replace("/", "_")
    output_path = os.path.join(args.output_dir, f"{model_slug}_injection_results.jsonl")

    if os.path.exists(output_path):
        os.remove(output_path)

    log.info(f"Model: {args.model}")
    log.info(f"Probes: {len(probes)}")
    log.info(f"Conditions: {[c[0] for c in conditions]}")
    log.info(f"Total calls: {len(probes) * len(conditions)}")
    log.info(f"Output: {output_path}")

    tasks = build_injection_tasks(
        probes, args.max_tokens,
        persona_templates, opinion_template,
        conditions=conditions,
    )
    log.info(f"Built {len(tasks)} tasks")

    # Set up API runner
    if args.keys_file:
        key_pool = KeyPool.from_file(args.keys_file)
    elif args.api_key:
        key_pool = KeyPool.from_single(args.api_key)
    elif os.environ.get("OPENROUTER_API_KEY"):
        key_pool = KeyPool.from_single(os.environ["OPENROUTER_API_KEY"])
    else:
        raise ValueError("API key required: --keys-file, --api-key, or OPENROUTER_API_KEY")

    runner = AsyncAPIRunner(args.model, key_pool)
    log.info(f"Workers: {args.workers}")
    results = asyncio.run(run_injection_async(runner, tasks, output_path, workers=args.workers))

    log.info(f"Done! {len(results)} results → {output_path}")


if __name__ == "__main__":
    main()
