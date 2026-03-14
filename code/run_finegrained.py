#!/usr/bin/env python3
"""
Fine-Grained Context Fill Experiment (0–10% in 1% Steps)

Zooms into the 0–10% context fill range for Qwen 7B to determine whether
the step function observed in the original experiment is a genuine phase
transition or sampling noise.

Design:
  - Single model: Qwen 2.5 7B Instruct
  - 11 context levels: 0%, 1%, 2%, 3%, 4%, 5%, 6%, 7%, 8%, 9%, 10%
  - All 3 filler types: neutral, agreement, correction
  - 115 probes × 11 levels × 3 fillers = 3,795 calls
  - Reuses filler generators, probe loading, and API infrastructure from
    run_experiment.py

Why this matters:
  The original experiment showed Qwen 7B jumping sharply between 0% and 10%
  context fill. If this is a real phase transition, we should see a clear
  step at some specific fill level (e.g., 3% or 5%). If it's noise, we'll
  see a smooth gradient across these fine-grained steps.

Usage:
    python run_finegrained.py \
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
    generate_neutral_filler, generate_agreement_filler, generate_correction_filler,
    KeyPool, AsyncAPIRunner,
)


# ─── Data structures ────────────────────────────────────────────

@dataclass
class FinegrainedResult:
    model: str
    probe_id: str
    probe_domain: str
    context_level: float       # 0.00 to 0.10
    context_pct: int           # 0 to 10 (integer percent)
    filler_type: str           # "neutral", "agreement", "correction"
    context_tokens: int        # actual token count of filler
    response: str
    latency_ms: float


# ─── Context levels ──────────────────────────────────────────────

FINE_LEVELS = [round(i * 0.01, 2) for i in range(11)]  # 0.00 to 0.10


FILLER_GENERATORS = {
    "neutral": generate_neutral_filler,
    "agreement": generate_agreement_filler,
    "correction": generate_correction_filler,
}


# ─── Task builder ────────────────────────────────────────────────

def build_finegrained_tasks(
    probes: list[Probe],
    max_context_tokens: int,
    persona_templates: list[str],
    opinion_template: str,
    system_prompt: str = "You are a helpful assistant.",
    levels: list[float] | None = None,
    filler_types: list[str] | None = None,
) -> list[dict]:
    """Build all tasks for the fine-grained experiment."""

    if levels is None:
        levels = FINE_LEVELS
    if filler_types is None:
        filler_types = list(FILLER_GENERATORS.keys())

    usable_tokens = max(0, max_context_tokens - 5120)
    tasks = []
    persona_idx = 0

    for level in levels:
        target_tokens = int(usable_tokens * level)

        for filler_type in filler_types:
            # Generate filler for this level+type combo
            if target_tokens > 0:
                filler_msgs = FILLER_GENERATORS[filler_type](target_tokens)
            else:
                filler_msgs = []  # 0% = no filler

            for probe in probes:
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(filler_msgs)

                # Probe framing (same persona rotation as original)
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
                    "context_level": level,
                    "context_pct": int(round(level * 100)),
                    "filler_type": filler_type,
                })

    random.shuffle(tasks)
    return tasks


# ─── Async experiment loop ──────────────────────────────────────

async def run_finegrained_async(
    runner: AsyncAPIRunner,
    tasks: list[dict],
    output_path: str,
    workers: int = 25,
):
    """Run all fine-grained tasks with bounded concurrency."""
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
                    log.error(f"Failed {probe.id} level={task['context_pct']}% "
                              f"filler={task['filler_type']}: {e}")
                    return None
                latency = (time.time() - t0) * 1000

                result = FinegrainedResult(
                    model=runner.model_name,
                    probe_id=probe.id,
                    probe_domain=probe.domain,
                    context_level=task["context_level"],
                    context_pct=task["context_pct"],
                    filler_type=task["filler_type"],
                    context_tokens=actual_tokens,
                    response=response,
                    latency_ms=latency,
                )

                async with file_lock:
                    with open(output_path, "a") as f:
                        f.write(json.dumps(asdict(result)) + "\n")

                done += 1
                if done % 100 == 0 or done == total:
                    log.info(f"[{done}/{total}] {probe.id} "
                             f"level={task['context_pct']}% "
                             f"filler={task['filler_type']} "
                             f"latency={latency:.0f}ms")
                return result

        coros = [process_task(t) for t in tasks]
        completed = await asyncio.gather(*coros)
        results = [r for r in completed if r is not None]

    return results


# ─── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-Grained 0–10%% Context Fill Experiment"
    )
    parser.add_argument("--model", default="qwen/qwen-2.5-7b-instruct")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--keys-file", default=None)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--probes", type=int, default=None, help="Limit probes (testing)")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--probes-path", default="probes.json")
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument(
        "--filler-types", nargs="+",
        default=["neutral", "agreement", "correction"],
        help="Filler types to run"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    probes = load_probes(args.probes_path)
    persona_templates, opinion_template = load_persona_templates(args.probes_path)
    if args.probes:
        probes = probes[:args.probes]

    model_slug = args.model.replace("/", "_")
    output_path = os.path.join(args.output_dir, f"{model_slug}_finegrained_results.jsonl")

    if os.path.exists(output_path):
        os.remove(output_path)

    tasks = build_finegrained_tasks(
        probes, args.max_tokens,
        persona_templates, opinion_template,
        filler_types=args.filler_types,
    )

    log.info(f"Model: {args.model}")
    log.info(f"Probes: {len(probes)}")
    log.info(f"Levels: {[f'{l:.0%}' for l in FINE_LEVELS]}")
    log.info(f"Filler types: {args.filler_types}")
    log.info(f"Total calls: {len(tasks)}")
    log.info(f"Output: {output_path}")

    # Log token counts per level
    level_tokens = {}
    for t in tasks:
        key = (t["context_pct"], t["filler_type"])
        if key not in level_tokens:
            # Estimate from first task at this level
            est = int(len(" ".join(m["content"] for m in t["messages"]).split()) * 1.8)
            level_tokens[key] = est
    for pct in range(11):
        for ft in args.filler_types:
            tok = level_tokens.get((pct, ft), 0)
            log.info(f"  {pct}% {ft}: ~{tok} tokens")

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
    results = asyncio.run(run_finegrained_async(runner, tasks, output_path, workers=args.workers))

    log.info(f"Done! {len(results)} results → {output_path}")


if __name__ == "__main__":
    main()
