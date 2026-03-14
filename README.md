# Context-Window Lock-In: Measuring How LLMs Break as Conversations Get Longer

As conversations fill an LLM's context window, sycophancy increases. We measure this empirically: 11,003 trials on Qwen 2.5 7B show a significant jump from 13.1% sycophancy at empty context to 21.2% at full context (p<0.001). The effect is driven more by conversational pattern than raw length — a history of agreement nearly doubles sycophancy compared to a history of corrections.

## Results (11,003 trials)

**Model:** Qwen 2.5 7B Instruct (32K context, fully stressed). All results scored by Claude Sonnet 4.6 judge.

| Metric | Value |
|---|---|
| Sycophancy at 0% context | 13.1% |
| Sycophancy at 100% context | 21.2% |
| Delta | **+8.1 pp** |
| Spearman ρ | 0.028 (p=0.0035) |
| Mixed-effects context β | 0.043 (p<0.0001) |
| Cohen's h | 0.215 (small) |

### The Phase Diagram

![Phase Diagram](code/figures/phase_diagram.png)

Sycophancy jumps sharply from 0% → 10% context (~13% → 23%), then plateaus around 21-23% across higher utilization levels. The effect isn't a gradual ramp — it's more like a step function triggered once any meaningful context is present. The 0% baseline (empty context, just the probe) is where the model performs best.

### Filler Type Is the Strongest Finding

![Filler Comparison](code/figures/filler_comparison_qwen_qwen-2.5-7b-instruct.png)

| Filler Type | Sycophancy Rate | What It Means |
|---|---|---|
| Agreement | 25.3% | Prior yes-man behavior amplifies future sycophancy |
| Neutral | 23.1% | Baseline context-length effect |
| Correction | 15.6% | Active correction history builds resistance |

Chi-squared p < 10⁻²⁵. Correction filler is strongly protective (β=−0.075, p < 10⁻²⁵) — a conversation where the model has been correcting the user makes it ~40% less likely to be sycophantic than one where it's been agreeing. This confirms the "behavioral ratchet" from the preprint.

### Domain Breakdown

![Domain Breakdown](code/figures/domain_breakdown_qwen_qwen-2.5-7b-instruct.png)

Clear hierarchy of vulnerability:

| Domain | Avg Sycophancy | Interpretation |
|---|---|---|
| Opinion | ~33% | Model readily validates debatable claims without pushback |
| Logic | ~29% | Formal reasoning errors hard to catch under context pressure |
| Factual | ~27% | Common misconceptions — model's weakest factual ground |
| Science | ~16% | Better grounding in scientific knowledge |
| CS | ~14% | Strong on computing fundamentals |
| Math | ~11% | Most resistant — formal math knowledge is robust |

Opinion probes are the most sycophancy-prone: the model validates debatable claims ~33% of the time without raising counterarguments. Math probes are most resistant — the model rarely agrees with mathematical falsehoods even under full context pressure.

### Heatmap

![Heatmap](code/figures/heatmap.png)

Visual summary: 0% context is distinctly green (13%), everything else is a lighter shade (21-24%). The step from empty to any context is the main event.

## How It Works

We construct synthetic conversations of varying length (0-100% of the 32K context window), then append a probe where the user states something wrong and asks the model to confirm. We measure agreement rate.

**Three filler types** isolate why sycophancy changes: neutral filler (cooking, travel) tests pure length effects; agreement filler (model agrees with user) tests behavioral momentum; correction filler (model corrects user) tests whether correction patterns protect. If agreement >> correction, the cause is conversational pattern, not just length.

**Six probe domains** (115 total probes): 100 factual probes with objectively wrong claims across factual, math, science, logic, and CS, plus 15 opinion probes with genuinely debatable claims. Factual probes are scored on whether the model corrects the false claim. Opinion probes are scored on whether the model engages critically vs blindly validates.

**Persona variation**: Each probe is delivered through one of 8 rotating framings — from bare assertion ("I'm pretty sure that X") to authority claims ("I have a PhD in this field and I'm confident that X"). This varies social pressure intensity rather than holding it constant.

**Scoring**: All 11,003 results scored by Claude Sonnet 4.6 as judge, with domain-aware rubrics. Only 37 results (0.3%) were ambiguous.

## Repo Structure

```
├── README.md                   # This file
├── experiment-protocol.md      # Full experimental protocol
├── research-notes/             # Background literature review and analysis
│   ├── architecture-deep-dive.md
│   ├── deep-dive-research-notes.md
│   ├── mitigation-architectures-analysis.md
│   ├── deep-dive-draft-v2.pdf
│   └── old-research-incomplete.pdf
│
└── code/
    ├── probes.json             # 115 probes (6 domains) + 8 persona templates
    ├── run_experiment.py       # Async experiment runner (30 workers)
    ├── llm_judge.py            # Domain-aware LLM judge (35 workers)
    ├── phase_diagram.py        # All figures: phase diagram, domain, filler, heatmap
    ├── statistical_tests.py    # Spearman, Mann-Whitney, chi-squared, mixed-effects
    ├── run_qwen.sh             # One-shot pipeline: experiment → judge → figures → stats
    ├── run_post_analysis.sh    # Re-run judge + analysis on existing results
    ├── results/                # Raw + judged JSONL (11K+ results)
    └── figures/                # Generated figures + stats_report.json
```

## Quick Start

```bash
cd code/
pip install httpx numpy matplotlib scipy statsmodels pandas
bash run_qwen.sh
```

Or manually:

```bash
python run_experiment.py --mode api --model qwen/qwen-2.5-7b-instruct \
  --keys-file keys.txt --max-tokens 32768 --workers 30 --repeats 3

python llm_judge.py --input results/qwen_qwen-2.5-7b-instruct_results.jsonl \
  --output results/qwen_qwen-2.5-7b-instruct_judged.jsonl \
  --judge-model anthropic/claude-sonnet-4-6 --keys-file keys.txt --workers 35

python phase_diagram.py --results-dir results/ --output-dir figures/
python statistical_tests.py --results-dir results/ --output figures/stats_report.json --verbose
```

Create `code/keys.txt` with one OpenRouter API key per line.

## Limitations

1. **Single model.** The Qwen result is strong but we need more models at their context limits to generalize.

2. **Step function, not ramp.** The effect mostly fires at 0→10% context rather than increasing linearly. This could mean the critical variable is "presence of any conversation history" rather than "amount of context." More granular levels between 0-10% would clarify.

3. **Template filler.** Five template pairs per filler type. Real conversations have more complex dynamics.

4. **Opinion probe subjectivity.** Judging sycophancy on debatable claims is noisier than on factual claims. The 33% opinion sycophancy rate may partly reflect the rubric's sensitivity.

## Citation

```bibtex
@misc{prasad2026contextlock,
  title={Context-Window Lock-In and Silent Degradation in Large Language Models},
  author={Prasad, Karan},
  year={2026},
  note={Preprint, Obvix Labs}
}
```
