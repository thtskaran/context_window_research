# Context-Window Lock-In: Measuring How LLMs Break as Conversations Get Longer

As conversations fill an LLM's context window, sycophancy increases — but the magnitude depends heavily on the model. We measure this empirically across two 32K-context models totalling 22,384 valid trials: **Qwen 2.5 7B** (11,003 trials) and **Mistral Small 24B** (11,381 trials). Qwen shows a significant jump from 13.1% → 21.2% sycophancy as context fills. Mistral barely moves: 3.0% → 4.9%. Both models confirm the same finding: **conversational pattern matters more than raw length** — agreement filler nearly doubles sycophancy compared to correction filler.

## Results Summary

All results scored by Claude Sonnet 4.6 as LLM judge with domain-aware rubrics.

### Cross-Model Comparison

| Metric | Qwen 2.5 7B | Mistral Small 24B |
|---|---|---|
| Trials (valid) | 11,003 | 11,381 |
| Sycophancy at 0% context | 13.1% | 3.0% |
| Sycophancy at 100% context | 21.2% | 4.9% |
| Delta | **+8.1 pp** | **+1.9 pp** |
| Spearman ρ | 0.028 (p=0.004) | 0.033 (p=0.0004) |
| Cohen's h | 0.215 (small) | 0.100 (negligible) |
| Trend | Non-monotonic (step at 0→10%) | Flat (gradual drift) |
| GLMM context β (log-odds) | 0.447 | 0.903 |

### The Phase Diagram

![Phase Diagram](code/figures/phase_diagram.png)

Two very different curves on the same axis. Qwen jumps sharply from 0% → 10% context (~13% → 23%), then plateaus around 21-23%. Mistral stays nearly flat the entire way, drifting from 3% to 5% with most movement in the 70-100% range.

The Qwen curve looks like a step function — any conversation history triggers the effect. The Mistral curve looks like mild attention decay — only extreme context pressure produces a detectable (but practically negligible) shift.

### Filler Type: The Strongest Finding Across Both Models

**Qwen 2.5 7B:**

![Filler Comparison Qwen](code/figures/filler_comparison_qwen_qwen-2.5-7b-instruct.png)

| Filler Type | Rate | GLMM β |
|---|---|---|
| Agreement | 25.3% | +0.233 |
| Neutral | 23.1% | (baseline) |
| Correction | 15.6% | −0.818 |

**Mistral Small 24B:**

![Filler Comparison Mistral](code/figures/filler_comparison_mistralai_mistral-small-24b-instruct-2501.png)

| Filler Type | Rate | GLMM β |
|---|---|---|
| Agreement | 5.6% | +0.692 |
| Neutral | 3.8% | (baseline) |
| Correction | 2.1% | −0.987 |

Both models show the same ordering: agreement > neutral > correction, with highly significant chi-squared tests (Qwen p < 10⁻²⁵, Mistral p < 10⁻¹⁴). Correction filler is protective — a conversation where the model has been correcting the user makes it substantially less likely to be sycophantic. The "behavioral ratchet" replicates across models.

### Domain Breakdown

**Qwen 2.5 7B:**

![Domain Qwen](code/figures/domain_breakdown_qwen_qwen-2.5-7b-instruct.png)

| Domain | Rate |
|---|---|
| Opinion | ~33% |
| Logic | ~29% |
| Factual | ~27% |
| Science | ~16% |
| CS | ~14% |
| Math | ~11% |

**Mistral Small 24B:**

![Domain Mistral](code/figures/domain_breakdown_mistralai_mistral-small-24b-instruct-2501.png)

| Domain | Rate |
|---|---|
| Opinion | 5.5% |
| Logic | 4.3% |
| Math | 4.0% |
| Science | 3.6% |
| Factual | 3.5% |
| CS | 0.0% |

Both models show Opinion and Logic as the most sycophancy-prone domains. Mistral's CS domain is completely immune — zero sycophantic responses across 495 trials. Math is most resistant for Qwen; CS is most resistant for Mistral.

### Heatmap

![Heatmap](code/figures/heatmap.png)

## How It Works

We construct synthetic conversations of varying length (0-100% of the 32K context window), then append a probe where the user states something wrong and asks the model to confirm. We measure agreement rate.

**Three filler types** isolate why sycophancy changes: neutral filler (cooking, travel) tests pure length effects; agreement filler (model agrees with user) tests behavioral momentum; correction filler (model corrects user) tests whether correction patterns protect. If agreement >> correction, the cause is conversational pattern, not just length.

**Six probe domains** (115 total probes): 100 factual probes with objectively wrong claims across factual, math, science, logic, and CS, plus 15 opinion probes with genuinely debatable claims. Factual probes are scored on whether the model corrects the false claim. Opinion probes are scored on whether the model engages critically vs blindly validates.

**Persona variation**: Each probe is delivered through one of 8 rotating framings — from bare assertion ("I'm pretty sure that X") to authority claims ("I have a PhD in this field"). This varies social pressure intensity as a covariate.

**Scoring**: All results scored by Claude Sonnet 4.6 as judge, with separate rubrics for factual vs opinion probes. Ambiguity rates: 0.3% (Qwen), 0.04% (Mistral).

**Statistical model**: Generalized linear mixed model (Bayesian binomial with logit link) with probe_id as random intercept — proper for binary outcomes with clustering.

## Running a New Model

```bash
cd code/

# Edit run_mistral.sh with your model details, or run manually:
python run_experiment.py --mode api --model <openrouter-model-id> \
  --keys-file keys.txt --max-tokens 32768 --workers 30 --repeats 3

python llm_judge.py --input results/<model_slug>_results.jsonl \
  --output results/<model_slug>_judged.jsonl \
  --judge-model anthropic/claude-sonnet-4-6 --keys-file keys.txt --workers 35

python phase_diagram.py --results-dir results/ --output-dir figures/
python statistical_tests.py --results-dir results/ --output figures/stats_report.json --verbose
```

Create `code/keys.txt` with one OpenRouter API key per line.

## Repo Structure

```
├── README.md                   # This file
├── experiment-protocol.md      # Full experimental protocol
├── research-note.md            # Findings and interpretation so far
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
    ├── statistical_tests.py    # Spearman, Mann-Whitney, chi-squared, GLMM
    ├── run_qwen.sh             # One-shot Qwen pipeline
    ├── run_mistral.sh          # One-shot Mistral Small 24B pipeline
    ├── results/                # Raw + judged JSONL (22K+ results across 2 models)
    └── figures/                # Generated figures + stats_report.json
```

## Limitations

1. **Two models so far.** The filler-type effect replicates, but the context-length effect varies dramatically. More models needed to understand what drives the difference (model size? training data? RLHF approach?).

2. **Qwen's step function.** The effect mostly fires at 0→10% context rather than ramping linearly. More granular levels between 0-10% would clarify whether this is "presence of any history" vs "first N tokens of context."

3. **Template filler.** 10 template pairs per filler type (expanded from 5). At 100% context, pairs repeat ~6x each. Real conversations have more complex dynamics.

4. **Opinion probe subjectivity.** Judging sycophancy on debatable claims is inherently noisier than on factual claims.

5. **Single judge model.** All scoring by Sonnet 4.6. Inter-rater reliability with a second judge not yet measured.

## Citation

```bibtex
@misc{prasad2026contextlock,
  title={Context-Window Lock-In and Silent Degradation in Large Language Models},
  author={Prasad, Karan},
  year={2026},
  note={Preprint, Obvix Labs}
}
```
