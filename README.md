# Context-Window Lock-In: Measuring How LLMs Break as Conversations Get Longer

Does sycophancy increase as an LLM's context window fills up? We test this across three 32K-context models totalling 33,751 valid trials: **Qwen 2.5 7B** (11,003), **Mistral Small 24B** (11,381), and **DeepSeek V3.1** (11,367). The context-length effect is weak and model-specific — only Qwen shows a meaningful jump. The universal finding across all three models is the **behavioral ratchet**: conversational pattern matters more than conversation length. Agreement filler roughly doubles sycophancy compared to correction filler (p < 10⁻¹⁴ in every model).

## Results Summary

All results scored by Claude Sonnet 4.6 as LLM judge with domain-aware rubrics.

### Cross-Model Comparison

| Metric | Qwen 2.5 7B | DeepSeek V3.1 | Mistral Small 24B |
|---|---|---|---|
| Parameters | 7B | ~37B active (MoE) | 24B |
| Trials (valid) | 11,003 | 11,367 | 11,381 |
| Overall sycophancy | 21.3% | 6.0% | 3.8% |
| Sycophancy at 0% context | 13.1% | 7.4% | 3.0% |
| Sycophancy at 100% context | 21.2% | 5.5% | 4.9% |
| Delta | **+8.1 pp** | **−1.8 pp** | **+1.9 pp** |
| Spearman ρ | 0.028 (p=0.004) | −0.008 (p=0.38) | 0.033 (p=0.0004) |
| Cohen's h | 0.215 (small) | −0.075 (negligible) | 0.100 (negligible) |
| Trend | Step at 0→10% | Flat | Flat (gradual drift) |
| GLMM context β (log-odds) | 0.447 | −0.217 | 0.903 |

### The Phase Diagram

![Phase Diagram](code/figures/phase_diagram.png)

Three very different profiles. Qwen jumps sharply from 0% → 10% context (~13% → 23%), then plateaus. DeepSeek sits flat around 5-7% with a slight downward drift. Mistral stays near 3-5% with minor upward drift at high utilization.

The context-length → sycophancy story is essentially a Qwen-specific finding. The other two models show no practically meaningful change across the full context range.

### Filler Type: The Universal Finding

The behavioral ratchet replicates across all three models with the same ordering: agreement > neutral > correction.

| Filler Type | Qwen 7B | DeepSeek V3.1 | Mistral 24B |
|---|---|---|---|
| Agreement | 25.3% | 8.6% | 5.6% |
| Neutral | 23.1% | 5.7% | 3.8% |
| Correction | 15.6% | 3.7% | 2.1% |
| Chi-squared p | < 10⁻²⁵ | < 10⁻¹⁸ | < 10⁻¹⁴ |

![Filler Comparison Qwen](code/figures/filler_comparison_qwen_qwen-2.5-7b-instruct.png)
![Filler Comparison DeepSeek](code/figures/filler_comparison_deepseek_deepseek-chat-v3.1.png)
![Filler Comparison Mistral](code/figures/filler_comparison_mistralai_mistral-small-24b-instruct-2501.png)

Correction filler is strongly protective in every model (GLMM β between −0.82 and −0.99). A conversation where the model has been correcting the user roughly halves the sycophancy rate compared to one where it's been agreeing. This is the paper's strongest result.

### Domain Breakdown

![Domain DeepSeek](code/figures/domain_breakdown_deepseek_deepseek-chat-v3.1.png)
![Domain Qwen](code/figures/domain_breakdown_qwen_qwen-2.5-7b-instruct.png)
![Domain Mistral](code/figures/domain_breakdown_mistralai_mistral-small-24b-instruct-2501.png)

All three models show the same broad hierarchy: Opinion and Logic are most vulnerable, Math and CS are most resistant.

| Domain | Qwen 7B | DeepSeek V3.1 | Mistral 24B |
|---|---|---|---|
| Opinion | ~33% | ~13% | 5.5% |
| Logic | ~29% | ~10% | 4.3% |
| Factual | ~27% | ~6% | 3.5% |
| Science | ~16% | ~6% | 3.6% |
| CS | ~14% | ~2% | 0.0% |
| Math | ~11% | ~1% | 4.0% |

### Heatmap

![Heatmap](code/figures/heatmap.png)

## How It Works

We construct synthetic conversations of varying length (0-100% of the 32K context window), then append a probe where the user states something wrong and asks the model to confirm. We measure agreement rate.

**Three filler types** isolate why sycophancy changes: neutral filler (cooking, travel) tests pure length effects; agreement filler (model agrees with user) tests behavioral momentum; correction filler (model corrects user) tests whether correction patterns protect.

**Six probe domains** (115 total probes): 100 factual probes with objectively wrong claims across factual, math, science, logic, and CS, plus 15 opinion probes with genuinely debatable claims. Factual probes scored on whether the model corrects the false claim. Opinion probes scored on whether the model engages critically vs blindly validates.

**Persona variation**: Each probe delivered through one of 8 rotating framings — from bare assertion to authority claims. This varies social pressure intensity as a covariate.

**Scoring**: All results scored by Claude Sonnet 4.6 as judge, with separate rubrics for factual vs opinion probes.

**Statistical model**: Bayesian binomial GLMM with probe_id as random intercept and logit link — proper for binary outcomes with clustering.

## Running a New Model

```bash
cd code/

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
│
└── code/
    ├── probes.json             # 115 probes (6 domains) + 8 persona templates
    ├── run_experiment.py       # Async experiment runner (30 workers)
    ├── llm_judge.py            # Domain-aware LLM judge (35 workers)
    ├── phase_diagram.py        # All figures: phase diagram, domain, filler, heatmap
    ├── statistical_tests.py    # Spearman, Mann-Whitney, chi-squared, GLMM
    ├── run_qwen.sh             # One-shot Qwen pipeline
    ├── run_mistral.sh          # One-shot Mistral Small 24B pipeline
    ├── run_deepseek.sh         # One-shot DeepSeek V3.1 pipeline
    ├── results/                # Raw + judged JSONL (33K+ results across 3 models)
    └── figures/                # Generated figures + stats_report.json
```

## Limitations

1. **Context-length effect is model-specific.** Only Qwen 7B shows a meaningful jump. DeepSeek and Mistral are flat. More models needed to understand what makes Qwen vulnerable (size? RLHF approach? architecture?).

2. **Qwen's step function.** The effect fires at 0→10% context rather than ramping linearly. More granular levels between 0-10% would clarify whether this is "presence of any history" vs "first N tokens of context."

3. **Template filler.** 10 template pairs per filler type. At 100% context, pairs repeat ~6x each. Real conversations have more complex dynamics.

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
