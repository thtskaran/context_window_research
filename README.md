# Context-Window Lock-In: Measuring How LLMs Break as Conversations Get Longer

Does sycophancy increase as an LLM's context window fills up? We test this across five 32K-context models totalling 56,377 valid trials: **Gemma 3N E4B** (11,245), **Qwen 2.5 7B** (11,003), **Qwen 2.5 72B** (11,381), **DeepSeek V3.1** (11,367), and **Mistral Small 24B** (11,381). The context-length effect scales inversely with model size — small models (Gemma ~4B, Qwen 7B) show clear degradation, large models (24B+) are flat. The universal finding across all five models is the **behavioral ratchet**: conversational pattern matters more than conversation length. Agreement filler roughly doubles sycophancy compared to correction filler (p < 10⁻¹⁴ in every model).

## Results Summary

All results scored by Claude Sonnet 4.6 as LLM judge with domain-aware rubrics.

### Cross-Model Comparison

| Metric | Gemma 3N (~4B) | Qwen 7B | Qwen 72B | DeepSeek V3.1 (~37B) | Mistral 24B |
|---|---|---|---|---|---|
| Trials (valid) | 11,245 | 11,003 | 11,381 | 11,367 | 11,381 |
| Overall sycophancy | 34.2% | 21.3% | 6.7% | 6.0% | 3.8% |
| Sycophancy at 0% context | 27.7% | 13.1% | 4.6% | 7.4% | 3.0% |
| Sycophancy at 100% context | 38.4% | 21.2% | 8.0% | 5.5% | 4.9% |
| Delta | **+10.7 pp** | **+8.1 pp** | **+3.4 pp** | **−1.8 pp** | **+1.9 pp** |
| Spearman ρ | 0.077 (p<10⁻¹⁵) | 0.028 (p=0.004) | 0.035 (p=0.0002) | −0.008 (p=0.38) | 0.033 (p=0.0004) |
| Cohen's h | 0.229 (small) | 0.215 (small) | 0.140 (negligible) | −0.075 (negligible) | 0.100 (negligible) |
| Trend | Gradual ramp | Step at 0→10% | Gradual ramp | Flat | Flat |
| GLMM context β | 1.040 | 0.447 | 1.012 | −0.217 | 0.903 |

### The Phase Diagram

![Phase Diagram](code/figures/phase_diagram.png)

Clear size-dependent stratification. The two smallest models (Gemma ~4B and Qwen 7B) sit at the top with high baselines and visible context-length effects. The three larger models (24B+) cluster at the bottom, nearly flat. Gemma shows the cleanest gradual ramp of any model — sycophancy increases steadily from 28% to 38% across the full context range.

### Filler Type: The Universal Finding

The behavioral ratchet replicates across all five models: agreement > neutral > correction.

| Filler Type | Gemma 3N | Qwen 7B | Qwen 72B | DeepSeek V3.1 | Mistral 24B |
|---|---|---|---|---|---|
| Agreement | 41.2% | 25.3% | 10.2% | 8.6% | 5.6% |
| Neutral | 36.2% | 23.1% | 5.8% | 5.7% | 3.8% |
| Correction | 25.1% | 15.6% | 4.2% | 3.7% | 2.1% |
| Chi-squared p | < 10⁻⁵⁰ | < 10⁻²⁵ | < 10⁻²⁶ | < 10⁻¹⁸ | < 10⁻¹⁴ |

![Filler Comparison Gemma](code/figures/filler_comparison_google_gemma-3n-e4b-it.png)
![Filler Comparison Qwen 7B](code/figures/filler_comparison_qwen_qwen-2.5-7b-instruct.png)
![Filler Comparison Qwen 72B](code/figures/filler_comparison_qwen_qwen-2.5-72b-instruct.png)
![Filler Comparison DeepSeek](code/figures/filler_comparison_deepseek_deepseek-chat-v3.1.png)
![Filler Comparison Mistral](code/figures/filler_comparison_mistralai_mistral-small-24b-instruct-2501.png)

Correction filler is strongly protective in every model (GLMM β between −0.77 and −1.02). A conversation where the model has been correcting the user roughly halves the sycophancy rate compared to one where it's been agreeing.

### Domain Breakdown

![Domain Gemma](code/figures/domain_breakdown_google_gemma-3n-e4b-it.png)
![Domain Qwen 7B](code/figures/domain_breakdown_qwen_qwen-2.5-7b-instruct.png)
![Domain Qwen 72B](code/figures/domain_breakdown_qwen_qwen-2.5-72b-instruct.png)
![Domain DeepSeek](code/figures/domain_breakdown_deepseek_deepseek-chat-v3.1.png)
![Domain Mistral](code/figures/domain_breakdown_mistralai_mistral-small-24b-instruct-2501.png)

All five models show the same hierarchy: Opinion and Logic most vulnerable, Math and CS most resistant. Gemma's domain breakdown is the most extreme — Opinion hits ~60% sycophancy and even Factual sits at ~18%.

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

## Cost

All experiments run via OpenRouter. Judge is Claude Sonnet 4.6 at $3/M input tokens.

| Model | Experiment | Judge | Total |
|---|---|---|---|
| Gemma 3N E4B ($0.02/M) | $3 | $68 | **$72** |
| Qwen 2.5 7B ($0.10/M) | $16 | $66 | **$82** |
| Qwen 2.5 72B ($0.12/M) | $20 | $68 | **$89** |
| DeepSeek V3.1 ($0.15/M) | $25 | $68 | **$94** |
| Mistral Small 24B ($0.05/M) | $9 | $68 | **$77** |
| **Total** | **$73** | **$339** | **$413** |

The judge dominates cost (~82%). The experiments themselves are cheap — even the 72B model only costs $20 for 11K calls.

## Repo Structure

```
├── README.md                   # This file
├── experiment-protocol.md      # Full experimental protocol
├── research-note.md            # Findings and interpretation so far
├── research-notes/             # Background literature review and analysis
│
└── code/
    ├── probes.json             # 115 probes (6 domains) + 8 persona templates
    ├── run_experiment.py       # Async experiment runner
    ├── llm_judge.py            # Domain-aware LLM judge
    ├── phase_diagram.py        # All figures: phase diagram, domain, filler, heatmap
    ├── statistical_tests.py    # Spearman, Mann-Whitney, chi-squared, GLMM
    ├── run_qwen.sh             # Qwen 7B pipeline
    ├── run_qwen72b.sh          # Qwen 72B pipeline
    ├── run_mistral.sh          # Mistral Small 24B pipeline
    ├── run_deepseek.sh         # DeepSeek V3.1 pipeline
    ├── run_gemma.sh            # Gemma 3N E4B pipeline
    ├── results/                # Raw + judged JSONL (56K+ results across 5 models)
    └── figures/                # Generated figures + stats_report.json
```

## Limitations

1. **Context-length effect is size-dependent.** Small models (~4-7B) show it clearly, large models (24B+) don't. The mechanism is likely attention capacity — small models can't maintain factual beliefs when attention is diluted across a full 32K context.

2. **Qwen 7B's step function vs Gemma's ramp.** Both small models degrade, but with different shapes. More small models would clarify whether step vs ramp is architecture-dependent.

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
