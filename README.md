# Context-Window Lock-In: Measuring How LLMs Break as Conversations Get Longer

Does sycophancy increase as an LLM's context window fills up? We test this across six 32K-context models totalling 67,708 valid trials. The context-length effect scales inversely with model size — small models (~4-12B) degrade measurably, large models (24B+) are flat. The universal finding across all six models is the **behavioral ratchet**: conversational pattern matters more than conversation length. Agreement filler roughly doubles sycophancy compared to correction filler (p < 10⁻¹⁴ in every model).

## Results Summary

All results scored by Claude Sonnet 4.6 as LLM judge with domain-aware rubrics.

### Cross-Model Comparison

| Metric | Gemma 3N (~4B) | Qwen 7B | Mixtral 8x7B (~12B) | Mistral 24B | DeepSeek V3.1 (~37B) | Qwen 72B |
|---|---|---|---|---|---|---|
| Trials (valid) | 11,245 | 11,003 | 11,331 | 11,381 | 11,367 | 11,381 |
| Overall sycophancy | 34.2% | 21.3% | 22.7% | 3.8% | 6.0% | 6.7% |
| At 0% context | 27.7% | 13.1% | 19.0% | 3.0% | 7.4% | 4.6% |
| At 100% context | 38.4% | 21.2% | 22.7% | 4.9% | 5.5% | 8.0% |
| Delta | **+10.7 pp** | **+8.1 pp** | **+3.7 pp** | **+1.9 pp** | **−1.8 pp** | **+3.4 pp** |
| Spearman ρ | 0.077*** | 0.028** | 0.021* | 0.033*** | −0.008 ns | 0.035*** |
| Cohen's h | 0.229 (small) | 0.215 (small) | 0.090 (negl.) | 0.100 (negl.) | −0.075 (negl.) | 0.140 (negl.) |
| Trend | Gradual ramp | Step at 0→10% | Mild ramp | Flat | Flat | Gradual ramp |
| GLMM context β | 1.040 | 0.447 | 0.350 | 0.903 | −0.217 | 1.012 |

### The Phase Diagram

![Phase Diagram](code/figures/phase_diagram.png)

Three clear clusters emerge. The small models (Gemma ~4B, Qwen 7B, Mixtral ~12B active) sit at the top with high baselines (20-34%) and visible context-length effects. The large models (Mistral 24B, DeepSeek ~37B, Qwen 72B) cluster at the bottom (4-7%), essentially flat. The size threshold for sycophancy resistance appears to be around 20-24B parameters.

### Filler Type: The Universal Finding

The behavioral ratchet replicates across all six models: agreement > neutral > correction.

| Filler Type | Gemma 3N | Qwen 7B | Mixtral 8x7B | Mistral 24B | DeepSeek V3.1 | Qwen 72B |
|---|---|---|---|---|---|---|
| Agreement | 41.2% | 25.3% | 27.9% | 5.6% | 8.6% | 10.2% |
| Neutral | 36.2% | 23.1% | 26.6% | 3.8% | 5.7% | 5.8% |
| Correction | 25.1% | 15.6% | 13.6% | 2.1% | 3.7% | 4.2% |
| Chi-squared p | < 10⁻⁵⁰ | < 10⁻²⁵ | < 10⁻⁵⁸ | < 10⁻¹⁴ | < 10⁻¹⁸ | < 10⁻²⁶ |

![Filler Comparison Gemma](code/figures/filler_comparison_google_gemma-3n-e4b-it.png)
![Filler Comparison Qwen 7B](code/figures/filler_comparison_qwen_qwen-2.5-7b-instruct.png)
![Filler Comparison Mixtral](code/figures/filler_comparison_mistralai_mixtral-8x7b-instruct.png)
![Filler Comparison Qwen 72B](code/figures/filler_comparison_qwen_qwen-2.5-72b-instruct.png)
![Filler Comparison DeepSeek](code/figures/filler_comparison_deepseek_deepseek-chat-v3.1.png)
![Filler Comparison Mistral](code/figures/filler_comparison_mistralai_mistral-small-24b-instruct-2501.png)

Correction filler is strongly protective in every model. Mixtral shows the strongest correction effect of any model (GLMM β = −1.74) — correction history halves its sycophancy rate from 27.9% to 13.6%.

### Persona Effect: The Credential Paradox

Each probe is delivered through one of 8 rotating persona templates ranging from casual ("I've been thinking...") to high-authority ("I have a PhD", "15 years experience") to social pressure ("everyone knows that"). The counterintuitive finding: **informal, social framings are the most sycophancy-inducing, not credential claims.**

| Authority Group | Gemma 3N | Qwen 7B | Mixtral 8x7B | Mistral 24B | DeepSeek V3.1 | Qwen 72B |
|---|---|---|---|---|---|---|
| Social pressure | 39.9% | 26.1% | 34.4% | 4.7% | 5.7% | 4.4% |
| Low authority | 30.1% | 20.6% | 30.1% | 5.2% | 5.5% | 5.8% |
| Credential claim | 30.9% | 18.9% | 25.4% | 1.6% | 4.9% | 2.6% |
| External source | 26.1% | 16.8% | 17.2% | 4.3% | 4.7% | 5.4% |
| Chi-squared p | < 10⁻¹⁵ | < 10⁻⁹ | < 10⁻³⁵ | < 10⁻¹⁵ | 0.37 (ns) | < 10⁻¹⁰ |

The spread is massive on vulnerable models — Gemma 3N ranges from 15% (Professor appeal) to 42% (Friend discussion) across individual personas. DeepSeek V3.1 is the only model where persona has no significant effect (χ²=8.04, p=0.33).

![Persona Heatmap](code/figures/persona_heatmap.png)
![Persona Authority Bars](code/figures/persona_authority_bars.png)
![Persona Spread](code/figures/persona_spread.png)
![Persona × Context Interaction](code/figures/persona_context_interaction.png)

### Domain Breakdown

![Domain Gemma](code/figures/domain_breakdown_google_gemma-3n-e4b-it.png)
![Domain Qwen 7B](code/figures/domain_breakdown_qwen_qwen-2.5-7b-instruct.png)
![Domain Mixtral](code/figures/domain_breakdown_mistralai_mixtral-8x7b-instruct.png)
![Domain Qwen 72B](code/figures/domain_breakdown_qwen_qwen-2.5-72b-instruct.png)
![Domain DeepSeek](code/figures/domain_breakdown_deepseek_deepseek-chat-v3.1.png)
![Domain Mistral](code/figures/domain_breakdown_mistralai_mistral-small-24b-instruct-2501.png)

All six models show the same hierarchy: Opinion and Logic most vulnerable, Math and CS most resistant.

### How Models Cave: Failure Mode Taxonomy

All 10,637 sycophantic responses classified by Claude Sonnet 4.6 into three failure modes:

| Failure Mode | Overall | Description |
|---|---|---|
| **Qualified agreement** | 49.1% | Contains hedges ("however", "mostly", "it depends") but net validates the false claim |
| **Elaborate justification** | 41.4% | Builds structured arguments, fabricated evidence, or step-by-step reasoning supporting the false claim |
| **Direct validation** | 9.5% | Blunt "You're right!" with no nuance or caveats |

Elaborate justification is the dominant failure mode for small models — Gemma 3N leads at 58%, actively constructing cases for false claims. DeepSeek V3.1 sits at the opposite extreme: 84% qualified, 0% direct — when it caves, it always hedges, never bluntly agrees. Mixtral 8x7B has the most diverse failure profile (51% qualified / 30% elaborate / 19% direct). By domain: math triggers the most elaborate failures (52%) as models generate step-by-step "proofs" of wrong answers. Opinion probes show 50% elaborate / 49% qualified with near-zero direct.

![Taxonomy](code/figures/taxonomy_stacked.png)

### Latency and Length: Sycophancy is Faster and Shorter

Sycophantic responses are **faster** than honest ones in 4/6 models (Gemma −4%, Qwen 7B −10%, Mixtral −7%, DeepSeek −2%), all significant at p < 0.001. The two large models that buck the trend (Mistral 24B +11%, Qwen 72B +8%) are the same ones with the most qualified/elaborate sycophancy — they take longer because they're hedging, not because they're reasoning harder.

Sycophantic responses are also **shorter** in 4/6 models (8-12% fewer words). Again, the two exceptions (Mistral 24B +17%, Qwen 72B +22%) write longer sycophantic responses because they're padding with qualifications. The pattern: small models cave quickly and cheaply; large models that do cave invest more effort in justifying it.

![Latency](code/figures/latency_comparison.png)
![Length](code/figures/length_comparison.png)

### Heatmap

![Heatmap](code/figures/heatmap.png)

## How It Works

We construct synthetic conversations of varying length (0-100% of the 32K context window), then append a probe where the user states something wrong and asks the model to confirm. We measure agreement rate.

**Three filler types** isolate why sycophancy changes: neutral filler (cooking, travel) tests pure length effects; agreement filler (model agrees with user) tests behavioral momentum; correction filler (model corrects user) tests whether correction patterns protect.

**Six probe domains** (115 total probes): 100 factual probes with objectively wrong claims across factual, math, science, logic, and CS, plus 15 opinion probes with genuinely debatable claims.

**Persona variation**: Each non-opinion probe delivered through one of 8 rotating framings — casual thinker, college-educated, PhD holder, friend discussion, professor appeal, 15-year professional, peer-reviewed citation, and social consensus ("everyone knows"). These map to 4 authority groups: low authority, credential claim, external source, and social pressure.

**Scoring**: All results scored by Claude Sonnet 4.6 as judge, with separate rubrics for factual vs opinion probes. Sycophantic responses are further classified into failure modes (direct/elaborate/qualified) by a second Sonnet 4.6 taxonomy judge pass. Inter-rater reliability validated with Claude 3.5 Haiku on a 1,200-trial stratified subsample (Cohen's κ = 0.705, 93.4% agreement).

**Statistical model**: Bayesian binomial GLMM with probe_id as random intercept and logit link.

## Running a New Model

```bash
cd code/

python run_experiment.py --mode api --model <openrouter-model-id> \
  --keys-file keys.txt --max-tokens 32768 --workers 30 --repeats 3

python llm_judge.py --input results/<model_slug>_results.jsonl \
  --output results/<model_slug>_judged.jsonl \
  --judge-model anthropic/claude-sonnet-4-6 --keys-file keys.txt --workers 35

python taxonomy_judge.py --input results/<model_slug>_judged.jsonl \
  --output results/<model_slug>_judged.jsonl \
  --judge-model anthropic/claude-sonnet-4-6 --keys-file keys.txt --workers 35

python phase_diagram.py --results-dir results/ --output-dir figures/
python statistical_tests.py --results-dir results/ --output figures/stats_report.json --verbose
python secondary_analysis.py
```

Create `code/keys.txt` with one OpenRouter API key per line.

## Cost

All experiments run via OpenRouter. Sonnet 4.6 judge at $3/M input, $15/M output tokens.

| Model | Experiment | Sycophancy Judge | Total |
|---|---|---|---|
| Gemma 3N E4B ($0.02/M) | $3 | $68 | **$71** |
| Qwen 2.5 7B ($0.10/M) | $16 | $66 | **$82** |
| Mixtral 8x7B ($0.54/M) | $92 | $68 | **$160** |
| Mistral Small 24B ($0.05/M) | $9 | $68 | **$77** |
| DeepSeek V3.1 ($0.15/M) | $25 | $68 | **$93** |
| Qwen 2.5 72B ($0.12/M) | $20 | $68 | **$88** |
| Taxonomy judge (all models) | — | ~$25 | **~$25** |
| **Total** | **$165** | **$431** | **~$596** |

The judge dominates cost (~72%). The experiments themselves are cheap — even the 72B model only costs $20 for 11K calls.

**Taxonomy judge cost (~$25, approximate):** Only the 10,637 sycophantic responses need taxonomy classification (second Sonnet 4.6 pass). Each call sends the taxonomy prompt (1,087 chars) + claim (avg 61 chars) + truth (avg 133 chars) + model response (avg 990 chars, truncated at 2,000 chars) = ~2,271 chars input. Output is one word (~2 tokens). At ~3.3 chars/token → ~788 tokens/call × 10,637 calls = ~8.4M input tokens × $3/M ≈ $25. Output cost is negligible ($0.32). This is an estimate — actual OpenRouter billing may differ slightly due to tokenizer differences.

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
    ├── llm_judge.py            # Domain-aware LLM judge (sycophantic/honest/ambiguous)
    ├── taxonomy_judge.py       # LLM judge for failure mode taxonomy (direct/elaborate/qualified)
    ├── phase_diagram.py        # All figures
    ├── statistical_tests.py    # GLMM, Spearman, Mann-Whitney, chi-squared
    ├── persona_analysis.py     # Persona template effect analysis
    ├── irr_check.py            # Inter-rater reliability (second judge)
    ├── secondary_analysis.py   # Taxonomy, latency, response length
    ├── run_qwen.sh             # Qwen 7B pipeline
    ├── run_qwen72b.sh          # Qwen 72B pipeline
    ├── run_mistral.sh          # Mistral Small 24B pipeline
    ├── run_mixtral.sh          # Mixtral 8x7B pipeline
    ├── run_deepseek.sh         # DeepSeek V3.1 pipeline
    ├── run_gemma.sh            # Gemma 3N E4B pipeline
    ├── results/                # Raw + judged JSONL (68K+ results across 6 models)
    └── figures/                # Generated figures + stats_report.json
```

## Limitations

1. **Context-length effect is size-dependent.** Small models (~4-12B) show it, large models (24B+) don't. The mechanism is likely attention capacity.

2. **Different degradation shapes.** Gemma ramps gradually, Qwen 7B steps, Mixtral drifts mildly. Architecture-dependent dynamics.

3. **Template filler.** 10 template pairs per filler type. At 100% context, pairs repeat ~6x each.

4. **Single judge model.** All scoring by Sonnet 4.6. Inter-rater reliability with Claude 3.5 Haiku: κ = 0.705 overall (substantial agreement), 93.4% raw agreement on 1,200 stratified trials. Haiku is systematically more lenient — 92% of disagreements are Sonnet calling sycophancy that Haiku forgives. Per-domain: science (κ=0.86) and factual (κ=0.82) are almost perfect; opinion (κ=0.44) is the weakest domain for inter-rater agreement.

5. **No Llama models.** No Meta Llama model has a 32K native context limit — Llama 3.1 8B is 128K on OpenRouter, so 32K only fills 12% of its window.

## Citation

```bibtex
@misc{prasad2026contextlock,
  title={Context-Window Lock-In and Silent Degradation in Large Language Models},
  author={Prasad, Karan},
  year={2026},
  note={Preprint, Obvix Labs}
}
```
