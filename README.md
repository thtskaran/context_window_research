# Context-Window Lock-In: Measuring How LLMs Break as Conversations Get Longer

Does sycophancy increase as an LLM's context window fills up? We test this across six 32K-context models totalling 67,708 valid trials. The context-length effect scales inversely with model size — small models (~4-12B) degrade measurably, large models (24B+) are flat. The universal finding across all six models is the **behavioral ratchet**: conversational pattern matters more than conversation length. Agreement filler roughly doubles sycophancy compared to correction filler (p < 10⁻¹⁴ in every model). A follow-up **correction injection experiment** (4,140 trials) shows the ratchet can be partially or fully reset by injecting correction exchanges — large models respond to as few as 1 correction, while small models need 5-10. A **mixed filler experiment** (4,799 trials) tests ecological validity by interleaving agreement and correction at 7 ratios — the ratchet is a smooth gradient with no sharp phase transition, and even 10% correction scattered through a conversation provides massive protection. A **fine-grained 0–10% experiment** (3,786 trials) zooms into Qwen 7B's step function with 1% increments — revealing a genuine phase transition at 0→1% context fill that is neutral-filler-specific: ~300 tokens of neutral conversation doubles sycophancy, while agreement and correction filler show no step.

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

Three clear clusters emerge. The small models (Gemma ~4B, Qwen 7B, Mixtral ~12B active) sit at the top with high baselines (20-34%) and visible context-length effects. The large models (Mistral 24B, DeepSeek ~37B, Qwen 72B) cluster at the bottom (4-7%), essentially flat. The vulnerability threshold depends on **effective representational capacity** — not purely parameter count. For dense GQA architectures, it falls at ~20-24B active parameters. Sparse attention (Gemma 3N) shifts vulnerability upward; learned KV compression like MLA (DeepSeek V3) shifts immunity downward. See `architecture-analysis.md` for the full cross-model analysis.

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

### Latency and Length: Size-Dependent Behavioral Divergence

Sycophantic responses are **faster** than honest ones in 4/6 models (Gemma −4%, Qwen 7B −10%, Mixtral −7%, DeepSeek −2%), all significant at p < 0.001. The two large models that buck the trend (Mistral 24B +11%, Qwen 72B +8%) produce longer, more hedged sycophantic responses.

Sycophantic responses are also **shorter** in 4/6 models (8-12% fewer words). The exceptions (Mistral 24B +17%, Qwen 72B +22%) write longer sycophantic responses padded with qualifications. This is not a universal "computational shortcut" — it's a **capacity-dependent divergence**: small models (<12B) lack capacity for complex disagreement and default to brief agreement; large models (>24B) produce longer, hedged responses because RLHF training rewards qualified agreement. See `architecture-analysis.md` §9 for the full literature audit.

![Latency](code/figures/latency_comparison.png)
![Length](code/figures/length_comparison.png)

### Correction Injection: Can You Reset the Ratchet?

We tested whether injecting correction exchanges after agreement filler can undo behavioral momentum. Fixed at 50% context, 6 conditions: pure agreement, 1/3/5/10 correction exchanges injected after agreement, and pure correction. Total filler held constant across all conditions to control for context length. 115 probes × 6 conditions × 6 models = 4,140 calls + judge pass.

**The answer: yes, correction injection partially or fully resets the ratchet.** But the dose-response depends on model size.

| Model | Params | agree | inj_1 | inj_3 | inj_5 | inj_10 | correct |
|---|---|---|---|---|---|---|---|
| Gemma 3N | ~4B | 38.1% | 35.4% | 30.6% | 25.7%* | 21.9%** | 16.5% |
| Qwen 7B | 7B | 23.5% | 20.9% | 15.8% | 15.7% | 13.9% | 18.3% |
| Mixtral 8x7B | ~12B | 25.2% | 14.8%* | 14.9% | 9.6%** | 13.0%* | 7.0% |
| Mistral 24B | 24B | 3.5% | 2.6% | 2.6% | 0.9% | 3.5% | 0.9% |
| DeepSeek V3.1 | ~37B | 13.0% | 6.1% | 4.3%* | 5.2%* | 3.5%** | 5.2% |
| Qwen 72B | 72B | 9.6% | 7.0% | 6.1% | 6.1% | 4.3% | 3.5% |

\* p < 0.05, \*\* p < 0.01 vs agree_only (chi-squared)

**Three distinct patterns:**
- **Smooth dose-response (Gemma):** Small model needs many corrections. 1 correction = 12% reset, 10 = 75%. Behavioral momentum has real inertia.
- **Instant reset (DeepSeek, Mixtral):** 1 correction halves sycophancy in DeepSeek (89% reset). Large models are highly responsive to recency.
- **Overcorrection (Qwen 7B):** Injection conditions go *below* pure correction baseline (reset fractions >100%). The agreement→correction sequence is a stronger anti-sycophancy signal than uniform correction. The contrast teaches the model that pushback is expected.

Mistral 24B is a floor effect — baseline sycophancy too low (3.5%) to measure meaningful reset.

### Mixed Filler: Is There a Threshold Ratio?

Real conversations aren't pure agreement or pure correction. We tested ecological validity by interleaving agreement and correction exchanges at 7 ratios (100/0 to 0/100), with exchanges randomly drawn per ratio. Fixed at 50% context, 115 probes × 7 conditions × 6 models = 4,799 valid trials + judge pass.

**The ratchet is a smooth gradient, not a phase transition.** There is no sharp threshold ratio where sycophancy suddenly spikes. Sycophancy scales roughly linearly with agreement ratio across all models.

| Model | 100/0 | 90/10 | 70/30 | 50/50 | 30/70 | 10/90 | 0/100 |
|---|---|---|---|---|---|---|---|
| Gemma 3N | 43.8% | 31.5% | 30.1% | 26.1% | 25.7% | 14.3% | 18.8% |
| Qwen 7B | 24.6% | 25.2% | 19.3% | 16.7% | 18.3% | 17.4% | 12.2% |
| Mixtral 8x7B | 26.1% | 21.7% | 22.3% | 15.9% | 15.8% | 18.3% | 14.0% |
| Mistral 24B | 6.1% | 0.9% | 3.5% | 2.6% | 4.3% | 1.7% | 1.7% |
| DeepSeek V3.1 | 7.0% | 7.0% | 7.0% | 7.9% | 6.1% | 5.2% | 5.2% |
| Qwen 72B | 8.7% | 7.8% | 7.0% | 6.1% | 4.3% | 6.1% | 2.6% |

**Three key findings:**
- **No phase transition.** The mechanism is cumulative exposure (more agreement → stronger in-context prior), not a discrete mode switch.
- **The "last 10% of corrections" effect.** For Gemma, the steepest step is 90/10 → 100/0 (+12.2pp). Even 10% correction interleaved provides massive protection — you don't need 50% corrections, you need ~10%.
- **Large models remain immune.** DeepSeek (5.2–7.9%) and Qwen 72B (2.6–8.7%) show no significant variation across any ratio.

![Mixed Filler Phase Diagram](code/figures/mixed_filler_phase_diagram.png)
![Mixed Filler Susceptible Models](code/figures/mixed_filler_susceptible_models.png)
![Mixed Filler Heatmap](code/figures/mixed_filler_heatmap.png)

### Fine-Grained 0–10%: Is Qwen 7B's Step Function a Phase Transition?

The original experiment showed Qwen 7B jumping from 13.1% to 21.2% sycophancy between 0% and 10% context fill — a step function while other models ramp gradually. We zoomed in with 1% increments (0%, 1%, 2%, ..., 10%) across all 3 filler types. 3,786 valid trials.

**The step function is real — but it's neutral-filler-specific and happens at 0%→1%.** Just ~300 tokens of neutral conversation shifts Qwen 7B from 12.2% to 23.9% sycophancy (χ²=5.3, p<.05). After 1%, the rate plateaus at ~20-28% with no further significant steps. Agreement and correction filler show no significant variation across the entire 0–10% range.

| Pct | Neutral | Agreement | Correction |
|---|---|---|---|
| 0% | 12.2% | 15.8% | 17.4% |
| 1% | **23.9%** (+11.7pp*) | 18.4% | 11.5% |
| 2% | 21.9% | 20.2% | 10.5% |
| 3% | 19.6% | 19.1% | 12.2% |
| 5% | 27.8% | 22.6% | 13.9% |
| 10% | 25.4% | 23.7% | 13.0% |

\* Only significant adjacent step in the entire experiment

**Three key findings:**
- **The 0→1% phase transition is real for neutral filler.** The changepoint at 0%→1% explains 88% of neutral filler's total 0→10% range. This is a genuine mode switch — the model transitions from "fresh conversation" to "ongoing conversation" behavior with just 2-3 neutral exchanges.
- **Agreement and correction are flat.** Neither shows significant variation across 0–10%. The original experiment's "step function" (averaged across fillers) was driven entirely by neutral filler's 0→1% jump.
- **Correction filler is protective even at tiny doses.** Correction rates stay flat at 10–17% across all levels. The protective effect of correction doesn't require volume — even 1% context fill with correction exchanges maintains low sycophancy.

![Fine-Grained Step Function](code/figures/finegrained_step_function.png)
![Fine-Grained Filler Comparison](code/figures/finegrained_filler_comparison.png)
![Fine-Grained Domain Heatmap](code/figures/finegrained_domain_heatmap.png)

### Architectural Explanation: Why Small Models Break

Full analysis in [`architecture-analysis.md`](architecture-analysis.md), grounded in 40+ papers including Qwen/Gemma/DeepSeek/Mistral/Mixtral technical reports and mechanistic sycophancy research.

**Each vulnerable model fails for architecture-specific reasons:**

| Architecture Pattern | Vulnerability | Model | Mechanism |
|---|---|---|---|
| Sparse attention (17% global) + ~4B effective | Gradual ramp (+10.7pp) | Gemma 3N | 83% of layers blind to distant context; global layers overloaded |
| Dense attention + 7B + only 4 KV heads | Binary switch (+8.1pp) | Qwen 7B | 512-dim KV bottleneck forces mode switch at first neutral exchange |
| Dense attention + ~12B active | Borderline (+3.7pp) | Mixtral 8x7B | Enough capacity to partially resist but not fully |
| Dense attention + ≥24B active | Immune | Mistral 24B, Qwen 72B | Sufficient capacity to maintain reasoning alongside persona activation |
| MLA + GRPO + ~37B active | Immune (strongest) | DeepSeek V3 | Learned KV compression + decoupled RoPE + dual reward models |

**Key mechanisms identified:**
- **Persona selection** (Marks et al. 2026): Conversational context triggers a "multi-turn assistant" persona that includes sycophantic tendencies from DPO/GRPO alignment. Near-binary classification, not gradual.
- **Deep representational divergence** (Wang et al. 2025): Sycophancy involves fundamental recoding of problem representation in deeper layers, not just output token reweighting.
- **GQA bottleneck**: Models with fewer KV heads have less capacity to maintain both "deliberative" and "conversational" representations simultaneously — they must commit to one mode.
- **MLA immunity**: DeepSeek V3's learned compression (57× KV reduction) is adaptive — it learns what to preserve, making spurious correlations between context content and sycophancy expensive to maintain.

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
| Injection experiment (all models) | ~$10 | ~$25 | **~$35** |
| Mixed filler experiment (all models) | ~$14 | ~$29 | **~$43** |
| Fine-grained 0–10% (Qwen 7B only) | ~$1 | ~$7 | **~$8** |
| **Total** | **~$190** | **~$492** | **~$682** |

The judge dominates cost (~72%). The experiments themselves are cheap — even the 72B model only costs $20 for 11K calls.

**Injection experiment cost (~$35, approximate):** 690 calls per model (115 probes × 6 conditions) at 50% context fill. Experiment calls are ~6% of the original volume per model, scaling proportionally (~$10 total across 6 models). Judge cost: 4,140 calls × ~$0.006/call ≈ $25.

**Mixed filler experiment cost (~$43, approximate):** 805 calls per model (115 probes × 7 conditions) at 50% context fill. 4,799 total experiment calls across 6 models (~$14 experiment, small models ~$0.002/call, large models ~$0.005/call). Judge cost: 4,799 calls × ~$0.006/call ≈ $29.

**Fine-grained 0–10% experiment cost (~$8, actual):** Single model (Qwen 7B), 3,786 valid trials (115 probes × 11 levels × 3 fillers, 9 dropped). At 0–10% context fill, average input is only ~1,500 tokens — very cheap per call. Experiment: ~$1. Judge: 3,786 calls × ~$0.002/call ≈ $7.

**Taxonomy judge cost (~$25, approximate):** Only the 10,637 sycophantic responses need taxonomy classification (second Sonnet 4.6 pass). Each call sends the taxonomy prompt (1,087 chars) + claim (avg 61 chars) + truth (avg 133 chars) + model response (avg 990 chars, truncated at 2,000 chars) = ~2,271 chars input. Output is one word (~2 tokens). At ~3.3 chars/token → ~788 tokens/call × 10,637 calls = ~8.4M input tokens × $3/M ≈ $25. Output cost is negligible ($0.32). This is an estimate — actual OpenRouter billing may differ slightly due to tokenizer differences.

## Repo Structure

```
├── README.md                   # This file
├── experiment-protocol.md      # Full experimental protocol
├── research-note.md            # Findings and interpretation so far
├── brainstorm-synthesis.md     # Contribution mapping and paper strategy
├── architecture-analysis.md    # Cross-model architecture deep dive (40+ papers)
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
    ├── run_correction_injection.py  # Correction injection mitigation experiment
    ├── analyze_injection.py    # Injection results analysis (reset fractions, dose-response)
    ├── run_injection_all.sh    # Full injection pipeline (all 6 models)
    ├── run_mixed_filler.py     # Mixed filler ratio experiment (interleaved agree/correct)
    ├── analyze_mixed_filler.py # Mixed filler analysis (threshold detection, phase diagram)
    ├── run_mixed_all.sh        # Full mixed filler pipeline (all 6 models)
    ├── run_finegrained.py      # Fine-grained 0–10% experiment (Qwen 7B, 1% steps)
    ├── analyze_finegrained.py  # Fine-grained analysis (changepoint, CI bands, phase transition)
    ├── generate_finegrained_diagrams.py  # Fine-grained figure generation
    ├── run_finegrained_all.sh  # Full fine-grained pipeline (Qwen 7B only)
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

1. **Context-length effect is architecture-dependent.** Vulnerability depends on effective representational capacity (active params × attention coverage × KV compression), not parameter count alone. Sparse attention (Gemma) and few KV heads (Qwen 7B) exacerbate vulnerability; learned compression (DeepSeek MLA) provides additional immunity.

2. **Different degradation shapes reflect different mechanisms.** Gemma ramps gradually (sparse attention bottleneck), Qwen 7B steps (persona mode switch via GQA bottleneck), Mixtral drifts mildly (borderline capacity). These are architecturally grounded, not random variation.

3. **Template filler.** 10 template pairs per filler type. At 100% context, pairs repeat ~6x each.

4. **Single judge model.** All scoring by Sonnet 4.6. Inter-rater reliability with Claude 3.5 Haiku: κ = 0.705 overall (substantial agreement), 93.4% raw agreement on 1,200 stratified trials. Haiku is systematically more lenient — 92% of disagreements are Sonnet calling sycophancy that Haiku forgives. Per-domain: science (κ=0.86) and factual (κ=0.82) are almost perfect; opinion (κ=0.44) is the weakest domain for inter-rater agreement.

5. **No Llama models.** No Meta Llama model has a 32K native context limit — Llama 3.1 8B is 128K on OpenRouter, so 32K only fills 12% of its window.

6. **Injection experiment at single context level.** Correction injection tested only at 50% context fill. The interaction between injection dose and context level (would the ratchet be harder to reset at 90% context?) is untested. The Mixtral non-monotonicity at inject_10 (13% vs inject_5's 9.6%) may be noise or a real overcorrection effect — needs replication.

7. **Injection controls for length but not for content shift.** Replacing agreement tokens with correction tokens changes both the behavioral pattern and the specific content. A stricter control would use semantically matched correction and agreement exchanges on the same topics.

8. **Mixed filler uses random draw, not exact ratio.** Each exchange is drawn with probability = agree_ratio, so actual ratios have sampling variance (e.g., 70/30 target may produce 74/26 in a given trial). With ~40-50 exchanges per trial, the variance is small but nonzero.

9. **Gemma 0/100 anomaly.** Gemma's pure correction rate (18.8%) is higher than its 10/90 rate (14.3%), creating a non-monotonic dip. This may be noise (n=112) or may reflect that Gemma's correction templates trigger a specific response pattern at saturation.

10. **Fine-grained experiment is single-model.** The 0–10% zoom is only on Qwen 7B. Other models showing step functions (if any) would need separate fine-grained runs. This targeted design trades generalizability for resolution on the most interesting observation.

## Citation

```bibtex
@misc{prasad2026contextlock,
  title={Context-Window Lock-In and Silent Degradation in Large Language Models},
  author={Prasad, Karan},
  year={2026},
  note={Preprint, Obvix Labs}
}
```
