# Research Note: Context-Window Sycophancy — Findings So Far

**Karan Prasad, Obvix Labs — March 2026**

## What We Set Out to Test

The hypothesis: as an LLM's context window fills up, its tendency to agree with the user (sycophancy) increases. Three possible mechanisms — attention dilution over long sequences, RLHF reward hacking toward agreeable outputs, and conversational momentum where prior agreement patterns self-reinforce. We built an automated pipeline to measure this at scale across multiple models and architectures.

## Experimental Setup

115 probes across 6 domains (factual, math, science, logic, CS, opinion). Each probe presents the model with a false or debatable claim and measures whether it pushes back or caves. 11 context-fill levels (0% to 100% of 32K tokens), 3 filler types (neutral, agreement, correction), 3 repeats per condition. 11,385 trials per model. All scoring by Claude Sonnet 4.6 with domain-specific rubrics.

All models tested at their 32K architectural limit — filling 100% of the context window creates real pressure. We confirmed a 1M-context model (Gemini Flash) at 3% utilization produces no effect.

## Models Tested

| Model | Parameters | Family | Trials (valid) | Overall Sycophancy |
|---|---|---|---|---|
| Google Gemma 3N E4B | ~4B effective (MoE) | Google | 11,245 | 34.2% |
| Qwen 2.5 7B | 7B | Alibaba | 11,003 | 21.3% |
| Mixtral 8x7B | ~12B active (MoE) | Mistral | 11,331 | 22.7% |
| Mistral Small 24B | 24B | Mistral | 11,381 | 3.8% |
| DeepSeek V3.1 | ~37B active (MoE) | DeepSeek | 11,367 | 6.0% |
| Qwen 2.5 72B | 72B | Alibaba | 11,381 | 6.7% |

Total: 67,708 valid trials across 6 models, 4 families. Total cost: ~$573 ($165 for experiments, $408 for Sonnet 4.6 judge).

## Key Findings

### 1. Context-length effect is real for small models, absent for large ones

The clearest pattern across 6 models: context-window degradation is a small-model problem.

| Size Tier | Model | 0% → 100% | Delta | Significant? |
|---|---|---|---|---|
| **Small** | Gemma 3N (~4B) | 27.7% → 38.4% | +10.7pp | Yes (ρ=0.077, p<10⁻¹⁵) |
| **Small** | Qwen 7B | 13.1% → 21.2% | +8.1pp | Yes (ρ=0.028, p=0.004) |
| **Mid** | Mixtral 8x7B (~12B) | 19.0% → 22.7% | +3.7pp | Weak (ρ=0.021, p=0.03) |
| **Large** | Mistral 24B | 3.0% → 4.9% | +1.9pp | Negligible (h=0.10) |
| **Large** | DeepSeek V3.1 (~37B) | 7.4% → 5.5% | −1.8pp | No (p=0.38) |
| **Large** | Qwen 72B | 4.6% → 8.0% | +3.4pp | Negligible (h=0.14) |

The threshold appears to be around 20-24B parameters. Below that, models degrade measurably. Above that, they're essentially immune. Mixtral 8x7B sits at the boundary — ~12B active parameters, with a mild drift that's barely significant.

The degradation shapes differ by architecture: Gemma shows a clean gradual ramp, Qwen 7B shows a step function at 0→10% context, Mixtral shows a mild initial ramp then plateau. But the direction is consistent across all small models — more context, more sycophancy.

### 2. Filler type is the universal finding

This is the paper's strongest result. Six models, four families, parameter scales from ~4B to 72B — the behavioral ratchet replicates every time:

| Filler | Gemma 3N | Qwen 7B | Mixtral 8x7B | Mistral 24B | DeepSeek V3.1 | Qwen 72B |
|---|---|---|---|---|---|---|
| Agreement | 41.2% | 25.3% | 27.9% | 5.6% | 8.6% | 10.2% |
| Neutral | 36.2% | 23.1% | 26.6% | 3.8% | 5.7% | 5.8% |
| Correction | 25.1% | 15.6% | 13.6% | 2.1% | 3.7% | 4.2% |

Agreement > neutral > correction. Every model. Chi-squared significance ranges from p < 10⁻¹⁴ to p < 10⁻⁵⁸.

Mixtral shows the strongest correction effect of any model (GLMM β = −1.74) — a correction history cuts its sycophancy in half. Correction filler is consistently protective across all models (β between −0.77 and −1.74).

The practical implication: if you want honest responses from a model, don't let the conversation become a chain of agreements. Periodically push back, correct the model, or introduce disagreement. The conversational pattern trains the model within the session.

### 3. Three clear clusters in the phase diagram

The phase diagram reveals a clean stratification:

**Cluster 1 — High sycophancy, context-sensitive (~4-12B):** Gemma 3N (34%), Mixtral 8x7B (23%), Qwen 7B (21%). These models have high baselines and show measurable degradation as context fills. They don't have enough capacity to maintain factual beliefs when attention is diluted across thousands of filler tokens.

**Cluster 2 — Low sycophancy, context-resistant (24B+):** Qwen 72B (6.7%), DeepSeek V3.1 (6.0%), Mistral Small 24B (3.8%). Flat lines at the bottom of the chart. The context effect is either negligible or nonexistent.

**The gap between clusters is dramatic** — Mixtral at 23% vs Mistral Small at 3.8% — and there's nothing in between. The transition from vulnerable to resistant happens sharply between 12B and 24B.

### 4. Within-family comparisons

**Qwen family (7B → 72B):** Scaling 10x drops sycophancy 3x (21% → 7%) and eliminates the step function. But the 72B retains the highest agreement β (1.40) of any model — Qwen's RLHF makes it particularly susceptible to agreement priming regardless of scale.

**Mistral family (Mixtral ~12B active → Small 24B):** Mixtral at 23% vs Mistral Small at 3.8% — a 6x difference within the same lab's models. Mixtral has the strongest correction β (−1.74) of any model, meaning correction history is unusually powerful for this architecture. Mistral Small is the least sycophantic model we tested.

### 5. Domain vulnerability is universal

All six models show the same hierarchy: Opinion and Logic are most vulnerable, Math and CS are most resistant. This holds regardless of model size or family. The ordering is remarkably stable — even though Gemma's Opinion rate is ~60% and Mistral's is 5.5%, both rank Opinion as their most sycophantic domain.

### 6. The credential paradox — informal personas outperform authority claims

We varied 8 persona templates across all non-opinion probes (58,865 persona-assigned trials). The naive hypothesis — "models defer more to claimed experts" — is wrong. The ranking across 5 of 6 models is:

**Social pressure ("everyone knows") > Low authority (casual/friend) > Credential claim (PhD/professional) > External source (professor/paper).**

The "Friend discussion" persona ("I was explaining to my friend...") is the single most sycophancy-inducing template for 3 of 6 models. Gemma 3N ranges from 15.0% (Professor appeal) to 41.8% (Friend discussion) — a 27pp spread. Mixtral shows a 25pp spread (13.5% PhD → 38.8% Friend discussion).

Why? Likely two mechanisms: (1) credential claims activate "expert verification" circuits in the model — if someone says "I have a PhD", the model may actually scrutinize the claim harder. (2) Social/casual framings trigger conversational agreeableness without triggering factual cross-checking. The model is trained to be helpful and friendly, not to defer to authority — so "my friend disagrees with me" is a stronger pull than "I'm an expert."

DeepSeek V3.1 is the only model immune to persona effects (χ²=8.04, p=0.33) — its low sycophancy rate is stable regardless of how the claim is framed.

The persona effect does not amplify meaningfully with context fill. The authority group spread at low context (0-50%) is similar to the spread at high context (50-100%) for all models. Context length and persona template are approximately independent effects.

### 7. Model size dominates all other variables

The single biggest predictor of sycophancy isn't context length, filler type, or probe domain — it's model size. Gemma (~4B) at 34% is 9x more sycophantic than Mistral (24B) at 3.8%. The between-model variance dwarfs all within-model effects.

## What This Means for the Paper

The original hypothesis — "context length causes sycophancy" — needs reframing. The more accurate claim: **context length degrades small models, but the effect disappears with scale.** The universal, scale-invariant finding is the behavioral ratchet.

The paper should have three main contributions:

1. **The size-dependent context effect.** Small models (~4-12B) degrade measurably as context fills. Large models (24B+) don't. The threshold is around 20-24B parameters. This is important for anyone deploying small models in long-conversation applications.

2. **The behavioral ratchet.** Conversational pattern shapes model honesty more than conversation length. Agreement compounds, correction protects. This holds universally across 6 models and 67,708 trials, and is the more practically actionable finding.

3. **The credential paradox.** Informal social framings ("my friend disagrees", "everyone knows") trigger more sycophancy than expert claims ("I have a PhD", "15 years experience"). Models are trained to be agreeable, not to defer to authority. The persona effect is significant in 5/6 models (χ² up to 406, p < 10⁻³⁵) and acts independently of context length.

## What We Haven't Tested Yet

- ~~Persona analysis~~ ✓ Done — credential paradox finding (see §6)
- ~~Inter-rater reliability~~ ✓ Done — κ = 0.705, substantial agreement (see §8)
- Models with different context limits (8K, 64K, 128K)
- More granular 0-10% context levels for Qwen 7B's step function

### 8. Inter-rater reliability validates the judge

We re-judged a stratified subsample of 1,200 trials (200 per model, proportional sycophantic/honest split) with Claude 3.5 Haiku as an independent second judge. Results:

| Metric | Value |
|---|---|
| Overall Cohen's κ | **0.705** (substantial) |
| Raw agreement | 93.4% |
| Sonnet sycophancy rate | 15.6% |
| Haiku sycophancy rate | 10.0% |

Haiku is systematically more lenient: of 79 total disagreements, 73 (92%) are cases where Sonnet flagged sycophancy but Haiku called it honest. Only 6 went the other direction. This means our Sonnet-based results are, if anything, a *conservative overestimate* — the true sycophancy rates may be slightly lower, but the relative rankings and all treatment effects hold.

Per-model κ ranges from 0.526 (Qwen 7B, moderate) to 0.862 (Qwen 72B, almost perfect). Per-domain: science (κ=0.859) and factual (κ=0.821) show almost perfect agreement; opinion (κ=0.439) is the weakest — unsurprising given that opinion judgments are inherently more subjective.

The key takeaway: a $408 Sonnet judge and a ~$3 Haiku judge agree on 93% of trials with substantial κ. The Sonnet judge is defensible.

## Statistical Methods

Primary model: Bayesian binomial GLMM with probe_id as random intercept and logit link. Fallback: GEE logistic → plain logistic. Supporting: Spearman rank correlation, Mann-Whitney U, chi-squared, Cohen's h. Inter-rater reliability: Cohen's κ on stratified 1,200-trial subsample with Claude 3.5 Haiku as second judge.

## Bottom Line

Across 6 models, 4 families, and 67,708 trials: **small models break as conversations get longer, large models don't, and conversational pattern matters more than conversation length for all models.** Agreement patterns compound sycophancy. Correction patterns protect against it. Informal social framings trigger more sycophancy than expert credentials — the model wants to be liked, not to defer to authority. These findings are robust, replicable, and practically actionable.
