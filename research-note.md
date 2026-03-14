# Research Note: Context-Window Sycophancy — Findings So Far

**Karan Prasad, Obvix Labs — March 2026**

## What We Set Out to Test

The hypothesis: as an LLM's context window fills up, its tendency to agree with the user (sycophancy) increases. Three possible mechanisms — attention dilution over long sequences, RLHF reward hacking toward agreeable outputs, and conversational momentum where prior agreement patterns self-reinforce. We built an automated pipeline to measure this at scale.

## Experimental Setup

115 probes across 6 domains (factual, math, science, logic, CS, opinion). Each probe presents the model with a false or debatable claim and measures whether it pushes back or caves. 11 context-fill levels (0% to 100% of 32K tokens), 3 filler types (neutral, agreement, correction), 3 repeats per condition. 11,385 trials per model. All scoring by Claude Sonnet 4.6 with domain-specific rubrics.

All models tested at their 32K architectural limit — filling 100% of the context window creates real pressure. We confirmed a 1M-context model (Gemini Flash) at 3% utilization produces no effect.

## Models Tested

| Model | Parameters | Trials (valid) | Overall Sycophancy |
|---|---|---|---|
| Google Gemma 3N E4B | ~4B effective (MoE) | 11,245 | 34.2% |
| Qwen 2.5 7B | 7B | 11,003 | 21.3% |
| Qwen 2.5 72B | 72B | 11,381 | 6.7% |
| DeepSeek V3.1 | ~37B active (MoE) | 11,367 | 6.0% |
| Mistral Small 24B | 24B | 11,381 | 3.8% |

Total: 56,377 valid trials across 5 models. Total cost: ~$413 ($73 for experiments, $339 for Sonnet 4.6 judge).

## Key Findings

### 1. Context-length effect is real — for small models

This is the finding that needed five models to crystallize. The context-length effect isn't universally weak — it's size-dependent.

| Model | Effective Params | 0% → 100% | Delta | Shape |
|---|---|---|---|---|
| Gemma 3N | ~4B | 27.7% → 38.4% | **+10.7pp** | Gradual ramp |
| Qwen 7B | 7B | 13.1% → 21.2% | **+8.1pp** | Step at 0→10% |
| Qwen 72B | 72B | 4.6% → 8.0% | +3.4pp | Gradual ramp |
| DeepSeek V3.1 | ~37B | 7.4% → 5.5% | −1.8pp | Flat |
| Mistral 24B | 24B | 3.0% → 4.9% | +1.9pp | Flat |

The two smallest models both show clear, statistically significant degradation — Gemma with a clean gradual ramp (ρ=0.077, p<10⁻¹⁵, Cohen's h=0.23), Qwen 7B with a step function. The three larger models (24B+) are essentially flat. Gemma is the strongest context-length result in the entire study — the cleanest monotonic increase, the highest Spearman ρ, and a Cohen's h that crosses into "small effect" territory.

The mechanism is likely attention capacity. A 4B-parameter model distributing attention over 32K tokens of filler simply can't maintain the same focus on the probe as when the context is empty. Larger models have more attention heads and more capacity to keep the critical information salient despite dilution.

The shapes differ between the two small models — Qwen steps, Gemma ramps — which suggests the transition dynamics are architecture-dependent even if the direction is consistent.

### 2. Filler type is the universal finding

The behavioral ratchet replicates across all 5 models, all architectures, parameter scales from ~4B to 72B:

| Filler | Gemma 3N | Qwen 7B | Qwen 72B | DeepSeek V3.1 | Mistral 24B |
|---|---|---|---|---|---|
| Agreement | 41.2% | 25.3% | 10.2% | 8.6% | 5.6% |
| Neutral | 36.2% | 23.1% | 5.8% | 5.7% | 3.8% |
| Correction | 25.1% | 15.6% | 4.2% | 3.7% | 2.1% |
| Chi-squared p | < 10⁻⁵⁰ | < 10⁻²⁵ | < 10⁻²⁶ | < 10⁻¹⁸ | < 10⁻¹⁴ |

Agreement > neutral > correction. Every model. Every time. The most significant result is Gemma's (p < 10⁻⁵⁰). Correction filler roughly halves sycophancy compared to agreement filler across all models.

This is the paper's strongest contribution. The practical implication is clear: conversational pattern shapes model behavior more than conversation length. A model that's been agreeing keeps agreeing. A model that's been correcting keeps correcting. If you want honest responses, don't let the conversation become a chain of agreements — periodically push back.

### 3. Domain vulnerability is consistent

All five models show the same hierarchy: Opinion and Logic most vulnerable, Math and CS most resistant. Gemma's domain breakdown is the most extreme — Opinion ~60%, Logic ~50%, Math ~45%, while Factual sits at ~18%. The domain ordering is remarkably stable across models despite 10x differences in baseline sycophancy rates.

### 4. Model size is the dominant variable

| Model | Effective Params | Overall Sycophancy |
|---|---|---|
| Gemma 3N | ~4B | 34.2% |
| Qwen 7B | 7B | 21.3% |
| Qwen 72B | 72B | 6.7% |
| DeepSeek V3.1 | ~37B | 6.0% |
| Mistral 24B | 24B | 3.8% |

The correlation between model size and sycophancy resistance is strong. Gemma (~4B effective) is 9x more sycophantic than Mistral (24B). Within the Qwen family, 10x the parameters cuts sycophancy by 3x. Model quality dominates all within-model effects we measured.

### 5. The Qwen within-family comparison

The Qwen 7B → 72B comparison is the cleanest controlled experiment in the study — same architecture, same training pipeline, different scale:

| | Qwen 7B | Qwen 72B |
|---|---|---|
| Overall sycophancy | 21.3% | 6.7% |
| Context delta | +8.1pp (step) | +3.4pp (ramp) |
| Agreement filler | 25.3% | 10.2% |
| Agreement GLMM β | +0.233 | +1.402 |

Scaling fixed the step function and dropped the baseline by 3x. But the 72B retains the highest agreement β of any model (1.40) — the Qwen family is particularly susceptible to agreement priming regardless of scale. This separates the capacity effect (step function, fixed by scale) from the RLHF effect (agreement sensitivity, persistent across scale).

### 6. Gemma 3N is the most degraded model we've tested

At ~4B effective parameters, Gemma shows the most severe sycophancy profile:
- 34.2% overall (highest baseline)
- +10.7pp context effect (largest delta)
- 41.2% under agreement filler (highest of any condition)
- Opinion probes hit ~60% sycophancy at high context
- Even CS probes (typically the most resistant domain) climb from 5% to 37%

This is what context-window degradation looks like when the model doesn't have enough capacity to resist it. Gemma's gradual ramp is arguably more concerning than Qwen 7B's step function — the step means the degradation is immediately visible (the model changes behavior as soon as any history exists), while the ramp means a user in a long conversation might not notice the gradual shift.

## What We Haven't Tested Yet

- Persona analysis — do authority claims amplify sycophancy? (data collected across all 5 models, not yet analyzed)
- Inter-rater reliability with a second judge model
- Models with different context limits (8K, 64K, 128K) to test whether the effect scales with window size
- More granular 0-10% context levels for Qwen 7B's step function

## Statistical Methods

Primary model: Bayesian binomial GLMM with probe_id as random intercept and logit link. Fallback: GEE logistic → plain logistic. Supporting: Spearman rank correlation, Mann-Whitney U, chi-squared, Cohen's h.

## Bottom Line

The context-length → sycophancy effect is real but size-dependent. Small models (~4-7B) degrade meaningfully as their context window fills. Large models (24B+) are essentially immune. The universal finding — across 5 models, 3 families, and 56,377 trials — is the behavioral ratchet: **what kind of conversation the model has been having matters more than how long it's been going.** Agreement patterns compound. Correction patterns protect.

For the paper, this gives us two clean stories: (1) context-window degradation is a small-model problem that scaling solves, and (2) conversational momentum shapes model honesty regardless of scale — and that's the more practically important finding.
