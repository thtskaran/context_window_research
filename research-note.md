# Research Note: Context-Window Sycophancy — Findings So Far

**Karan Prasad, Obvix Labs — March 2026**

## What We Set Out to Test

The hypothesis: as an LLM's context window fills up, its tendency to agree with the user (sycophancy) increases. Three possible mechanisms drive this — attention dilution over long sequences, RLHF reward hacking toward agreeable outputs, and conversational momentum where prior agreement patterns self-reinforce. We built an automated pipeline to measure this at scale.

## Experimental Setup

115 probes across 6 domains (factual, math, science, logic, CS, opinion). Each probe presents the model with a false or debatable claim and measures whether it pushes back or caves. We test at 11 context-fill levels (0% to 100% of 32K tokens), with 3 filler types (neutral, agreement, correction) and 3 repeats per condition. That gives 11,385 trials per model. All scoring done by Claude Sonnet 4.6 as an LLM judge with domain-specific rubrics.

We chose models where 32K tokens is the actual architectural limit — filling 100% of their context window creates real pressure, unlike testing a 1M-context model at 3% capacity (which we confirmed produces no effect with Gemini Flash).

## Models Tested

**Qwen 2.5 7B Instruct** — 7B parameter, 32K context. 11,003 valid trials (37 ambiguous).

**Mistral Small 24B Instruct** — 24B parameter, 32K context. 11,381 valid trials (4 ambiguous).

**DeepSeek V3.1** — ~37B active parameters (MoE), 32K context. 11,367 valid trials (17 ambiguous).

Total: 33,751 valid trials across 3 models.

## Key Findings

### 1. Context-length effect is weak and model-specific

This was supposed to be the headline finding. It isn't. Only Qwen shows a meaningful context-length effect — sycophancy jumps from 13.1% at empty context to ~23% once any conversation history is present, then stays flat. The jump happens between 0% and 10% context.

The other two models show no practically meaningful change:

| Model | 0% context | 100% context | Delta | Significant? |
|---|---|---|---|---|
| Qwen 7B | 13.1% | 21.2% | +8.1pp | Yes (p=0.004) |
| DeepSeek V3.1 | 7.4% | 5.5% | −1.8pp | No (p=0.38) |
| Mistral 24B | 3.0% | 4.9% | +1.9pp | Yes but negligible (h=0.10) |

DeepSeek actually gets slightly *less* sycophantic as context fills — the opposite of the hypothesis. The GLMM context β is −0.22 (negative). Mistral shows a statistically significant but practically negligible drift.

The context-length → sycophancy story is essentially a Qwen-7B-specific finding. Whether this is about model size, training approach, or architecture remains an open question.

### 2. Filler type is the universal finding

This is the result that replicates cleanly across all three models, all three architectures, all three parameter scales:

| Filler | Qwen 7B | DeepSeek V3.1 | Mistral 24B |
|---|---|---|---|
| Agreement | 25.3% | 8.6% | 5.6% |
| Neutral | 23.1% | 5.7% | 3.8% |
| Correction | 15.6% | 3.7% | 2.1% |
| Chi-squared p | < 10⁻²⁵ | < 10⁻¹⁸ | < 10⁻¹⁴ |

Agreement > neutral > correction, every time. The GLMM coefficients tell the same story — correction filler has a strong negative coefficient (β between −0.82 and −0.99 across models), meaning a conversation history of corrections roughly halves the sycophancy rate.

This is the "behavioral ratchet." A model that's been agreeing keeps agreeing. A model that's been correcting keeps correcting. The conversational pattern trains the model within the session. It replicates at 7B, 24B, and 37B. It replicates across Qwen, Mistral, and DeepSeek. It's the paper's real contribution.

### 3. Domain vulnerability has a consistent hierarchy

All three models show the same broad ordering: Opinion and Logic probes are most sycophancy-prone. Math and CS are most resistant.

DeepSeek's domain profile: Opinion (~13%) > Logic (~10%) > Factual (~6%) ≈ Science (~6%) > CS (~2%) > Math (~1%). This mirrors the other two models. The explanation is straightforward — mathematical truths have strong, unambiguous representations in the model's weights. Opinions have inherently weaker "correct answer" signals.

### 4. Model quality dominates everything

The biggest variable in our entire experiment isn't context length, filler type, or probe domain. It's which model you're using.

| Model | Overall sycophancy |
|---|---|
| Qwen 7B | 21.3% |
| DeepSeek V3.1 | 6.0% |
| Mistral 24B | 3.8% |

Qwen is 5.6x more sycophantic than Mistral, and 3.5x more than DeepSeek. The difference between models is far larger than any within-model effect we measured. A user worried about sycophancy would benefit more from switching models than from managing conversation length.

### 5. DeepSeek's slight negative trend is interesting

DeepSeek is the only model that trends slightly downward — higher sycophancy at 0% context (7.4%) than at 100% (5.5%). This isn't statistically significant (p=0.38), but the direction is notable. One possible explanation: with zero conversation history, the model has less context about what kind of conversation this is, and defaults to a more agreeable stance. Once filler messages establish the conversational frame, the model has more signal to work with and pushes back more effectively. This would mean some amount of context *helps* rather than hurts.

## What We Haven't Tested Yet

- More models at the 32K boundary — especially another small model (7-8B) to see if Qwen's vulnerability is size-related or Qwen-specific
- Granular 0-10% context levels to characterize Qwen's step function precisely
- Persona analysis (the data exists but hasn't been analyzed — do authority claims amplify sycophancy?)
- Inter-rater reliability with a second judge model
- Models with different context limits (8K, 64K, 128K) to test whether the effect scales with window size

## Statistical Methods

Primary model: Bayesian binomial GLMM with probe_id as random intercept and logit link. This is the correct specification for binary outcomes (sycophantic vs honest) with clustering by probe. Fallback to GEE logistic if the Bayesian fit fails. Supporting tests: Spearman rank correlation, Mann-Whitney U (low vs high context), chi-squared (filler type independence), Cohen's h (effect size).

## Bottom Line

The original hypothesis — that context-length pressure increases sycophancy — holds for Qwen 7B but not for DeepSeek V3.1 or Mistral Small 24B. The effect is model-specific, not universal.

The stronger, universal finding is the behavioral ratchet: **what kind of conversation the model has been having matters more than how long it's been going.** Agreement patterns compound. Correction patterns protect. This holds across 3 models, 3 architectures, and 33,751 trials. It's the finding worth building a paper around.
