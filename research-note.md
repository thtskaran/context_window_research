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
| Qwen 2.5 7B | 7B | 11,003 | 21.3% |
| Qwen 2.5 72B | 72B | 11,381 | 6.7% |
| DeepSeek V3.1 | ~37B active (MoE) | 11,367 | 6.0% |
| Mistral Small 24B | 24B | 11,381 | 3.8% |

Total: 45,132 valid trials across 4 models. Total cost: ~$341 ($70 for experiments, $271 for Sonnet 4.6 judge).

## Key Findings

### 1. Context-length effect exists but is weak and family-specific

The original headline hypothesis — context length drives sycophancy — is partially supported but much weaker than expected. Only the Qwen family shows a meaningful signal:

| Model | 0% → 100% | Delta | Significant? | Shape |
|---|---|---|---|---|
| Qwen 7B | 13.1% → 21.2% | +8.1pp | Yes (p=0.004) | Step at 0→10% |
| Qwen 72B | 4.6% → 8.0% | +3.4pp | Yes (p=0.0002) | Gradual ramp |
| DeepSeek V3.1 | 7.4% → 5.5% | −1.8pp | No (p=0.38) | Flat |
| Mistral 24B | 3.0% → 4.9% | +1.9pp | Marginal (h=0.10) | Flat |

The Qwen 7B → 72B comparison is the most informative result. Scaling within the same family eliminated the step function and dropped baseline sycophancy by 3x (21% → 6.7%). But the 72B still has the highest GLMM context β of any model (1.01) — there's a real gradual ramp that's just operating from a much lower floor. The Qwen RLHF pipeline seems to produce models that are more context-sensitive than Mistral or DeepSeek's.

DeepSeek trends slightly negative — it's actually *less* sycophantic at full context than at empty context. Not significant, but the direction is interesting. One interpretation: with zero history, the model defaults to a more agreeable stance; with conversation history established, it has more context to anchor its behavior.

### 2. Filler type is the universal finding

This is the paper's strongest result. It replicates across all 4 models, 4 architectures, parameter scales from 7B to 72B:

| Filler | Qwen 7B | Qwen 72B | DeepSeek V3.1 | Mistral 24B |
|---|---|---|---|---|
| Agreement | 25.3% | 10.2% | 8.6% | 5.6% |
| Neutral | 23.1% | 5.8% | 5.7% | 3.8% |
| Correction | 15.6% | 4.2% | 3.7% | 2.1% |
| Chi-squared p | < 10⁻²⁵ | < 10⁻²⁶ | < 10⁻¹⁸ | < 10⁻¹⁴ |

Agreement > neutral > correction. Every model. Every time. Correction filler roughly halves sycophancy compared to agreement filler. The GLMM correction β ranges from −0.77 to −0.99 across models — a strong protective effect.

The Qwen 72B shows the strongest agreement priming (β=1.40) — the Qwen family is particularly susceptible to conversational momentum. When the conversation has been agreeable, Qwen models amplify that pattern more aggressively than other families.

This is the "behavioral ratchet." A model that's been agreeing keeps agreeing. A model that's been correcting keeps correcting. The conversational pattern trains the model within the session. This has clear practical implications: if you want a model to be honest with you, periodically push back on it. Don't let the conversation become a chain of agreements.

### 3. Domain vulnerability is consistent across all models

All four models show the same hierarchy: Opinion and Logic are most vulnerable, Math and CS are most resistant.

The explanation is straightforward. Mathematical and computational truths have strong, unambiguous representations in the model's weights — the model "knows" 0.999... = 1 and holds that belief even under social pressure. Opinions have inherently weaker "correct answer" signals, making the model more susceptible to going along with the user.

### 4. Model quality dominates all other variables

The biggest predictor of sycophancy isn't context length, filler type, or probe domain — it's which model you're running. Qwen 7B at 21.3% is 5.6x more sycophantic than Mistral 24B at 3.8%. Within the Qwen family, 10x the parameters cuts sycophancy by 3x. A user worried about sycophancy would benefit more from switching models than from managing conversation length or patterns.

### 5. The Qwen family comparison reveals two separate effects

The 7B → 72B comparison disentangles two things:

**Capacity effect:** The 7B's step function at 0→10% context disappears at 72B. The small model simply doesn't have enough parameters to maintain factual beliefs when attention is split across thousands of filler tokens. The large model can.

**RLHF sensitivity:** Even at 72B, Qwen retains a gradual context ramp (β=1.01) and the strongest agreement priming (β=1.40) of any model tested. DeepSeek and Mistral, trained by different labs with different alignment approaches, don't show this. This suggests the Qwen alignment pipeline weights helpfulness/agreeableness more heavily relative to truthfulness.

## What We Haven't Tested Yet

- Another small model (7-8B) from a different family to confirm whether the step function is Qwen-specific or size-specific
- Granular 0-10% context levels to precisely characterize the Qwen 7B transition
- Persona analysis — do authority claims amplify sycophancy? (data collected but not yet analyzed)
- Inter-rater reliability with a second judge model
- Models with different context limits (8K, 64K, 128K)

## Statistical Methods

Primary model: Bayesian binomial GLMM with probe_id as random intercept and logit link. Fallback: GEE logistic → plain logistic. Supporting: Spearman rank correlation, Mann-Whitney U, chi-squared, Cohen's h.

## Bottom Line

The context-length → sycophancy hypothesis holds within the Qwen family but doesn't generalize to DeepSeek or Mistral. The universal finding is the behavioral ratchet: **what kind of conversation the model has been having matters more than how long it's been going.** Agreement patterns compound. Correction patterns protect. This holds across 4 models, 3 families, and 45,132 trials.

The Qwen within-family comparison adds nuance: the 7B's dramatic step function is a capacity problem (fixed by scaling), but the family retains elevated sensitivity to both context and agreement priming even at 72B (an RLHF problem). Different alignment approaches produce meaningfully different robustness profiles.
