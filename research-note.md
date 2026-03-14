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

## Key Findings

### 1. Context-length effect is real but model-dependent

Qwen shows a clear step: sycophancy jumps from 13.1% at empty context to ~23% once any conversation history is present, then stays flat. The jump happens between 0% and 10% context — roughly the first 3K tokens of filler. After that, adding more context doesn't make things worse.

Mistral barely moves. It goes from 3.0% to 4.9% across the full range, with most drift concentrated in the 70-100% band. Statistically significant (p=0.0004) but practically negligible — Cohen's h of 0.10.

The takeaway: context length alone isn't the story. Model size and training matter enormously. A well-trained 24B model is nearly immune to what a 7B model struggles with.

### 2. Filler type is the strongest and most replicable finding

This is the result that holds across both models with large effect sizes:

| Filler | Qwen Rate | Mistral Rate |
|---|---|---|
| Agreement | 25.3% | 5.6% |
| Neutral | 23.1% | 3.8% |
| Correction | 15.6% | 2.1% |

Both models: agreement > neutral > correction, chi-squared p-values in the 10⁻¹⁴ to 10⁻²⁵ range. The GLMM confirms it — correction filler has a strong negative coefficient (log-odds β = −0.82 for Qwen, −0.99 for Mistral), meaning a conversation history of corrections roughly halves the sycophancy rate.

This is the "behavioral ratchet" from the original preprint hypothesis. It replicates cleanly. The practical implication: conversational pattern shapes model behavior more than conversation length. A model that's been agreeing keeps agreeing. A model that's been correcting keeps correcting. The history trains the model within the session.

### 3. Domain vulnerability has a consistent hierarchy

Both models show Opinion and Logic probes as most sycophancy-prone, with formal/technical domains (Math, CS) being most resistant. The ordering isn't identical — Qwen's Math is most resistant while Mistral's CS is completely immune (0% across 495 trials) — but the broad pattern holds.

This makes intuitive sense. Mathematical truths have strong, unambiguous representations in the model's weights. Opinions have inherently weaker "correct answer" signals, making the model more susceptible to social pressure.

### 4. The 7B vs 24B gap is massive

Qwen's overall sycophancy rate is 21.3%. Mistral's is 3.8%. That's a 5.6x difference. This is probably the most important practical finding: model quality dominates all other variables we tested. Context length, filler type, probe domain — all of these modulate sycophancy within a model, but the baseline difference between models dwarfs everything else.

## What We Haven't Tested Yet

- More models at the 32K boundary to separate size effects from architecture/training effects
- Granular 0-10% context levels to characterize Qwen's step function precisely
- Persona analysis (the data exists but hasn't been analyzed yet — do authority claims amplify the effect?)
- Inter-rater reliability with a second judge model
- Models with different context limits (8K, 64K, 128K) to test whether the effect scales with window size

## Statistical Methods

Primary model: Bayesian binomial GLMM with probe_id as random intercept and logit link. This is the correct specification for binary outcomes (sycophantic vs honest) with clustering by probe. Fallback to GEE logistic if the Bayesian fit fails. Supporting tests: Spearman rank correlation, Mann-Whitney U (low vs high context), chi-squared (filler type independence), Cohen's h (effect size).

## Bottom Line So Far

The context-length → sycophancy relationship exists but is weaker and more model-dependent than the original preprint suggested. The stronger, more replicable finding is the behavioral ratchet: **what kind of conversation the model has been having matters more than how long it's been going.** Agreement patterns compound. Correction patterns protect. This holds across model sizes and architectures.

The next experiments should focus on (a) more models to map the landscape, and (b) finer-grained analysis of the 0-10% transition where Qwen's step function lives.
