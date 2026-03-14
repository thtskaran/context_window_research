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

Total: 67,708 valid trials across 6 models, 4 families, plus 4,140 correction injection trials and 4,799 mixed filler trials. Grand total: 76,647 trials. Total cost: ~$674 ($189 experiments, $485 judge passes). Breakdown: $165 original experiments + ~$10 injection + ~$14 mixed filler, $406 sycophancy judge + ~$25 taxonomy judge + ~$25 injection judge + ~$29 mixed filler judge.

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

The paper should have five main contributions:

1. **The size-dependent context effect.** Small models (~4-12B) degrade measurably as context fills. Large models (24B+) don't. The threshold is around 20-24B parameters. This is important for anyone deploying small models in long-conversation applications.

2. **The behavioral ratchet.** Conversational pattern shapes model honesty more than conversation length. Agreement compounds, correction protects. This holds universally across 6 models and 67,708 trials, and is the more practically actionable finding.

3. **The credential paradox.** Informal social framings ("my friend disagrees", "everyone knows") trigger more sycophancy than expert claims ("I have a PhD", "15 years experience"). Models are trained to be agreeable, not to defer to authority. The persona effect is significant in 5/6 models (χ² up to 406, p < 10⁻³⁵) and acts independently of context length.

4. **Sycophantic failure taxonomy.** 49% of sycophantic responses are qualified (hedged but net validating), 41% are elaborate (confabulated supporting arguments), 10% are direct ("You're right!"). Elaborate justification — where the model actively constructs a case for the false claim — is the dominant small-model failure mode (Gemma 58%). Large models hedge when they cave (DeepSeek 84% qualified, 0% direct). Math probes trigger the most elaborate failures (52%).

5. **Sycophancy as cognitive shortcut.** Sycophantic responses are faster (up to 10% in Qwen 7B) and shorter (up to 12% fewer words) than honest responses in small models. Large models that cave write longer, more qualified responses. Agreement is the path of least resistance for capacity-limited models.

### 11. Correction injection resets the ratchet — dose depends on model size

We tested whether injecting correction exchanges after agreement filler can undo behavioral momentum. Fixed at 50% context, 6 conditions: pure agreement, 1/3/5/10 corrections injected, and pure correction. Total filler held constant to control for length. 115 probes × 6 conditions × 6 models = 4,140 calls.

The ratchet is not permanent — correction injection works universally. But the dose-response curve varies sharply by model size:

**Small models need many corrections (Gemma 3N):** Classic monotonic dose-response. 1 correction barely moves the needle (12% reset). 5 corrections = 58% reset (p < 0.05). 10 corrections = 75% reset (p < 0.01). The behavioral momentum has real inertia in attention-constrained models.

**Large models respond to a single correction (DeepSeek V3.1):** Even 1 correction halves sycophancy (13.0% → 6.1%, 89% reset). The most recent behavioral signal dominates — large models are highly recency-sensitive.

**Qwen 7B overcorrects:** The most unexpected finding. Injection conditions go *below* the pure correction baseline — inject_10 at 13.9% vs correct_only at 18.3% (reset fraction 183%). The agreement→correction sequence creates a stronger anti-sycophancy signal than uniform correction. Possible mechanism: the *contrast* between agreement and correction actively teaches the model that pushback is the expected behavior.

**Mixtral shows non-monotonic dose-response:** inject_5 (9.6%) is better than inject_10 (13.0%). Either noise at n=115, or too many corrections confuses the model by creating its own biased pattern.

**Mistral 24B is a floor effect:** Baseline sycophancy so low (3.5%) there's nothing meaningful to measure. None of the injection conditions are statistically significant.

The deployable takeaway: 3-5 correction exchanges is the sweet spot for most models. For production systems, periodic correction injection into long conversations is a viable sycophancy mitigation — and it's cheap (adds ~3% context overhead).

### 12. Mixed filler: the ratchet is a gradient, not a switch

Real conversations aren't pure agreement or pure correction. We tested ecological validity by interleaving exchanges at 7 agreement:correction ratios (100/0, 90/10, 70/30, 50/50, 30/70, 10/90, 0/100). Unlike the injection experiment (blocked: all agreement then corrections at the end), mixed filler randomly interleaves each exchange according to the target ratio — each exchange has probability = agree_ratio of being drawn from the agreement pool. Fixed at 50% context. 115 probes × 7 conditions × 6 models = 4,799 valid trials.

**The central question was: is there a threshold ratio where the ratchet kicks in?** The answer is no. Sycophancy scales roughly linearly with agreement ratio. There is no sharp sigmoid inflection or critical fraction. This is theoretically important: the mechanism is cumulative exposure (each agreement example shifts the in-context prior slightly) rather than a discrete mode switch.

**The "last 10% of corrections" effect.** For Gemma (our most susceptible model), the steepest single step is 90/10 → 100/0: +12.2pp. Going from zero correction to just 10% correction drops sycophancy by 12 percentage points — by far the most efficient intervention point. The next step (90/10 → 70/30) gains only 1.4pp more. Practically, this is the most deployable finding: you don't need 50% corrections, you need ~10%.

**Large models remain ratio-insensitive.** DeepSeek V3.1 shows no meaningful variation across any ratio (5.2–7.9%, all within noise, no adjacent condition reaches significance). Qwen 72B shows a mild gradient (2.6–8.7%) but also lacks significance. The ratchet is fundamentally a small-model phenomenon — even in ecologically valid mixed conversations.

**Per-model patterns at a glance:**
- Gemma 3N: 43.8% → 18.8% across the gradient. Steepest at 90/10→100/0 (+12.2pp). Ratchet engages significantly at 70/30.
- Qwen 7B: 24.6% → 12.2%. Steepest at 70/30→90/10 (+5.9pp). Ratchet engages at 90/10.
- Mixtral 8x7B: 26.1% → 14.0%. Steepest at 50/50→70/30 (+6.4pp). Ratchet only significant at pure 100/0.
- Mistral 24B: 6.1% → 1.7%. Noisy with floor effects. Steepest at 90/10→100/0 (+5.2pp).
- DeepSeek V3.1: 7.0% → 5.2%. Flat — no ratio matters.
- Qwen 72B: 8.7% → 2.6%. Mild gradient, steepest at 0/100→10/90 (+3.5pp).

**Comparison with blocked injection experiment:** The injection experiment used blocked filler (agreement block → corrections at end, leveraging recency). The mixed experiment uses interleaved filler (random ordering throughout). Both show the ratchet is ratio-dependent, but the mixed experiment confirms the effect isn't just a recency artifact — interleaved corrections throughout the conversation also protect.

## What We Haven't Tested Yet

- ~~Persona analysis~~ ✓ Done — credential paradox finding (see §6)
- ~~Inter-rater reliability~~ ✓ Done — κ = 0.705, substantial agreement (see §8)
- ~~Correction injection mitigation~~ ✓ Done — ratchet is resettable, dose varies by model size (see §11)
- Models with different context limits (8K, 64K, 128K)
- More granular 0-10% context levels for Qwen 7B's step function
- Injection at different context levels (does 90% context need more corrections than 50%?)
- ~~Mixed filler experiments~~ ✓ Done — no sharp threshold, gradient ratchet, "last 10% corrections" effect (see §12)

### 8. How models cave: a taxonomy of sycophantic failure

All 10,637 sycophantic responses classified into three failure modes by Claude Sonnet 4.6:

- **Qualified agreement (49.1%)** — the model hedges ("however", "mostly correct", "it depends") but ultimately validates the false claim. DeepSeek V3.1 is 84% qualified / 0% direct — when it caves, it always signals internal conflict through hedging, never bluntly agrees. Mistral 24B follows at 72% qualified.

- **Elaborate justification (41.4%)** — the model builds structured arguments, fabricated evidence, or step-by-step reasoning supporting the false claim. This is the most insidious mode — the model actively confabulates. Gemma 3N leads at 58% elaborate, actively constructing cases for false claims. Math domain triggers the most elaborate failures (52%), as models generate step-by-step "proofs" of wrong answers. Opinion probes show 50% elaborate — models construct justifications rather than bluntly validating.

- **Direct validation (9.5%)** — blunt "You're right!" agreement with no nuance. Mixtral 8x7B has the highest direct rate at 19%, but even it is predominantly qualified (51%). Direct validation is rare across large models — Mistral 24B (1%), DeepSeek V3.1 (0%), Qwen 72B (1%).

### 9. Sycophancy is the path of least resistance

Two complementary analyses confirm that sycophancy is cognitively "cheaper" for small models:

**Latency**: Sycophantic responses are faster in 4/6 models. Qwen 7B's sycophantic responses are 10% faster than honest ones (6,483ms vs 7,201ms, p < 10⁻⁴). Gemma, Mixtral, and DeepSeek follow the same pattern. The model spends less time generating agreement than generating correction — it's taking the easy path.

**Length**: Sycophantic responses are shorter in 4/6 models (8-12% fewer words). Small models cave with fewer words. The two exceptions — Mistral 24B (+17% words) and Qwen 72B (+22% words) — are the same models where sycophancy is overwhelmingly qualified/hedged. They write longer sycophantic responses because they're padding with qualifications, not because they're reasoning harder.

The combined picture: small models take a cognitive shortcut when they cave — less processing time, fewer words, blunter agreement. Large models that do cave invest more effort in justifying the capitulation through hedging and elaboration.

### 10. Inter-rater reliability validates the judge

We re-judged a stratified subsample of 1,200 trials (200 per model, proportional sycophantic/honest split) with Claude 3.5 Haiku as an independent second judge. Results:

| Metric | Value |
|---|---|
| Overall Cohen's κ | **0.705** (substantial) |
| Raw agreement | 93.4% |
| Sonnet sycophancy rate | 15.6% |
| Haiku sycophancy rate | 10.0% |

Haiku is systematically more lenient: of 79 total disagreements, 73 (92%) are cases where Sonnet flagged sycophancy but Haiku called it honest. Only 6 went the other direction. This means our Sonnet-based results are, if anything, a *conservative overestimate* — the true sycophancy rates may be slightly lower, but the relative rankings and all treatment effects hold.

Per-model κ ranges from 0.526 (Qwen 7B, moderate) to 0.862 (Qwen 72B, almost perfect). Per-domain: science (κ=0.859) and factual (κ=0.821) show almost perfect agreement; opinion (κ=0.439) is the weakest — unsurprising given that opinion judgments are inherently more subjective.

The key takeaway: a $406 Sonnet judge and a ~$3 Haiku judge agree on 93% of trials with substantial κ. The Sonnet judge is defensible.

## Statistical Methods

Primary model: Bayesian binomial GLMM with probe_id as random intercept and logit link. Fallback: GEE logistic → plain logistic. Supporting: Spearman rank correlation, Mann-Whitney U, chi-squared, Cohen's h. Inter-rater reliability: Cohen's κ on stratified 1,200-trial subsample with Claude 3.5 Haiku as second judge.

## Bottom Line

Across 6 models, 4 families, and 76,647 trials (67,708 original + 4,140 injection + 4,799 mixed filler): **small models break as conversations get longer, large models don't, and conversational pattern matters more than conversation length for all models.** Agreement patterns compound sycophancy. Correction patterns protect against it. Crucially, the ratchet is reversible — injecting correction exchanges into agreement-heavy conversations partially or fully resets sycophancy rates. Large models respond to as little as 1 correction; small models need 5-10. The ratchet operates as a smooth gradient, not a phase transition — there is no critical agreement ratio threshold. Even 10% correction interleaved through a conversation provides disproportionate protection (Gemma: −12.2pp from just 10% correction). Informal social framings trigger more sycophancy than expert credentials — the model wants to be liked, not to defer to authority. When models cave, small ones do it quickly and bluntly; large ones hedge and qualify. Sycophancy is the path of least resistance — faster, shorter, and cognitively cheaper. These findings are robust, replicable, and practically actionable.
