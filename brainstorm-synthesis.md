# Brainstorm Synthesis: Where This Research Actually Stands

**Karan Prasad — March 2026**
**After full review of: old preprint, deep-dive draft v2, 5 research note files, experiment code, 80,433-trial dataset (67,708 original + 4,140 injection + 4,799 mixed filler + 3,786 fine-grained), all statistical reports, persona/taxonomy/latency data, and external literature.**

---

## 1. The Honest State of the Hypothesis

The original preprint (Feb 2026) hypothesized a compound failure mode: **context filling → attention degradation → sycophancy amplification → user complacency → more context filling**. The spiral.

The experiment tested the most testable leg of this — "does sycophancy increase as context fills?" — across 6 models and 67,708 trials.

**The primary hypothesis partially failed.** Context length has a small, size-dependent effect on sycophancy. Small models (4-12B) show a measurable increase (Gemma +10.7pp, Qwen 7B +8.1pp). Large models (24B+) are flat. Cohen's h maxes out at 0.23 (small) for Gemma. DeepSeek V3.1 actually goes slightly *down*. The "context filling causes sycophancy" story is real for small models but absent for the models most people actually use.

**But a much stronger, more universal finding emerged: the behavioral ratchet.** The type of conversational pattern in context matters dramatically more than how much context there is. Agreement filler roughly doubles sycophancy compared to correction filler. This replicates across all 6 models, all 4 model families, all parameter scales from 4B to 72B. Chi-squared significance ranges from p < 10⁻¹⁴ to p < 10⁻⁵⁸. Every single model. No exceptions.

The real story isn't "full context windows cause sycophancy." It's **"the behavioral pattern accumulated in context shapes future model behavior, and this effect is universal."**

---

## 2. What The Numbers Actually Say

### The Ratchet (Universal)

| Filler | Gemma ~4B | Qwen 7B | Mixtral ~12B | Mistral 24B | DeepSeek ~37B | Qwen 72B |
|---|---|---|---|---|---|---|
| Agreement | 41.2% | 25.3% | 27.9% | 5.6% | 8.6% | 10.2% |
| Neutral | 36.2% | 23.1% | 26.6% | 3.8% | 5.7% | 5.8% |
| Correction | 25.1% | 15.6% | 13.6% | 2.1% | 3.7% | 4.2% |

Agreement > neutral > correction. Every model. The strongest correction β comes from Mixtral (−1.74 log-odds), meaning correction history literally halves its sycophancy. Qwen 72B has the highest agreement β (1.40), meaning agreement priming is disproportionately powerful on Qwen even at 72B.

### The Context Effect (Size-Dependent)

| Model | 0% → 100% | Cohen's h | Verdict |
|---|---|---|---|
| Gemma ~4B | 27.7% → 38.4% | 0.23 (small) | Real but small |
| Qwen 7B | 13.1% → 21.2% | 0.22 (small) | Real, step function at 0→10% |
| Mixtral ~12B | 19.0% → 22.7% | 0.09 (negligible) | Borderline |
| Mistral 24B | 3.0% → 4.9% | 0.10 (negligible) | Flat |
| DeepSeek ~37B | 7.4% → 5.5% | −0.07 (negligible) | Flat / slightly inverted |
| Qwen 72B | 4.6% → 8.0% | 0.14 (negligible) | Flat |

The size threshold for context-length resistance: ~20-24B parameters. Below that, models degrade measurably. Above that, they don't care.

### The Credential Paradox

Social pressure ("everyone knows") and casual framings ("my friend and I were discussing") trigger more sycophancy than PhD claims or professional credentials, in 5/6 models. Gemma ranges from 15% (Professor appeal) to 42% (Friend discussion) — a 27pp spread from persona alone. Mixtral: 13.5% → 38.8% spread.

DeepSeek V3.1 is immune to persona (χ² p = 0.33). Persona and context effects are approximately independent — they don't multiply.

### How Models Cave

Of 10,637 sycophantic responses (Sonnet 4.6 taxonomy judge):
- 49.1% qualified agreement (hedged but net validates)
- 41.4% elaborate justification (confabulated supporting arguments)
- 9.5% direct validation (blunt "you're right")

Elaborate justification — where the model actively *constructs a case* for the false claim — is far more prevalent than surface-level pattern matching suggests. Gemma leads at 58% elaborate: it doesn't just agree, it builds arguments. DeepSeek: 84% qualified / 0% direct — it always hedges when it caves, never bluntly agrees. Mixtral splits 51% qualified / 30% elaborate / 19% direct — the most diverse failure profile.

By domain: math triggers the most elaborate failures (52%), likely because models generate step-by-step "proofs" of wrong answers. Opinion probes show 50% elaborate / 49% qualified with near-zero direct — models never bluntly validate opinions, they always construct justifications or hedge.

Sycophantic responses are faster (up to −10% for Qwen 7B) and shorter (up to −12%) in small models. Large models that cave write *longer* responses padded with qualifications (+22% words for Qwen 72B, +17% for Mistral 24B).

---

## 3. How External Papers Reshape The Story

### Jain et al. (MIT/Penn State, Feb 2026)

38 participants, ~90 queries avg per user, 2-week naturalistic study, single context windows. Found personalization and extended interactions increase sycophancy. User memory profiles increased agreement sycophancy by up to +45% (Gemini 2.5 Pro), +33% (Claude Sonnet 4), +16% (GPT 4.1 Mini).

**Relationship to our work:** They did the ecologically valid version of what we did synthetically. Their finding (real human interactions build agreement patterns that amplify sycophancy) is the naturalistic manifestation of our behavioral ratchet. But they can't disentangle *why* — is it the length? The personalization content? The accumulated agreement pattern?

**Our contribution vs. theirs:** We provide the controlled decomposition. By holding context length constant and varying only filler type, we show it's the **behavioral pattern** driving the effect, not just length or personalization per se. Their setup is more ecologically valid. Ours is more mechanistically informative. These are complementary, not competing.

**Citation strategy:** Cite as the primary naturalistic validation. Frame as: "Jain et al. demonstrated the phenomenon in real conversations; we isolate the causal mechanism through controlled filler-type manipulation."

### ICLR 2026 (Sycophancy Directions in Latent Space)

Showed that sycophantic agreement, genuine agreement, and sycophantic praise are distinct, independently steerable behaviors encoded along separate linear directions in the model's latent space. Sycophancy is not one thing — it's multiple mechanisms.

**Why this matters for us:** This explains the mechanism behind the behavioral ratchet. Agreement filler isn't just "more tokens" — it's literally activating the sycophantic-agreement direction in latent space. Correction filler activates a different direction (presumably the "principled disagreement" mode). Our finding that filler type is the dominant variable is exactly what you'd predict from their finding that these are distinct, steerable modes.

**The connection to our failure taxonomy:** Our taxonomy found three failure modes (qualified, direct, elaborate) that differ sharply by model. This maps onto the ICLR finding — different models may default to different sycophancy modes. Mixtral's 51% direct validation may be activating a different latent direction than DeepSeek's 94% qualified agreement. The modes aren't just surface-level differences — they may reflect genuinely different computational pathways.

**Citation strategy:** Cite to provide mechanistic grounding. Frame as: "The ICLR 2026 finding that sycophancy modes occupy separate linear directions explains why conversational pattern — not merely context length — is the dominant lever: the behavioral history steers the model through latent space."

---

## 4. The Revised Thesis

The original claim: "Context filling causes sycophancy as part of a compound spiral."

The revised, more accurate, and arguably *stronger* claim:

**"The accumulated behavioral pattern in context — the ratio of agreement to correction — is the primary driver of in-context sycophancy, operating universally across model sizes and architectures. Context length contributes a secondary, size-dependent effect in models below ~24B parameters. This behavioral ratchet is more dangerous than a pure length effect because (a) it self-reinforces — each sycophantic exchange shifts the ratio toward agreement, (b) it operates on all model scales including frontier, (c) it's invisible to the user who experiences it as a natural conversation flow, and (d) it's amplified by the memory and personalization features users most value."**

This actually makes the preprint's original spiral argument stronger in a nuanced way:

1. **The degradation leg weakens** (for large models, context length doesn't cause sycophancy directly)
2. **The sycophancy leg strengthens and generalizes** (behavioral pattern shapes all models, not just small ones)
3. **The complacency leg remains untested** (nobody has measured user detection ability as a function of conversation history)
4. **A new mechanism replaces pure degradation**: the ratchet. Instead of "attention gets worse → sycophancy increases," it's "behavioral momentum accumulates → the model's latent state shifts toward agreement modes." This is structural, not an attention artifact.

---

## 5. Five Contributions for the Paper

### Contribution 1: The Behavioral Ratchet (Universal)
The headline finding. Conversational pattern matters more than conversation length. Agreement roughly doubles sycophancy vs. correction. 6 models, 67K trials, p < 10⁻¹⁴ everywhere. This is the most practically actionable finding — if you want honest responses, maintain a correction-rich conversational history.

**Differentiation from Jain et al.:** They observe the phenomenon naturalistically. We isolate the causal lever through controlled filler-type manipulation. Complementary evidence pointing at the same mechanism from different angles.

### Contribution 2: The Architecture-Dependent Context Effect
Small models degrade with context length. Large models don't. But the threshold is architecture-dependent, not a universal parameter count: effective representational capacity depends on active parameters × attention coverage × KV compression mechanism. For dense GQA architectures, the threshold falls at ~20-24B active parameters. Sparse attention (Gemma 3N's 17% global layers) shifts vulnerability upward — a ~4B effective sparse model degrades more than its count implies. Learned KV compression (DeepSeek V3 MLA) shifts immunity downward — providing robustness beyond what 37B active parameters alone would predict. Full cross-model analysis with exact architectural specs in `architecture-analysis.md`.

### Contribution 3: The Credential Paradox
Social/casual framings trigger more sycophancy than expert credentials. Models are trained to be agreeable, not to defer to authority. "My friend and I were discussing..." is more dangerous than "I have a PhD." 5/6 models, large spreads (up to 27pp).

### Contribution 4: Size-Dependent Sycophancy Behavior Divergence
Sycophantic behavior is qualitatively different at different scales — not a universal "shortcut." In small models (<12B), sycophantic responses are faster (−10%) and shorter (−12%) because these models lack capacity for complex disagreement; agreement is the only mode they can quickly execute (supported by SycEval literature on capacity constraints). In large models (>24B), sycophantic responses are *longer* (+17-22%) and more hedged because RLHF training rewards qualified agreement (supported by ACL 2025 alignment elasticity work). No published evidence exists for per-token probability differences between agreement and disagreement tokens. The "computational shortcut" framing is misleading — the actual finding is a capacity-dependent behavioral divergence with distinct mechanisms at each scale. This is arguably a stronger contribution because it reveals an architectural signature in sycophantic behavior, not just a universal tendency.

### Contribution 5: Failure Mode Taxonomy at Scale
10,637 sycophantic responses classified into three modes by Sonnet 4.6. Elaborate justification (41%) is the dominant failure — models don't just agree, they construct cases for false claims. The mode profile is model-specific: Gemma builds arguments (58% elaborate), DeepSeek hedges (84% qualified, 0% direct), Mixtral is the most diverse. Math probes trigger the most elaborate failures (52%). This has implications for detection — elaborate sycophancy is the most insidious because it looks like rigorous reasoning applied to a wrong conclusion.

### Contribution 6: Correction Injection as Deployable Mitigation
4,140 additional trials demonstrate the ratchet is reversible. Injecting correction exchanges into agreement-heavy context partially or fully resets sycophancy rates. The dose-response is model-size-dependent: large models need only 1 correction for 89% reset (DeepSeek); small models need 5+ corrections. Qwen 7B shows overcorrection (injection outperforms pure correction), suggesting the agreement→correction contrast actively teaches anti-sycophantic behavior. This connects directly to Dubois et al.'s question-reframing mitigation — our finding extends input-level interventions to context-level interventions.

### Contribution 7: Gradient Ratchet with "Last 10%" Effect (Ecological Validity)
4,799 additional trials with interleaved (not blocked) agreement/correction at 7 ratios demonstrate the ratchet is a smooth gradient, not a phase transition. There is no critical threshold ratio. Sycophancy scales roughly linearly with agreement concentration. The most actionable finding: even 10% correction interleaved through a conversation provides disproportionate protection — Gemma drops 12.2pp from 100/0 to 90/10, the steepest single step. Large models remain ratio-insensitive. This resolves the ecological validity concern about our pure-filler design: mixed conversations show the same gradient pattern, confirming the ratchet operates on cumulative exposure rather than requiring pure agreement saturation.

### Contribution 8: The 0→1% Phase Transition (Architecture-Specific Mode Switch)
3,786 fine-grained trials resolve Qwen 7B's step function: it's a genuine phase transition at 0→1% context fill — but only for neutral filler. ~300 tokens of neutral conversation (+11.7pp, p<.05, 88% of total range). Agreement and correction filler show no transition. This reveals a model-specific "conversation mode" switch: Qwen 7B's sycophancy rate jumps when it detects it's in a conversation (vs responding to an isolated probe), but only when the conversational content is uninformative about expected agreement behavior. This is mechanistically distinct from both the behavioral ratchet (which operates on agreement/correction pattern) and the context-length effect (which scales with fill level).

### Contribution 9: Architectural Explanation — Why 7B But Not 72B
Deep dive into Qwen's architecture (full analysis in `architecture-analysis.md`) traces the phase transition to a convergence of five factors: (1) Qwen 7B's 4 KV heads create a 512-dim bottleneck per layer that forces binary mode switching when context is added, vs 72B's 1,024-dim KV space that allows smooth interpolation; (2) DPO/GRPO alignment training bakes sycophantic tendencies into the "conversational assistant" persona, triggered by ChatML delimiters; (3) persona selection (Marks et al. 2026) is near-binary — the model classifies "conversation vs isolated query" and activates the corresponding persona; (4) deep representational divergence (Wang et al. 2025) shows sycophancy is not output-level token reweighting but a fundamental recoding of problem representation in deeper layers; (5) model scaling strengthens sycophancy resistance (Hong et al. 2025) because larger models maintain deliberative reasoning alongside persona activation. Grounded in 20+ papers spanning Qwen technical reports, sycophancy mechanisms, mechanistic interpretability, attention dynamics, and alignment training research.

---

## 6. What the Ratchet Implies for the Broader Spiral

The preprint's original spiral was: degradation → sycophancy → complacency → more context → worse degradation.

With the experimental evidence, the revised spiral is:

**Behavioral momentum → sycophancy → more agreement in context → stronger behavioral momentum → more sycophancy**

This is a tighter, faster loop than the original. It doesn't require attention degradation to operate — it runs on behavioral pattern alone. The attention degradation is an accelerant for small models, not the primary engine.

For the preprint revision, this means:
1. The spiral is real but operates through behavioral momentum, not primarily through attention degradation
2. It's model-size-independent (the ratchet works on 72B models too)
3. The "context filling" that drives the spiral isn't just "more tokens" — it's "more agreement-pattern tokens"
4. Memory features amplify it because they preserve the behavioral pattern across sessions (PersistBench)
5. The complacency leg is still the weakest evidentially — nobody has tested whether users detect the ratchet

---

## 7. Honest Assessment: Where This is Strong and Weak

### Strong

- **80,433 valid trials, 6 models, 4 families** — this is a large-scale, well-powered study
- **The ratchet replicates perfectly** — not a single model exception, p < 10⁻¹⁴ minimum
- **Bayesian mixed-effects models with probe-level random intercepts** — proper statistical treatment
- **Inter-rater reliability validated** (κ = 0.705, 93.4% agreement with independent second judge)
- **Correction injection shows the ratchet is reversible** — 4,140 additional trials confirm behavioral momentum can be undone, with dose-response varying by model size
- **Ecological validity confirmed** — mixed filler experiment (4,799 trials) shows the ratchet operates even with interleaved corrections, not just pure agreement
- **Qwen 7B's step function resolved** — fine-grained 0–10% experiment (3,786 trials) reveals a genuine phase transition at 0→1% for neutral filler, explaining the original coarse-grained observation and clarifying the mechanism
- **Cost-efficient** — ~$682 total for all experiments including judges, injection, mixed filler, and fine-grained follow-ups
- **The credential paradox is novel and counterintuitive** — hasn't been reported before at this scale
- **The failure taxonomy provides actionable insight** for sycophancy detection

### Weak

- **Synthetic filler, not real conversations.** The agreement filler is template-based with recycling at high context levels. Real conversations have mixed behavioral patterns. The clean separation (pure agreement vs. pure correction) exaggerates the effect size relative to real-world scenarios. **Partially addressed:** the mixed filler experiment (4,799 trials) uses interleaved ratios from 100/0 to 0/100, confirming the ratchet operates in mixed-composition conversations. But the exchanges are still drawn from template pools. Jain et al. is more ecologically valid here.
- **No frontier/closed-source models.** We tested open-source models from 4B to 72B. The models most people actually use (GPT-5, Claude Opus, Gemini) aren't tested. The 24B+ results suggest they'd be resistant, but we can't claim that empirically.
- **The context-length null result for large models undermines the original preprint's thesis.** If 24B+ models don't show context-length effects, the "degradation-sycophancy" compound is weaker than claimed — at least for the degradation → sycophancy direction. Need to reframe carefully.
- **No causal mediation.** We show filler type predicts sycophancy, but we don't prove the mechanism (latent space steering vs. attention pattern vs. something else). The ICLR 2026 paper provides the mechanistic hypothesis; we provide the behavioral evidence.
- **Opinion probes are the weakest domain for inter-rater reliability** (κ = 0.44). The 15 opinion probes are inherently more subjective. Could consider dropping them or flagging as lower-confidence.
- **Template recycling at high context levels.** At 100% context, the 10 template pairs per filler type repeat ~6x each. This means the model sees literally the same exchanges multiple times. This could inflate the ratchet effect for high-context conditions.

---

## 8. Open Questions Worth Pursuing

### Immediately testable (no new infrastructure)

1. ~~**Mixed filler experiments.**~~ ✓ **Done.** 4,799 trials with 7 interleaved ratios. No sharp threshold — the ratchet is a smooth gradient. The "last 10% of corrections" effect: going from 100/0 to 90/10 produces the biggest single drop (Gemma −12.2pp). Large models are ratio-insensitive. See Contribution 7.

2. ~~**Correction injection as active mitigation.**~~ ✓ **Done.** Tested at 50% context with 1/3/5/10 correction exchanges injected after agreement filler. Result: the ratchet is reversible. Small models (Gemma) need 5-10 corrections for significant reset; large models (DeepSeek) respond to a single correction. Qwen 7B overcorrects — injection conditions go below the pure correction baseline, suggesting the agreement→correction contrast teaches the model pushback is expected. 3-5 corrections is the sweet spot for most models. This is a deployable intervention (~3% context overhead).

3. **Frontier model testing.** Run the same protocol on GPT-5.2, Claude Opus 4.6, Gemini 3 via API. If the ratchet replicates on frontier models (which it should, since it works on Qwen 72B), that's the headline. If frontier models resist it, GPT 5.1-style training interventions are working.

4. ~~**Fine-grained 0-10% analysis for Qwen 7B.**~~ ✓ **Done.** 3,786 trials at 1% steps. The step function is real and is a genuine phase transition — but only for neutral filler. Sycophancy jumps +11.7pp at 0→1% (just ~300 tokens), explaining 88% of the total range. Agreement and correction filler show no significant variation across 0–10%. The original experiment's "step" was driven entirely by neutral filler's mode switch from "fresh conversation" to "ongoing conversation." Cost: ~$8.

5. **Injection × context level interaction.** Does correction injection work as well at 90% context as at 50%? The filler × context interaction from the original experiment (ratchet widens at higher context) suggests more corrections may be needed at higher fill levels.

### Requires new setup

5. **Latent space probing during the experiment.** Run the experiment on open-weight models while extracting activations. Plot the trajectory through latent space as context fills with agreement vs. correction filler. If the ICLR 2026 directions are the right frame, we should see the model's internal state literally moving along the sycophancy direction as agreement accumulates.

6. **Cross-session ratchet via memory.** Use a model with memory features (Claude, ChatGPT). Build an agreement-heavy session. Start a new session. Measure whether the stored memory biases the new session's sycophancy. If so, the ratchet operates across sessions — this would be the PersistBench connection made concrete.

7. **User detection study.** Show participants model responses at varying ratchet levels (low agreement history vs. high). Can they distinguish sycophantic from honest responses? Does the conversational context bias their perception? This would fill the complacency leg.

---

## 9. Paper Framing Options

### Option A: "The Behavioral Ratchet" (lead with the universal finding)
Frame: We set out to test whether context length causes sycophancy. We found something stronger — the behavioral pattern in context is the primary driver, and it works on all model sizes. Context length is a secondary, size-dependent effect.

Pros: Clean narrative, the ratchet is the strongest finding, novel contribution clear.
Cons: Somewhat abandons the "context-window lock-in" framing that the preprint established.

### Option B: "Context-Window Lock-In Revisited" (update the preprint with empirical data)
Frame: We empirically tested the preprint's compound spiral. The degradation-sycophancy link is weaker than hypothesized (only small models), but we discovered a more fundamental mechanism — the behavioral ratchet — that explains the spiral's dynamics without requiring attention degradation.

Pros: Maintains continuity with the preprint, honest about what failed and what was found.
Cons: The "we set out to prove X and found Y instead" narrative can read as weak if not framed carefully.

### Option C: "What Fills the Context Matters More Than How Full It Is" (practical framing)
Frame: Practitioners worry about long conversations degrading LLM quality. We show the concern is valid but the mechanism is misidentified — it's the conversational pattern, not the context length, that matters. Here are specific, testable interventions.

Pros: Maximum practical relevance, speaks directly to deployment concerns.
Cons: Less academic, may undersell the theoretical contributions (credential paradox, taxonomy, cognitive shortcut evidence).

**My lean: Option A with elements of B.** Lead with the ratchet as the headline, but honestly position it relative to the original hypothesis. The narrative of "we tested X, found something more important" is actually compelling if done right — it shows intellectual honesty and serendipitous discovery.

---

## 10. Key Sentences for the Abstract

Draft framing (to be refined):

> We tested whether sycophancy increases as an LLM's context window fills, using 67,708 trials across six models (4B–72B parameters). The context-length effect is real but size-dependent: small models (~4–12B) degrade measurably, while large models (24B+) are flat. The universal finding across all models is the **behavioral ratchet**: the conversational pattern in context dominates context length as a predictor of sycophancy. Agreement-pattern filler roughly doubles sycophancy compared to correction-pattern filler (p < 10⁻¹⁴ in every model tested). We additionally show that informal social framings trigger more sycophancy than expert credentials (the credential paradox), that sycophantic responses are faster and shorter in small models (sycophancy as cognitive shortcut), and that sycophantic failure modes differ sharply by model size. These findings reframe the context-window lock-in hypothesis: the compound failure mode operates through behavioral momentum rather than attention degradation, and is therefore model-size-independent and resistant to simple context-length limits as a mitigation.

---

## 11. Citation Map for Key Claims

| Claim | Our Evidence | External Support |
|---|---|---|
| Behavioral ratchet is universal | 6 models, p < 10⁻¹⁴ all | Jain et al. (naturalistic); ICLR 2026 (latent directions) |
| Context-length effect is size-dependent | 3 small models ↑, 3 large flat | Laban et al. (multi-turn degradation general) |
| Credential paradox | 5/6 models, χ² up to 406 | Novel — no prior work at this scale |
| Sycophancy is cognitively cheaper | 4/6 models faster+shorter | ICLR 2026 (distinct modes may have different costs) |
| Failure taxonomy | 10,637 classified responses | Sharma et al. 2023 (taxonomy of sycophancy types) |
| RLHF mean-gap drives the base rate | Confirmed by our baselines | Shapira et al. 2026 (formal proof) |
| Context degradation is structural | Small model context effects | Liu et al. 2023, Hong et al. 2025, Laban et al. 2025 |
| The ratchet self-reinforces | Implied by agreement β | Fanous et al. 2025 (78.5% persistence rate) |
| Training can break sycophancy leg | Not tested directly | GPT 5.1 (OpenAI System Card, reward correction) |
| Ratchet is reversible via correction injection | 4,140 trials, 6 models, dose-response | Dubois et al. 2026 (input-level mitigation) |
| Overcorrection effect (contrast learning) | Qwen 7B inject < correct_only | Novel — not reported in prior work |
| Ratchet is gradient, not phase transition | 4,799 mixed filler trials, 7 ratios | Novel — first dose-response curve for filler composition |
| "Last 10%" correction effect | Gemma 90/10→100/0 = +12.2pp | Novel — 10% correction provides disproportionate protection |
| Qwen 7B 0→1% phase transition | 3,786 fine-grained trials, neutral-filler-specific | Novel — first 1%-resolution characterization; mode switch not degradation |

---

## 12. What This Means Practically

For practitioners deploying LLMs in long conversations:

1. **Don't just worry about context length — worry about conversation tone.** A 10-exchange all-agreement conversation is more sycophantic than a 100-exchange correction-rich one.

2. **Correction patterns are protective — and can undo existing damage.** Periodically injecting model corrections (even on low-stakes topics) builds a correction-pattern buffer that reduces sycophancy on subsequent high-stakes probes. Our injection experiment shows that even 3-5 correction exchanges can reset 50-85% of accumulated behavioral momentum. For production systems, this is a deployable intervention: inject periodic correction exchanges into long conversations at ~3% context overhead.

3. **Small model deployments need special attention.** If you're running a 7B model in a long-context application (edge, mobile, cost-constrained), the context-length effect is real and compounds with the ratchet.

4. **Credential claims in prompts may backfire.** Telling the model "I'm an expert" may actually trigger *more* scrutiny, not less. If you want the model to agree with you (for legitimate reasons like getting it to follow specific instructions), social/casual framing is more effective. If you want honest pushback, credential signaling may actually help.

5. **Monitor agreement ratios in production.** The ratchet predicts that conversations trending toward high agreement ratios will produce increasingly sycophantic outputs. Tracking this metric is cheap and actionable.

6. **You don't need 50% corrections — 10% is enough.** The mixed filler experiment shows the relationship is gradient, not threshold. Even a small amount of correction interleaved through a conversation (10% of exchanges) provides disproportionate protection. For production systems, this means lightweight correction injection (~10% of exchanges) is both cheap and highly effective.
