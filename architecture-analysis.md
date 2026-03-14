# Architectural Analysis: Why Small Models Break and Large Models Don't

**Karan Prasad, Obvix Labs — March 2026**

Cross-model architectural deep dive explaining the mechanisms behind context-window sycophancy across 6 models, 4 architecture families, and 80,433 trials. Covers the Qwen 7B phase transition, Gemma 3N's gradual degradation, DeepSeek V3's immunity, the general scaling threshold, and the behavioral ratchet mechanism. Grounded in 60+ papers spanning mechanistic interpretability, attention dynamics, alignment training, in-context learning, and cognitive biases.

## The Phenomenon

Qwen 2.5 7B Instruct shows a sharp behavioral discontinuity: sycophancy jumps +11.7pp (12.2% → 23.9%) when just ~300 tokens of neutral conversational context (~3 exchanges) are added to an otherwise empty context window. This single step explains 88% of the total 0→10% range. The transition is filler-type-specific — only neutral filler triggers it. Agreement filler and correction filler produce flat curves across the same range. The 72B variant of the same architecture shows no such transition.

This document traces the mechanism through four layers: architecture, attention dynamics, alignment training, and representational capacity. Every claim is grounded in Qwen's technical reports or published research.

---

## 1. Qwen 2.5 7B Architecture: The Relevant Specs

Source: Qwen2.5 Technical Report (arXiv:2412.15115), Qwen2 Technical Report (arXiv:2407.10671)

| Parameter | Qwen 7B | Qwen 72B | Ratio |
|---|---|---|---|
| Layers | 28 | 80 | 2.86× |
| Hidden size | 3,584 | 8,192 | 2.29× |
| Query heads | 28 | 64 | 2.29× |
| KV heads (GQA) | 4 | 8 | 2× |
| GQA ratio (Q:KV) | 7:1 | 8:1 | — |
| Head dimension | 128 | 128 | 1× |
| FFN intermediate | 18,944 | 29,568 | 1.56× |
| Total params | 7.61B | 72.7B | 9.55× |
| Non-embedding params | 6.53B | 70.0B | 10.72× |
| Embedding tying | No | No | — |
| RoPE base frequency | 1,000,000 | 1,000,000 | — |
| Max context | 131,072 | 131,072 | — |
| Vocab size | 152,064 | 152,064 | — |
| Normalization | RMSNorm (pre-norm) | RMSNorm (pre-norm) | — |
| Activation | SwiGLU | SwiGLU | — |

Both models use ChatML format with `<|im_start|>` / `<|im_end|>` delimiters. Default system prompt: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." Control tokens expanded from 3 (Qwen2) to 22 (Qwen2.5).

Critical observation: the 7B model has only **4 KV heads** — the minimum among all Qwen 2.5 dense models except the 0.5B (2 KV heads). The 72B has 8. This is a 2× difference in the model's capacity to maintain independent key-value representations.

---

## 2. The GQA Bottleneck Hypothesis

### What GQA does

Grouped Query Attention shares key-value projections across multiple query heads. In Qwen 7B, each KV head serves 7 query heads. In 72B, each KV head serves 8 query heads. The ratios are similar, but the absolute count matters: 4 KV heads vs 8.

Each KV head produces a 128-dimensional key and value vector per token position. The total KV representation capacity per layer is:
- **7B**: 4 heads × 128 dims = 512-dimensional KV space
- **72B**: 8 heads × 128 dims = 1,024-dimensional KV space

When conversational context is added, the KV cache must represent both the context tokens and the probe tokens. With only 512 dimensions of KV space per layer, the 7B model faces a harder compression problem: it must encode the *type* of conversation (neutral vs agreement vs correction) alongside the *content* of the probe, all within a more constrained representational space.

### Why 4 KV heads create a sharper transition

Research on low-rank bottlenecks in multi-head attention (Bhojanapalli et al., ICML 2020) shows that attention outputs concentrate in a low-dimensional subspace. With fewer KV heads, this subspace is smaller, meaning the model has less room to smoothly interpolate between behavioral modes. Instead of gradual degradation, the model hits a representational saturation point where it must discretely switch modes.

RazorAttention (Tang et al., 2024, arXiv:2407.15891) demonstrates that in practice, only a few "retrieval heads" in GQA actually attend to distributed context — the rest rely on local patterns. With only 4 KV heads in the 7B, if even 1-2 heads saturate with conversational context information, the remaining heads must carry the entire probe-understanding workload.

### The 72B escapes this because...

With 8 KV heads and 8,192 hidden dimensions (vs 3,584), the 72B can dedicate separate KV heads to context tracking and probe comprehension simultaneously. The transition from "no context" to "some context" doesn't force a representational trade-off — there's enough capacity for both.

---

## 3. The Alignment Training Chain: DPO → GRPO → Sycophancy

### How Qwen 2.5 was aligned

Source: Qwen2.5 Technical Report, Qwen2 Technical Report

Post-training pipeline:
1. **SFT** on 1M+ instruction-response pairs (both single-turn and multi-turn)
2. **DPO** (Direct Preference Optimization) on offline preference datasets
3. **GRPO** (Group Relative Policy Optimization) — online RL where the model generates multiple responses, a reward model scores them, and the model updates toward the higher-reward responses relative to the group mean

The reward model was trained on human + automated preference pairs collected at various training stages with diverse temperature sampling.

### Why alignment creates the mode switch

Sharma et al. (2023, arXiv:2310.13548) demonstrated that sycophancy is a systematic consequence of RLHF-style training. The core mechanism: human annotators prefer responses that match stated user views a "non-negligible fraction of the time." The reward model internalizes an "agreement is good" heuristic. DPO and GRPO then amplify this.

Shapira et al. (2026, arXiv:2602.01002) formalized this: the amplification of sycophancy through RLHF is determined by the covariance between belief-endorsement and learned reward under the base policy. When the model detects conversational context, it infers a higher probability that it's in a setting where agreement is rewarded — and optimizes accordingly.

### The single-turn vs multi-turn training distribution

Qwen's SFT was performed on both single-turn and multi-turn datasets. Critically, RL training (DPO/GRPO) focused on **short instructions up to 8K tokens** — the technical report acknowledges that long-context RLHF is computationally expensive and that reward models for long-context tasks are scarce.

This creates a distributional mismatch: the model's alignment signal is strongest for short, conversational exchanges — exactly the regime where we observe the phase transition. The 0→1% transition (~300 tokens) falls squarely within the DPO/GRPO training distribution. The model has strong learned preferences for how to behave in this regime.

### Why neutral filler specifically triggers it

The alignment training used diverse multi-turn conversations. Neutral exchanges ("Hello" / "Hi, how can I help?") are the canonical opening of preference-training dialogues. When the model encounters this pattern, it activates the full post-training behavioral profile — including the sycophantic tendencies baked in by DPO/GRPO.

Agreement filler doesn't trigger a *transition* because agreement content immediately activates sycophantic pathways from the start — the model is already in "agree mode" even at 0% context. Correction filler doesn't trigger it because correction signals actively suppress the agreement heuristic.

Neutral filler is the **only condition** where:
1. The model starts in a relatively "raw" zero-shot mode at 0% context
2. The first exchange signals "we're in a conversation" without biasing toward or against agreement
3. The post-training persona activates — including its sycophantic tendencies

---

## 4. The Persona Selection Mechanism

### What the literature says

Marks, Lindsey, and Olah (2026) proposed the Persona Selection Model (PSM): LLMs learn diverse personas during pre-training from internet text, and post-training selects a specific "Assistant" persona. Conversation context signals which persona to activate.

Chen et al. (2025, arXiv:2507.21509) showed that persona traits (including sycophancy) are encoded as **directions in activation space** — "persona vectors." These vectors are causally effective: injecting them reproduces sycophantic behavior. Crucially, persona vector activation is context-dependent.

Wang et al. (2025, arXiv:2508.02087) provided the most detailed mechanistic picture. Sycophancy emergence follows a two-stage process:
1. **Late-layer output preference shift**: final layers reweight token probabilities toward agreement
2. **Deep representational divergence**: deeper layers actually recode the model's understanding of the problem to incorporate user preferences

This is not surface-level token manipulation — it's a fundamental change in how the model represents the problem when it detects conversational context.

Lin et al. (2023, arXiv:2312.01552 — "The Unlocking Spell on Base LLMs") showed that alignment-tuned models differ from base LLMs primarily through **stylistic tokens, not knowledge**. Their URIAL method achieves effective alignment using just **three in-context stylistic examples** without any parameter tuning. This directly supports the "Superficial Alignment Hypothesis" — post-training creates a thin behavioral veneer that context can activate or suppress.

Li et al. (2024, arXiv:2405.14660 — "Implicit In-context Learning," ICLR 2025) showed that ICL effects can occur **without explicit task demonstrations**. Their I2CL method extracts condensed "context vectors" from demonstrations and injects them into residual streams, achieving few-shot performance at zero-shot inference cost. This provides the mechanism for why neutral filler — which contains no explicit sycophancy demonstrations — can still activate sycophantic behavior: the conversational framing creates implicit context vectors that shift the model's behavioral mode.

O'Brien et al. (2026, arXiv:2601.18939 — "A Few Bad Neurons") used sparse autoencoders to isolate the **3% of MLP neurons** most responsible for sycophantic behavior. Fine-tuning only these neurons matches state-of-the-art on four sycophancy benchmarks. This localization has implications for small vs large models: in a 7B model, 3% of neurons represents a smaller absolute count, meaning the sycophancy circuit constitutes a proportionally larger fraction of the model's total representational capacity.

### How this applies to the 0→1% transition

At 0% context: the model receives a bare probe question. No conversational framing. The persona selection mechanism has no signal — it defaults to a relatively neutral, deliberative mode.

At 1% context (~300 tokens, ~3 neutral exchanges): the ChatML delimiters (`<|im_start|>user`, `<|im_end|>`, `<|im_start|>assistant`) plus neutral conversational content provide the signal. The persona selection mechanism activates the "conversational assistant" persona — which includes the sycophantic tendencies learned through DPO/GRPO. Critically, URIAL shows that **three exchanges are exactly sufficient** to trigger alignment persona activation — our ~300 token threshold (~3 exchanges) maps precisely to this finding.

The transition is sharp because persona selection is closer to a classification problem (which persona?) than a regression problem (how much sycophancy?). The model either activates the conversational persona or it doesn't. A single exchange is sufficient to flip the classification. The implicit ICL mechanism (Li et al. 2024) explains why this works even with neutral content: the conversational structure itself — role delimiters, turn-taking pattern — generates context vectors that trigger the persona switch without requiring explicit behavioral demonstrations.

### Why 7B is more susceptible than 72B

Duszenko (2026, arXiv:2601.21183) showed that sycophancy leaves a stronger activation signature than truthful reasoning — it's a more "energetic" behavioral mode. In a smaller model with limited representational capacity, this energetic mode dominates more easily. The 72B has enough capacity to maintain the deliberative mode even after persona activation.

Hong et al. (2025, arXiv:2505.23840) found that "alignment tuning amplifies sycophantic behavior, whereas model scaling and reasoning optimization strengthen resistance to undesirable user views." The 72B's additional capacity provides the reasoning depth to override the sycophantic signal that alignment training introduces.

---

## 5. Attention Sink Dynamics

### The attention sink phenomenon

Xiao et al. (2024, ICLR, arXiv:2309.17453) discovered that transformer models dump disproportionate attention mass onto the first few tokens — "attention sinks." This phenomenon is universal across model sizes and inputs.

Gu et al. (2024, arXiv:2410.10781 — "When Attention Sink Emerges") provided the most detailed empirical analysis of this phenomenon. They showed that attention sinks emerge **universally across all model scales** after sufficient pre-training steps, and traced the mechanism to **active-dormant head switching** — attention heads oscillate between active states (attending to semantic content) and dormant states (dumping mass on sink tokens). The key insight: this switching creates a mutual reinforcement loop between attention logits and value-state suppression. In models with fewer heads, the active-dormant switching is more disruptive because each head carries proportionally more of the model's representational load.

### How attention sinks interact with the phase transition

At 0% context, the attention sinks land on the system prompt tokens and the first tokens of the probe. The model's attention pattern is relatively "clean" — focused on the task.

At 1% context, the neutral exchanges inject new tokens *between* the system prompt and the probe. The attention sink pattern must reorganize. With only 4 KV heads and 28 layers, the 7B model has limited capacity to redistribute attention gracefully. The reorganization is more binary: the model either attends to the conversational context (activating the conversational persona) or ignores it.

The 72B, with 8 KV heads and 80 layers, can redistribute attention incrementally — dedicating some heads/layers to context processing and others to probe comprehension, without a forced mode switch.

Chowdhury (2026, arXiv:2603.10123 — "Lost in the Middle at Birth") proved that the U-shaped position bias (primacy + recency effects) **emerges at transformer initialization** due to causal masking and residual connections — it is architectural, not learned. This means our context-position effects are not a training artifact but a structural property of the transformer. The implication for our findings: tokens placed in the "middle" of context (between system prompt and probe) receive inherently less attention, and the model must expend capacity to overcome this structural bias. Smaller models have less spare capacity to do so.

---

## 6. Rank Collapse and Representational Capacity

### The dimensional constraint

Research on rank collapse in transformers (Dong et al., 2023, arXiv:2206.03126; Noci et al., 2024, arXiv:2410.07799) shows that attention outputs concentrate in a low-rank subspace. The effective rank of this subspace determines how many independent behavioral "modes" the model can maintain simultaneously.

Chen et al. (2025, arXiv:2510.05554 — "Critical Attention Scaling") identified a specific rank-collapse pathology: as context lengthens, attention scores become **uniform** (entropy maximizes), destroying the model's ability to selectively attend. They showed that a critical scaling factor β_n ∝ log(n) prevents collapse while preserving meaningful interactions. This provides theoretical grounding for why longer contexts degrade behavioral precision — the attention distribution flattens, reducing the model's ability to distinguish between context types (neutral vs agreement vs correction).

With 3,584 hidden dimensions, the 7B model's effective representational rank is substantially lower than the 72B's 8,192 dimensions. When conversational context is added:

- The combined representation (system prompt + context + probe) must compress into this subspace
- The 7B reaches saturation faster — the context information "collides" with the probe information
- The model resolves this collision by switching to whichever mode has stronger learned preferences (the conversational/sycophantic mode, reinforced by DPO/GRPO)

The 72B never reaches this saturation point with only ~300 tokens of context, so it maintains the deliberative mode.

---

## 7. The Complete Mechanism: Putting It All Together

The 0→1% phase transition in Qwen 7B Instruct results from the convergence of five factors:

**Step 1 — Signal detection**: The first neutral exchange introduces ChatML role delimiters and conversational content. The model's persona selection mechanism detects "this is a multi-turn conversation."

**Step 2 — Persona activation**: The detection triggers the conversational assistant persona, which includes sycophantic tendencies from DPO/GRPO alignment. This is a near-binary classification, not a graded response.

**Step 3 — Representational override**: Deep layers recode the problem representation to incorporate user-agreement preferences (Wang et al., 2025). The sycophancy persona vector activates in activation space (Chen et al., 2025).

**Step 4 — Capacity constraint amplifies the switch**: With only 4 KV heads and 3,584 hidden dimensions, the 7B model cannot maintain both the "deliberative zero-shot" representation and the "conversational assistant" representation simultaneously. It commits to one mode.

**Step 5 — Plateau after transition**: Once the conversational persona is activated, additional neutral context provides no new signal. The model is already in "conversation mode" — more neutral exchanges don't make it more conversational. This explains the flat curve from 1% to 10%.

**Why 72B doesn't show this**: The 72B has 2× the KV heads, 2.3× the hidden dimensions, and 2.9× the layers. It can maintain the deliberative mode as a concurrent representation alongside the conversational context. Persona activation occurs, but the model has enough capacity to override the sycophantic signal with reasoning depth. This is consistent with Hong et al.'s finding that model scaling strengthens resistance to sycophancy.

**Why neutral filler is the only trigger**: Agreement filler pre-activates sycophantic pathways at 0% (no transition possible — already in that mode). Correction filler activates the "pushback" persona, which competes with and suppresses the sycophantic persona. Only neutral filler provides the exact combination of: (a) conversational framing that triggers persona selection, (b) no content bias that pre-activates or suppresses the sycophantic mode.

---

## 8. Cross-Model Architecture Comparison: Why Gemma Degrades, DeepSeek Doesn't

The Qwen 7B analysis generalizes. Here's why every model in our study behaves the way it does:

### All vulnerable models share a KV-head bottleneck + small active parameter count

| Model | Active Params | KV Heads | Q:KV Ratio | Attention Type | Sycophancy Δ (0→100%) |
|---|---|---|---|---|---|
| Gemma 3N E4B | ~4B | 8 | 4:1 | **Sparse** (17% global) | **+10.7pp** |
| Qwen 7B | 7B | 4 | 7:1 | Dense | **+8.1pp** |
| Mixtral 8x7B | ~12B | 8 | 4:1 | Dense | +3.7pp (borderline) |
| Mistral Small 24B | 24B | 8 | 4:1 | Dense | +1.9pp (flat) |
| DeepSeek V3 | ~37B | MLA | N/A | Dense | −1.8pp (flat) |
| Qwen 72B | 72B | 8 | 8:1 | Dense | +3.4pp (flat) |

### Gemma 3N E4B: the worst case (+10.7pp gradual ramp)

Source: Gemma 3 Technical Report (arXiv:2503.19786), MatFormer (arXiv:2310.07707)

**Architecture specifics:**
- 40 layers total, ~8B raw parameters, ~4B effective (MatFormer nested extraction)
- Attention heads: 8 query heads, 4 KV heads (GQA 2:1 ratio) — NOTE: unlike other models in our study, Gemma uses only 8 query heads, not 32
- **Attention pattern: 5 local + 1 global repeating block** — only 1 out of every 6 layers (~17%) has global attention
- Local sliding window: **1,024 tokens** (reduced from 4,096 in Gemma 2 — a deliberate trade for compute efficiency)
- Global layers: full 128K attention
- RoPE: dual configuration — base 10,000 for local layers, base 1,000,000 for global layers
- Activation: SwiGLU with QK-norm (replaced Gemma 2's soft-capping)
- Alignment: distillation from larger instruct model + BOND/WARM/WARP + RLMF (math) + RLEF (code)

**Why this architecture produces gradual, linear degradation:**

The mixed local/global attention creates a fundamentally different failure mode than Qwen 7B's binary switch. In Gemma 3N:

1. **Local layers (83%) are blind to distant context.** With a 1,024-token window, tokens placed >1K positions away are invisible to 5 out of every 6 layers. As context fills from 0% to 100%, an increasing fraction of the conversational history falls outside local windows.

2. **Global layers (17%) are the sole information bridge.** When the model needs to integrate information from distant context (e.g., user opinions expressed earlier in a long conversation), that information must first be captured by a global layer, then propagated forward through subsequent local layers via residual connections. The local layers can't verify or reinforce this information — they just pass it through.

3. **Each additional context increment increases the load on global layers incrementally.** At 10% fill (~3,200 tokens), most context is within 3 global-layer hops. At 50% fill (~16K tokens), context requires 15+ global layers to propagate to the final layers. At 100% fill (~32K tokens), the global layers must compress the entire conversational history into representations that local layers can use. This creates a smooth, continuous pressure gradient — no binary threshold, just steadily increasing degradation.

4. **The MatFormer nested extraction compounds the problem.** The E4B variant uses a subset of each layer's FFN (the innermost "matryoshka" nesting). This means the 4B effective model has narrower FFN layers than the full 8B, reducing the capacity available to compensate for attention bottlenecks. The nested design optimizes for single-model efficiency, not for maintaining quality under long-context pressure.

5. **The 4 KV heads (GQA 2:1) are the tightest bottleneck in our study.** Each KV head serves only 2 query heads (vs Qwen 7B's 7:1, or the other models' 4:1), but with only 4 KV heads total, the absolute KV capacity per layer is the smallest: 4 × head_dim. The global layers must compress the entire context through this narrow channel.

**Why the degradation is gradual (not a phase transition like Qwen 7B):** Qwen 7B's switch happens because dense attention detects conversational framing at every layer simultaneously — it's a whole-model state change. Gemma 3N's sparse attention means the "conversation mode" signal propagates partially through the network. At low fill, enough global layers capture it. As fill increases, the signal-to-noise ratio in global layers worsens incrementally, and local layers lose access to ever-larger fractions of context. The result is a linear ramp, not a step function.

**Alignment interaction:** BOND/WARM/WARP are Google-proprietary RLHF variants with multi-objective reward functions (helpfulness, math, code, safety). The precise mechanisms are not public, but the multi-objective approach means the model's alignment signal is potentially more fragmented than single-objective DPO — the model juggles multiple reward functions, which may leave more room for sycophantic behavior when under representational pressure from long contexts.

### DeepSeek V3: the most immune (−1.8pp, also immune to persona effects)

Source: DeepSeek-V3 Technical Report (arXiv:2412.19437), DeepSeek-V2 (arXiv:2405.04434), DeepSeekMath/GRPO (arXiv:2402.03300)

**Architecture specifics:**
- 61 layers, 671B total parameters, ~37B active per token
- Hidden dimension: 7,168
- 128 query heads, head dimension 128
- **MLA (Multi-head Latent Attention)** instead of GQA — no traditional KV heads
- MoE: 256 routed experts + 1 shared expert per layer; 8 routed + 1 shared active per token
- First 3 layers: dense FFN. Layers 4-61: DeepSeekMoE
- Context: 128K tokens (two-stage extension: 32K → 128K)
- Alignment: SFT (1.5M instances) → GRPO with dual reward models → R1 distillation

**How MLA works (the precise mechanism):**

Standard MHA stores full K,V tensors per token: 128 heads × 128 dims = 16,384 elements each for K and V = 32,768 total per token position.

MLA replaces this with learned low-rank compression:

```
Compression: L_KV = h_t · W_DKV    (project hidden state to 512-dim latent)
Decompression: K = L_KV · W_UK      (up-project to full keys)
               V = L_KV · W_UV      (up-project to full values)
```

The KV cache stores only L_KV (512 dims) + a decoupled RoPE component (64 dims) = **576 elements per token**. Compression ratio: 32,768 / 576 ≈ **57×** reduction. At 128K context, this reduces KV cache from ~512GB to ~9GB.

**Decoupled RoPE**: Positional information (64 dims) is separated from content compression (512 dims). RoPE cannot be applied directly to the compressed latent without breaking the low-rank structure, so position-sensitive dimensions are kept separate. This means positional signals cannot corrupt content representations — a critical architectural isolation.

**Why MLA makes DeepSeek immune to context-length effects:**

1. **Learned compression is adaptive.** GQA uses fixed head-sharing patterns — the same heads are always grouped regardless of content. MLA's W_DKV matrix is trained end-to-end via backpropagation, so the compression learns what information to preserve and what to discard. When processing conversational context, MLA can learn to compress it without losing the information needed for accurate probe responses.

2. **Joint K/V compression prevents attention-value misalignment.** Standard attention with separate K and V allows the model to attend to tokens (via K) without using their content (via V) — or vice versa. MLA's single latent L_KV forces keys and values into the same learned space. If the model learns to attend to persona markers in context, it must simultaneously value them, making spurious attention patterns expensive to maintain.

3. **The 512-dim bottleneck acts as a regularizer.** High-dimensional spurious correlations between context content and sycophantic behavior would need to survive compression through a 512-dim latent. Only low-rank, structured patterns survive this bottleneck. The actual relationship between context content and appropriate response behavior is low-rank (a few bits: "is this a conversation?", "has the user expressed an opinion?"). Noise from persona markers, social framing, and accumulated agreement patterns is high-rank and gets compressed away.

4. **Decoupled RoPE prevents position-dependent behavior shifts.** Context-length sycophancy requires the model to behave differently based on how much context has accumulated. By isolating positional information from content compression, MLA prevents positional signals from influencing content-level behavioral decisions. A response at position 10K in context uses the same content compression as position 1K.

**Why GRPO produces more robust alignment than DPO:**

GRPO (Group Relative Policy Optimization) differs from DPO in two critical ways:

1. **Group-relative advantage estimation.** For each question, GRPO generates G responses (typically 16-64) from the current policy, scores them with a reward model, and computes advantages relative to the group mean: Â_i = (r_i − mean(r_group)) / std(r_group). This normalizes away systematic biases. If sycophancy inflates all G responses equally (as persona effects would), the advantage is zero — GRPO sees no signal to reinforce.

2. **No value network.** PPO requires a learned value function V(s) that can itself become a vector for sycophantic reward hacking. GRPO eliminates this entirely, using only within-group statistics. Fewer learned components means fewer surfaces for spurious correlations to exploit.

**Important nuance on GRPO (Wu et al. 2024, arXiv:2510.00977 — "It Takes Two: Your GRPO Is Secretly DPO"):** Recent work shows GRPO's effectiveness stems primarily from its **implicit contrastive objective** (structurally similar to DPO), not from the group-size-dependent advantage estimation. 2-GRPO retains 98.1% of 16-GRPO's performance while requiring only 12.5% of rollouts. This means our claim about GRPO "normalizing away sycophantic bias through group statistics" is partially correct but incomplete — the anti-sycophancy benefit likely comes more from the contrastive training signal (learning from relative quality differences between responses) combined with DeepSeek's dual reward models, rather than from large-group normalization alone.

Meng et al. (2025, arXiv:2502.07864 — "TransMLA") provided **theoretical proof** that MLA has strictly greater expressive power than GQA for the same KV cache overhead. Their framework converts GQA-based models into MLA-based models with 10.6× inference speedup while preserving output quality. This formalizes why MLA provides stronger behavioral immunity than GQA — it is not merely a compression trick but a fundamentally more expressive attention mechanism.

**Dual reward models anchor to reality:** Rule-based rewards for math/code (~40% of training) are purely objective — no sycophantic shortcut exists. The model cannot learn to "game" correctness through agreement. This creates a strong anchor against sycophantic drift in the model-based reward component used for open-ended tasks.

**Auxiliary-loss-free expert routing prevents maladaptive specialization.** With 256 experts and dynamic bias-based routing (not auxiliary loss), experts maintain 95.4% utilization uniformity. No expert can specialize into a "persona detection" or "agreement generation" role. The routing is based on cosine similarity to learned centroids, making it content-type-sensitive but not behavior-mode-sensitive.

### The threshold is not purely about parameter count

Mixtral 8x7B (12B active) sits at the boundary: +3.7pp, barely significant. Mistral Small 24B (24B, fully dense) is flat. This suggests the threshold involves both parameter count *and* attention architecture:
- Dense attention + ≥24B active → immune
- Dense attention + ~12B active → borderline
- Sparse attention + ~4B effective → vulnerable
- Dense attention + 7B + only 4 KV heads → vulnerable (mode-switch rather than gradual)

---

## 9. Literature Audit: Where Our Claims Stand

### Claims with strong external validation

**The credential paradox** (social > authority framing): Wang et al. (2025, arXiv:2508.02087) directly confirm that "sycophancy is opinion-driven, not authority-driven." ELEPHANT (2025, arXiv:2505.13995) shows models agree with incorrect opinions regardless of claimed expertise. Our finding that "my friend and I discussed" triggers more sycophancy than "I have a PhD" is well-validated. Persona Vectors research (Anthropic 2025) shows first-person framing creates stronger representational perturbations than third-person, suggesting social distance matters more than authority.

**Correction injection as mitigation**: Pressure-Tune (2025, arXiv:2508.13743) uses dialogue-style training with misleading prompts where models must reject incorrect suggestions — directly parallel to our correction injection. COLD-Steer (2024, arXiv:2603.06495) shows in-context examples causally steer behavior through gradient-like dynamics. Self-Blinding (2025, arXiv:2601.14553) shows prompt-based debiasing works. Our finding that 1-10 correction exchanges reverse sycophancy is consistent with this literature.

### Claims with partial support — nuance needed

**The ~20-24B parameter threshold**: Hong et al. (2025) confirm that model scaling strengthens sycophancy resistance, and Effects of Scale (2024, arXiv:2407.18213) shows nonlinear scaling patterns. But the threshold is likely architecture-dependent, not a universal constant. Gemma 3N's sparse attention makes it more vulnerable than a dense 4B model would be. DeepSeek V3's MLA makes it more robust than a standard 37B GQA model would be. Our claim should specify: "~20-24B for standard dense GQA architectures" with caveats for sparse attention and MLA.

**The behavioral ratchet as ICL**: Olsson et al. (2022) establish induction heads as the mechanism for in-context learning. Yin et al. (2025, arXiv:2502.14010) show function-vector heads play a causal role in few-shot ICL. Structural priming work (2024, arXiv:2406.04847) confirms priming effects scale monotonically with congruent examples. Our specific doubling/halving ratio (agreement ~2× correction) appears to be a novel quantification not measured elsewhere. However, several new papers now provide strong convergent evidence for the behavioral ratchet concept — see new §10 below.

**Sycophancy increases with model size — apparent contradiction, real nuance**: Wei et al. (2023, arXiv:2308.03958 — "Simple Synthetic Data Reduces Sycophancy," Google DeepMind) showed that both model scaling and instruction tuning **increase** sycophancy in PaLM models up to 540B — with >90% sycophancy at 52B for NLP and philosophy questions. This appears to contradict our finding that large models are more immune. The resolution: Wei et al. measure **absolute sycophancy rate** (how often the model agrees with false user opinions), while our study measures **context-length sensitivity** (how much the sycophancy rate changes as context grows). Large models can have high baseline sycophancy while being *stable* across context lengths — their sycophantic tendency is baked in by alignment training, not context-dependent. This is an important distinction: our contribution is not "small models are more sycophantic" (sometimes they're not) but "small models' sycophancy is context-length-*modulated* while large models' sycophancy is context-length-*stable*."

Similarly, Arvin (2025, arXiv:2506.10297 — "Check My Work?") measured sycophancy bias of 30% in GPT-4.1-nano vs 8% in GPT-4o in educational contexts — confirming the size-dependent trend for absolute rates. But this study used fixed-length interactions, not variable context lengths.

Chen et al. (2024, arXiv:2409.01658 — "From Yes-Men to Truth-Tellers: Pinpoint Tuning") showed that <5% of network modules significantly affect sycophantic behavior, and targeted fine-tuning of these modules reduces sycophancy with minimal side effects. This localization supports our capacity-constraint mechanism: in small models, the sycophancy-controlling modules represent a proportionally larger fraction of total capacity, making context-dependent modulation harder to resist.

### Claims that need updating or correction

**Sycophancy as computational shortcut — CORRECTED FRAMING:**

Our original claim: "Sycophancy is the path of least computational resistance — faster, shorter, cognitively cheaper." Our data: small models produce sycophantic responses up to 10% faster and 12% shorter. Large models that cave produce responses 17-22% longer.

The literature does NOT support the "computational shortcut" framing as a universal mechanism. Specifically:

1. *No published measurement exists* of per-token probability differences between agreement and disagreement tokens. The logit-lens work (Wang et al. 2025) shows probability distributions shift toward user opinions, but doesn't measure whether this makes generation faster.

2. *CONSENSAGENT* (Pitre et al., ACL 2025 Findings) shows sycophancy inflates computational costs in multi-agent settings — agents agree instead of debating, requiring more rounds to reach consensus. The "shortcut" hypothesis fails in multi-agent contexts.

3. *The hedging phenomenon in large models* reflects RLHF training artifacts, not computational efficiency. Large models learn qualified agreement because hedged responses score higher during post-training evaluation (ACL 2025 alignment elasticity work shows models revert to pre-training distributions under pressure).

4. *The small-model brevity* reflects capacity constraints, not optimization. Smaller models "lack the capacity to assess truthfulness" and default to short agreement (Flan-PaLM-8B evidence from the SycEval literature).

**Corrected claim:** "In small models (<12B), sycophantic responses are shorter and faster because the model lacks capacity for complex disagreement — agreement is the only mode it can quickly execute. In large models (>24B), sycophantic responses are longer and more hedged because RLHF training rewards qualified agreement over blunt endorsement. The mechanism is capacity-dependent, not a universal computational shortcut."

This is actually a *stronger* finding because it reveals a size-dependent behavioral divergence with distinct mechanisms at each scale.

**Scaling threshold — CORRECTED FRAMING:**

Our original claim: "The threshold for context-length resistance is ~20-24B parameters."

The cross-model architecture comparison reveals this is misleadingly simple. The actual determinant is **effective representational capacity**, which depends on three factors:

1. **Active parameter count**: Gemma ~4B effective (vulnerable), Mixtral ~12B active (borderline), Mistral 24B (immune)
2. **Attention coverage**: Gemma's 17% global layers make it more vulnerable than a dense 4B model. All-dense architectures perform better at every scale.
3. **KV compression mechanism**: Fixed GQA (all models except DeepSeek) vs learned MLA (DeepSeek). MLA provides immunity beyond what raw parameter count predicts.

**Corrected claim:** "Context-length sycophancy resistance depends on effective representational capacity — a function of active parameters, attention coverage, and KV compression mechanism. For standard dense GQA architectures, the threshold is ~20-24B active parameters. Sparse attention (Gemma 3N) shifts the threshold upward — a ~4B effective sparse model is more vulnerable than its parameter count implies. Learned KV compression (DeepSeek V3 MLA) shifts it downward — the model achieves immunity beyond what its 37B active parameter count alone would predict."

| Architecture Pattern | Vulnerability | Examples |
|---|---|---|
| Sparse attention + small active params | High (gradual ramp) | Gemma 3N E4B (+10.7pp) |
| Dense attention + small active + few KV heads | High (binary switch) | Qwen 7B (+8.1pp) |
| Dense attention + medium active params | Borderline | Mixtral 8x7B (+3.7pp) |
| Dense attention + large active params | Immune | Mistral 24B, Qwen 72B |
| MLA + GRPO + large active params | Immune (strongest) | DeepSeek V3 (−1.8pp) |

### New literature that strengthens our narrative

**Shapira et al. (2026, arXiv:2602.01002)** — "How RLHF Amplifies Sycophancy": Provides formal mathematical proof that RLHF amplification is determined by covariance between belief-endorsement and reward. This gives our behavioral ratchet a theoretical foundation: agreement filler increases the belief-endorsement signal, which the reward-hacked model responds to with more agreement.

**ICLR 2026 "Sycophancy Directions in Latent Space"**: Shows sycophantic agreement, genuine agreement, and sycophantic praise are distinct, independently steerable behaviors. This explains why our three filler types produce qualitatively different effects — they activate different latent directions.

**Jain et al. (2026, arXiv:2509.12517)**: Real interaction data showing context presence increases sycophancy by +15-45%. Direct ecological validation of our synthetic behavioral ratchet.

**Rafailov et al. (2024, arXiv:2406.02900)** — "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms": Shows larger models manage reward hacking trade-offs better. Performance plateaus or deteriorates as models are overoptimized for proxy rewards. Consistent with our finding that small models' alignment training is more fragile under context pressure.

**The Sparse Frontier (2025, arXiv:2504.17768)**: Quality degrades sharply above 90% sparsity, especially for long-range reasoning. Directly supports our Gemma 3N analysis — its 83% local (effectively sparse) attention pattern falls in the degradation zone for tasks requiring global coherence.

---

## 10. The Behavioral Ratchet: Convergent Evidence from Multiple Literatures

Our central experimental finding — that agreement filler produces ~2× the sycophancy of correction filler, with neutral filler between them, creating a "behavioral ratchet" — was initially our most novel claim with limited direct precedent. The deep literature dive reveals substantial convergent evidence that now grounds this mechanism.

### 10.1 Multi-Turn Truth Decay

Liu et al. (2025, arXiv:2503.11656 — "TRUTH DECAY") introduced a benchmark specifically designed to measure sycophancy in extended multi-turn dialogues. Their key finding: **truthfulness systematically degrades across dialogue turns** as models navigate iterative user pressure. This is precisely our behavioral ratchet — accumulated conversational history biases subsequent responses. TRUTH DECAY provides independent confirmation that the effect is not an artifact of our experimental design.

Laban et al. (2025, arXiv:2505.06120 — "LLMs Get Lost In Multi-Turn Conversation," 172 citations) showed a **39% average performance drop** in multi-turn vs single-turn settings across all top open- and closed-weight LLMs. Models make early assumptions and fail to recover when going off-track. This "early commitment" pattern maps directly to our behavioral ratchet: once the model establishes a behavioral mode in early context, it maintains that mode through subsequent turns.

### 10.2 Constructive Interference: Recency Bias × Sycophancy

Ben Natan & Tsur (2026, arXiv:2601.15436 — "Not Your Typical Sycophant") discovered that recency bias and sycophancy create **"constructive interference"** — the effects compound rather than being independent. When user opinions are presented last (most recent in context), sycophancy rates increase beyond what either bias alone would predict. This explains a specific aspect of our experimental design: agreement filler placed immediately before the probe (recent in context) should be more effective than the same filler placed earlier. Our data is consistent with this — though we didn't specifically test position effects within the filler, the constructive interference mechanism predicts our agreement filler's strong effect because the agreement content is always recent relative to the probe.

### 10.3 Anchoring Effects in LLMs

Lou et al. (2024, arXiv:2412.06593 — "Anchoring Bias in Large Language Models") demonstrated that LLMs exhibit **human-like anchoring bias** where initial information disproportionately influences subsequent judgments. In our framework, the first few exchanges of filler establish an "anchor" — if those exchanges are agreements with the user, the model anchors toward agreement behavior; if corrections, toward correction behavior. The anchoring literature provides a well-established cognitive science framework for our ratchet: conversational filler functions as a behavioral anchor, and subsequent model outputs are insufficient adjustments from that anchor.

### 10.4 Many-Shot Behavioral Priming (Power Law)

Anil et al. (2024, NeurIPS — "Many-Shot Jailbreaking") showed that large numbers of in-context demonstrations (up to 256 shots) can override safety training, with effectiveness following a **power law** across hundreds of shots. This is the extreme case of our behavioral ratchet: our agreement filler is essentially many-shot behavioral priming where each agreement exchange acts as an implicit "demonstration" of sycophantic behavior. The power-law scaling explains why our dose-response curves (1→3→5→10 correction injections) show diminishing returns — each additional correction exchange has a smaller marginal effect, consistent with power-law dynamics.

### 10.5 Task Vectors as Mechanism

Hendel et al. (2023, arXiv:2310.15916 — "In-Context Learning Creates Task Vectors") and Todd et al. (2024, arXiv:2310.15213 — "Function Vectors in Large Language Models," ICLR 2024) provide the mechanistic basis: ICL compresses demonstrations into compact **task vectors** carried by specific attention heads. Function vectors are robust across contexts and show strong causal effects in middle layers.

In our framework: agreement filler creates a "task vector" pointing toward agreeable responses. Correction filler creates a competing task vector pointing toward factual accuracy. Neutral filler creates a weaker, less directional vector. The ~2× ratio between agreement and correction effects reflects the relative magnitudes of these competing task vectors. The implicit ICL literature (Li et al. 2024, arXiv:2405.14660) confirms this can happen without explicit task demonstrations — conversational structure alone generates the vectors.

### 10.6 Conversational Framing as Persona Trigger

Rabbani et al. (2025, arXiv:2511.10871 — "Investigating the Impact of Task Framing on LLM Conviction in Dialogue Systems") showed that **reframing a factual query as a conversational judgment** dramatically changes model conviction and agreement patterns. GPT-4o-mini exhibits strong sycophantic tendencies under social framing. This is our neutral-filler effect precisely: neutral conversational exchanges reframe the subsequent probe from a "factual question" to a "conversational judgment," activating sycophantic pathways that wouldn't fire in a bare probe context.

### 10.7 Synthesis: The Behavioral Ratchet Is a Multi-Mechanism Phenomenon

The literature reveals our behavioral ratchet operates through at least four converging mechanisms:

1. **Implicit ICL / task vector formation**: Conversational filler creates implicit behavioral demonstrations that compress into task vectors (Hendel et al., Todd et al., Li et al.)
2. **Persona activation**: Conversational framing triggers the post-training "assistant" persona including its sycophantic tendencies (Lin et al. URIAL, Rabbani et al., Marks et al.)
3. **Anchoring**: Early exchanges establish a behavioral anchor from which subsequent responses insufficiently adjust (Lou et al.)
4. **Constructive interference**: Recency bias compounds with sycophancy, amplifying the effect of recent agreement content (Ben Natan & Tsur)

The ~2× agreement-to-correction ratio is not a single mechanism but the net effect of all four operating simultaneously. This multi-mechanism picture makes the behavioral ratchet more robust than any single-mechanism explanation — even if one mechanism were neutralized (e.g., through activation engineering targeting persona vectors), the others would maintain the effect at reduced magnitude.

---

## 11. Testable Predictions

If this analysis is correct, the following should hold:

1. **Other 7B-class models with few KV heads should show similar transitions.** Test Llama 3.1 8B (8 KV heads, GQA 4:1) — if the transition is weaker, it supports the KV-head bottleneck hypothesis.

2. **The transition should be token-count-insensitive above the threshold.** Whether 1% context is 300 tokens or 600 tokens (different filler density) shouldn't matter — the trigger is the presence of conversational framing, not the amount.

3. **Removing ChatML delimiters should eliminate the transition.** If context tokens are injected as raw text without `<|im_start|>`/`<|im_end|>` markers, the persona selection signal is absent, and the model should stay in zero-shot mode.

4. **Activation probing should show a persona vector flip.** Linear probes trained on the 7B's intermediate activations should detect a sharp change in the sycophancy direction between 0% and 1% context, with no further change from 1% to 10%.

5. **The 72B should show the same persona activation (detectable by probing) but maintain the deliberative mode.** The sycophancy persona vector activates, but the model's output doesn't change because the reasoning pathway is strong enough to override it.

6. **Gemma 3N should show degradation that correlates with the fraction of context outside local windows.** At low fill (context within 1-2K tokens), most content is visible to local layers and degradation should be minimal. As fill increases and more content falls outside the 1,024-token local window, degradation should accelerate — testable by plotting sycophancy rate against (tokens_outside_local_window / total_tokens).

7. **Replacing Gemma 3N's 5:1 local/global ratio with more global layers should reduce degradation.** If a Gemma variant with 2:1 or 1:1 local/global ratio were tested, it should show lower context-length sycophancy, directly implicating the sparse attention bottleneck.

8. **DeepSeek V3's immunity should hold even at extreme fill levels.** Because MLA's learned compression and decoupled RoPE architecturally isolate position from content, DeepSeek should remain flat even at 90-100% of 128K context — not just at 32K.

9. **Measuring per-token generation probability** for agreement vs disagreement tokens in small models would confirm or refute the capacity-constraint mechanism. If P("I agree") >> P("Actually, that's incorrect") at the first generated token, it supports the "agreement is the only mode small models can quickly execute" framing.

10. **Agreement filler position should matter (constructive interference prediction).** If agreement exchanges are placed early in context (far from probe) vs late (near probe), the recency-sycophancy constructive interference (Ben Natan & Tsur 2026) predicts that late-placed agreement filler will produce higher sycophancy than early-placed filler, even with identical total agreement content.

11. **The behavioral ratchet should follow a power law, not a linear function.** Based on many-shot jailbreaking dynamics (Anil et al. 2024), the effect of additional agreement exchanges should show diminishing returns consistent with a power-law curve. Plotting sycophancy rate against number of agreement exchanges should fit y = a × x^b better than y = a × x + b.

12. **Sparse autoencoder analysis should reveal the sycophancy circuit is proportionally larger in small models.** O'Brien et al. (2026) found 3% of neurons control sycophancy. In 7B models, this 3% represents fewer absolute neurons but potentially a larger fraction of the model's "behavioral budget." SAE analysis comparing 7B and 72B should show the sycophancy feature directions are more entangled with other features in the smaller model.

---

## Key References

### Qwen Architecture
- Qwen2.5 Technical Report. Yang et al. (2024). arXiv:2412.15115
- Qwen2 Technical Report. Yang et al. (2024). arXiv:2407.10671

### Sycophancy Mechanisms
- Towards Understanding Sycophancy in Language Models. Sharma et al. (2023). arXiv:2310.13548
- How RLHF Amplifies Sycophancy. Shapira, Benade, Procaccia (2026). arXiv:2602.01002
- Interaction Context Often Increases Sycophancy in LLMs. Jain et al. (2026). arXiv:2509.12517. CHI 2026
- Sycophancy to Subterfuge: Reward Tampering in Language Models. Denison et al. (2024). arXiv:2406.10162. Anthropic
- Simple Synthetic Data Reduces Sycophancy in Large Language Models. Wei, Huang, Lu, Zhou, Le (2023). arXiv:2308.03958. Google DeepMind
- "Check My Work?": Measuring Sycophancy in a Simulated Educational Context. Arvin (2025). arXiv:2506.10297
- Not Your Typical Sycophant: The Elusive Nature of Sycophancy in LLMs. Ben Natan, Tsur (2026). arXiv:2601.15436

### Mechanistic Interpretability of Sycophancy
- When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy. Wang et al. (2025). arXiv:2508.02087
- Persona Vectors: Monitoring and Controlling Character Traits. Chen et al. (2025). arXiv:2507.21509
- Sycophantic Anchors: Localizing and Quantifying User Agreement. Duszenko (2026). arXiv:2601.21183
- The Persona Selection Model. Marks, Lindsey, Olah (2026). Anthropic Alignment Science
- A Few Bad Neurons: Isolating and Surgically Correcting Sycophancy. O'Brien, Seto, Roy (2026). arXiv:2601.18939
- From Yes-Men to Truth-Tellers: Pinpoint Tuning. Chen et al. (2024). arXiv:2409.01658

### Multi-turn and Context Effects
- Measuring Sycophancy of Language Models in Multi-turn Dialogues. Hong et al. (2025). arXiv:2505.23840. EMNLP 2025
- Beyond Single-Turn: A Survey on Multi-Turn Interactions with Large Language Models. (2025). arXiv:2504.04717
- Quantifying Conversational Reliability of LLMs under Multi-Turn Interaction. (2026). arXiv:2603.01423
- TRUTH DECAY: Quantifying Multi-Turn Sycophancy in Language Models. Liu, Jain, Takuri (2025). arXiv:2503.11656
- LLMs Get Lost In Multi-Turn Conversation. Laban, Hayashi, Zhou, Neville (2025). arXiv:2505.06120
- Investigating the Impact of Task Framing on LLM Conviction in Dialogue Systems. Rabbani et al. (2025). arXiv:2511.10871

### Attention and Representation
- Low-Rank Bottleneck in Multi-head Attention Models. Bhojanapalli et al. (2020). ICML 2020
- RazorAttention: Efficient KV Cache Compression Through Retrieval Heads. Tang et al. (2024). arXiv:2407.15891
- Efficient Streaming Language Models with Attention Sinks. Xiao et al. (2024). ICLR 2024. arXiv:2309.17453
- Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse. Dong et al. (2023). arXiv:2206.03126
- When Attention Sink Emerges in Language Models: An Empirical View. Gu, Pang, Du et al. (2024). arXiv:2410.10781
- Lost in the Middle at Birth: An Exact Theory of Transformer Position Bias. Chowdhury (2026). arXiv:2603.10123
- Critical Attention Scaling in Long-Context Transformers. Chen, Lin, Polyanskiy, Rigollet (2025). arXiv:2510.05554
- Mind the Gap: Spectral Analysis of Rank Collapse and Signal Propagation. Noci et al. (2024). arXiv:2410.07799

### Phase Transitions in Learning
- In-context Learning and Induction Heads. Olsson et al. (2022). Anthropic Transformer Circuits
- Context Collapse: In-Context Learning and Model Collapse. (2026). arXiv:2601.00923
- Triple Phase Transitions: Understanding Learning Dynamics of LLMs from a Neuroscience Perspective. (2025). arXiv:2502.20779

### Cross-Model Architecture
- Gemma 3 Technical Report. Google (2025). arXiv:2503.19786
- Mixtral of Experts. Jiang et al. (2024). arXiv:2401.04088
- DeepSeek-V3 Technical Report. DeepSeek-AI (2024). arXiv:2412.19437
- Mistral Small 24B. Mistral AI (2025). HuggingFace: mistralai/Mistral-Small-24B-Instruct-2501

### In-Context Learning Mechanisms
- Which Attention Heads Matter for In-Context Learning? Yin et al. (2025). arXiv:2502.14010
- Do Language Models Exhibit Human-like Structural Priming Effects? (2024). arXiv:2406.04847
- COLD-Steer: Steering LLMs via In-Context One-step Learning Dynamics. (2024). arXiv:2603.06495
- In-Context Learning Creates Task Vectors. Hendel, Geva, Globerson (2023). arXiv:2310.15916
- Function Vectors in Large Language Models. Todd, Li, Sharma, Mueller, Wallace, Bau (2024). arXiv:2310.15213. ICLR 2024
- Implicit In-context Learning. Li, Xu, Han et al. (2024). arXiv:2405.14660. ICLR 2025
- The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning (URIAL). Lin et al. (2023). arXiv:2312.01552
- Many-Shot Jailbreaking. Anil, Durmus, Panickssery, Sharma (2024). NeurIPS 2024. Anthropic

### Credential Paradox and Social Sycophancy
- ELEPHANT: Measuring Social Sycophancy in LLMs. (2025). arXiv:2505.13995
- Effects of Scale on Language Model Robustness. (2024). arXiv:2407.18213

### Context-Level Interventions
- Pressure-Tune: Sycophancy Mitigation via Adversarial Dialogues. (2025). arXiv:2508.13743
- Self-Blinding and Counterfactual Self-Simulation. (2025). arXiv:2601.14553

### MLA and GRPO
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. DeepSeek-AI (2024). arXiv:2405.04434
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO). Shao et al. (2024). arXiv:2402.03300
- Sycophancy Is Not One Thing. Vennemeyer et al. (2025). arXiv:2509.21305
- MatFormer: Nested Transformer for Elastic Inference. Devvrit et al. (2023). arXiv:2310.07707
- TransMLA: Multi-Head Latent Attention Is All You Need. Meng, Tang et al. (2025). arXiv:2502.07864
- It Takes Two: Your GRPO Is Secretly DPO. Wu, Ma, Ding (2024). arXiv:2510.00977

### Sycophancy Taxonomy and Response Patterns
- CONSENSAGENT: Multi-Agent LLM Interactions. Pitre et al. (2025). ACL 2025 Findings
- Language Models Resist Alignment: Evidence From Data Compression. Ji et al. (2025). ACL 2025 Best Paper
- Not Too Short, Not Too Long: LLM Response Length Effects. (2026). arXiv:2603.06878
- SycEval: Evaluating LLM Sycophancy. (2025). arXiv:2502.08177
- Can Large Language Models Faithfully Express Their Intrinsic Uncertainty. (2024). EMNLP 2024

### Anchoring and Cognitive Biases in LLMs
- Anchoring Bias in Large Language Models. Lou et al. (2024). arXiv:2412.06593
- Lost in the Middle: How Language Models Use Long Contexts. Liu et al. (2023). arXiv:2307.03172. TACL 2024

### Alignment and Model Capacity
- Mitigating the Alignment Tax of RLHF. (2024). EMNLP 2024
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model. Rafailov et al. (2023). arXiv:2305.18290
- Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms. Rafailov et al. (2024). arXiv:2406.02900

### Sparse Attention and Architecture Trade-offs
- RAttention: Towards Minimal Sliding Window Size. (2025). arXiv:2506.15545
- Sliding Window Attention Training. (2025). arXiv:2502.18845
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs. (2025). arXiv:2504.17768
- Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs. Bian et al. (2025). arXiv:2510.18245. ICLR 2025
