# The Behavioral Ratchet: How Conversational History Shapes LLM Sycophancy Across 80,433 Trials

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19026682.svg)](https://doi.org/10.5281/zenodo.19026682)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Trials](https://img.shields.io/badge/trials-80%2C433-blue)]()
[![Models](https://img.shields.io/badge/models-6-green)]()

**Paper:** [`paper/behavioral-ratchet-sycophancy-2026.pdf`](paper/behavioral-ratchet-sycophancy-2026.pdf)
**Author:** Karan Prasad ([ORCID](https://orcid.org/0009-0009-0747-2311)) — Obvix Labs

Does sycophancy increase as an LLM's context window fills up? We test this across six 32K-context models totalling **80,433 trials** (67,708 original + 4,140 correction injection + 4,799 mixed filler + 3,786 fine-grained). The context-length effect scales inversely with model size — small models (~4-12B) degrade measurably, large models (24B+) are flat. The universal finding across all six models is the **behavioral ratchet**: conversational pattern matters more than conversation length. Agreement filler roughly doubles sycophancy compared to correction filler (p < 10^-14 in every model).

Three follow-up experiments probe the ratchet's mechanics: a **correction injection** experiment (4,140 trials) shows the ratchet can be partially or fully reset — large models respond to 1 correction, small models need 5-10. A **mixed filler** experiment (4,799 trials) tests ecological validity with 7 interleaved ratios — the ratchet is a smooth gradient, and even 10% correction provides massive protection. A **fine-grained 0-10%** experiment (3,786 trials) reveals a genuine phase transition in Qwen 7B at 0->1% context fill that is neutral-filler-specific.

## Key Figures

| Phase Diagram | Behavioral Ratchet | Mixed Filler Gradient |
|:---:|:---:|:---:|
| ![Phase Diagram](code/figures/phase_diagram.png) | ![Filler Comparison](code/figures/filler_comparison_google_gemma-3n-e4b-it.png) | ![Mixed Filler](code/figures/mixed_filler_phase_diagram.png) |
| Small models cluster high; large models flat | Agreement > Neutral > Correction, universally | Smooth gradient, no threshold — 10% correction protects |

| Correction Injection | Fine-Grained Step | Persona Heatmap |
|:---:|:---:|:---:|
| ![Heatmap](code/figures/heatmap.png) | ![Step Function](code/figures/finegrained_step_function.png) | ![Persona](code/figures/persona_heatmap.png) |
| Dose-response reset varies by model size | Qwen 7B's 0->1% phase transition is neutral-filler-specific | Social pressure > credentials for inducing sycophancy |

## Results Summary

All results scored by Claude Sonnet 4.6 as LLM judge with domain-aware rubrics.

### Cross-Model Comparison

| Metric | Gemma 3N (~4B) | Qwen 7B | Mixtral 8x7B (~12B) | Mistral 24B | DeepSeek V3.1 (~37B) | Qwen 72B |
|---|---|---|---|---|---|---|
| Trials (valid) | 11,245 | 11,003 | 11,331 | 11,381 | 11,367 | 11,381 |
| Overall sycophancy | 34.2% | 21.3% | 22.7% | 3.8% | 6.0% | 6.7% |
| At 0% context | 27.7% | 13.1% | 19.0% | 3.0% | 7.4% | 4.6% |
| At 100% context | 38.4% | 21.2% | 22.7% | 4.9% | 5.5% | 8.0% |
| Delta | **+10.7 pp** | **+8.1 pp** | **+3.7 pp** | **+1.9 pp** | **-1.8 pp** | **+3.4 pp** |
| Spearman rho | 0.077*** | 0.028** | 0.021* | 0.033*** | -0.008 ns | 0.035*** |
| Trend | Gradual ramp | Step at 0->10% | Mild ramp | Flat | Flat | Gradual ramp |

### The Behavioral Ratchet (Filler Type Effect)

| Filler Type | Gemma 3N | Qwen 7B | Mixtral 8x7B | Mistral 24B | DeepSeek V3.1 | Qwen 72B |
|---|---|---|---|---|---|---|
| Agreement | 41.2% | 25.3% | 27.9% | 5.6% | 8.6% | 10.2% |
| Neutral | 36.2% | 23.1% | 26.6% | 3.8% | 5.7% | 5.8% |
| Correction | 25.1% | 15.6% | 13.6% | 2.1% | 3.7% | 4.2% |
| Chi-squared p | < 10^-50 | < 10^-25 | < 10^-58 | < 10^-14 | < 10^-18 | < 10^-26 |

Correction filler is strongly protective in every model. Mixtral shows the strongest correction effect (GLMM beta = -1.74) — correction history halves its sycophancy rate.

## How It Works

We construct synthetic conversations of varying length (0-100% of the 32K context window), then append a probe where the user states something wrong and asks the model to confirm. We measure agreement rate.

**Three filler types** isolate why sycophancy changes: neutral filler (cooking, travel) tests pure length effects; agreement filler (model agrees with user) tests behavioral momentum; correction filler (model corrects user) tests whether correction patterns protect.

**Six probe domains** (115 total probes): 100 factual probes with objectively wrong claims across factual, math, science, logic, and CS, plus 15 opinion probes with genuinely debatable claims.

**Persona variation**: Each non-opinion probe delivered through one of 8 rotating framings — casual thinker, college-educated, PhD holder, friend discussion, professor appeal, 15-year professional, peer-reviewed citation, and social consensus. These map to 4 authority groups: low authority, credential claim, external source, and social pressure.

**Scoring**: All results scored by Claude Sonnet 4.6 as judge, with separate rubrics for factual vs opinion probes. Sycophantic responses are further classified into failure modes (direct/elaborate/qualified) by a second Sonnet 4.6 taxonomy pass. Inter-rater reliability validated with Claude 3.5 Haiku on a 1,200-trial subsample (Cohen's kappa = 0.705, 93.4% agreement).

**Statistical model**: Bayesian binomial GLMM with probe_id as random intercept and logit link.

## Running a New Model

```bash
cd code/

# Run experiment
python run_experiment.py --mode api --model <openrouter-model-id> \
  --keys-file keys.txt --max-tokens 32768 --workers 30 --repeats 3

# Judge sycophancy
python llm_judge.py --input results/<model_slug>_results.jsonl \
  --output results/<model_slug>_judged.jsonl \
  --judge-model anthropic/claude-sonnet-4-6 --keys-file keys.txt --workers 35

# Classify failure modes
python taxonomy_judge.py --input results/<model_slug>_judged.jsonl \
  --output results/<model_slug>_judged.jsonl \
  --judge-model anthropic/claude-sonnet-4-6 --keys-file keys.txt --workers 35

# Generate figures and stats
python phase_diagram.py --results-dir results/ --output-dir figures/
python statistical_tests.py --results-dir results/ --output figures/stats_report.json --verbose
python secondary_analysis.py
```

Create `code/keys.txt` with one OpenRouter API key per line.

## Cost

All experiments run via OpenRouter. Sonnet 4.6 judge at $3/M input, $15/M output tokens.

| Component | Experiment | Judge | Total |
|---|---|---|---|
| 6 models x 11K trials | ~$165 | ~$406 | **~$571** |
| Taxonomy judge (all models) | — | ~$25 | **~$25** |
| Injection experiment (4,140 trials) | ~$10 | ~$25 | **~$35** |
| Mixed filler experiment (4,799 trials) | ~$14 | ~$29 | **~$43** |
| Fine-grained 0-10% (3,786 trials) | ~$1 | ~$7 | **~$8** |
| **Total** | **~$190** | **~$492** | **~$682** |

The judge dominates cost (~72%). The experiments themselves are cheap — even the 72B model only costs $20 for 11K calls.

## Repo Structure

```
├── README.md                          # This file
├── CITATION.cff                       # Machine-readable citation metadata
├── LICENSE                            # CC-BY-4.0
├── requirements.txt                   # Python dependencies
│
├── paper/
│   └── behavioral-ratchet-sycophancy-2026.pdf  # Preprint (2-column, 5 pages)
│
├── docs/
│   ├── experiment-protocol.md         # Full experimental protocol
│   ├── research-note.md               # Findings and interpretation
│   ├── brainstorm-synthesis.md        # Contribution mapping and paper strategy
│   └── architecture-analysis.md       # Cross-model architecture deep dive (60+ papers)
│
├── research-notes/                    # Background literature review PDFs
│
└── code/
    ├── probes.json                    # 115 probes (6 domains) + 8 persona templates
    ├── run_experiment.py              # Async experiment runner
    ├── llm_judge.py                   # Domain-aware LLM judge
    ├── taxonomy_judge.py              # Failure mode taxonomy judge
    ├── phase_diagram.py               # All figures
    ├── statistical_tests.py           # GLMM, Spearman, Mann-Whitney, chi-squared
    ├── persona_analysis.py            # Persona template effect analysis
    ├── irr_check.py                   # Inter-rater reliability
    ├── secondary_analysis.py          # Taxonomy, latency, response length
    ├── run_correction_injection.py    # Correction injection experiment
    ├── analyze_injection.py           # Injection results analysis
    ├── run_mixed_filler.py            # Mixed filler ratio experiment
    ├── analyze_mixed_filler.py        # Mixed filler analysis
    ├── run_finegrained.py             # Fine-grained 0-10% experiment
    ├── analyze_finegrained.py         # Fine-grained analysis
    ├── generate_finegrained_diagrams.py
    ├── results/                       # Raw + judged JSONL (80K+ results)
    └── figures/                       # Generated figures + stats JSON
```

## Limitations

1. **Context-length effect is architecture-dependent.** Vulnerability depends on effective representational capacity (active params x attention coverage x KV compression), not parameter count alone.

2. **Template filler.** 10 template pairs per filler type. At 100% context, pairs repeat ~6x each.

3. **Single judge model.** All scoring by Sonnet 4.6. IRR with Haiku: kappa = 0.705 (substantial), 93.4% raw agreement on 1,200 stratified trials.

4. **No Llama models.** No Meta Llama model has a native 32K context limit — Llama 3.1 8B is 128K on OpenRouter, so 32K only fills 12% of its window.

5. **Injection at single context level.** Correction injection tested only at 50% fill. Interaction with context level is untested.

6. **Mixed filler uses random draw.** Actual ratios have sampling variance around targets (~40-50 exchanges per trial).

7. **Fine-grained is single-model.** The 0-10% zoom covers only Qwen 7B.

See the paper for full discussion.

## Citation

```bibtex
@dataset{prasad2026behavioral_ratchet,
  author    = {Prasad, Karan},
  title     = {The Behavioral Ratchet: How Conversational History Shapes
               LLM Sycophancy Across 80,433 Trials},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19026682},
  url       = {https://doi.org/10.5281/zenodo.19026682},
  license   = {CC-BY-4.0}
}
```

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt for any purpose with attribution.
