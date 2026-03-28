# Algorithm Details

## Overview

MAJEC uses an **Expectation-Maximization (EM) algorithm** enhanced with evidence-based priors and momentum acceleration to resolve read ambiguity. The algorithm's accuracy stems from two key innovations:

1. **A hierarchical two-phase EM** that respects the information quality hierarchy between unique and multi-mapping reads.
2. **A multi-layered prior adjustment system** that incorporates splice junction evidence, annotation quality, and structural integrity.

---

## Two-Phase Hierarchical EM

MAJEC processes reads in two distinct phases, recognizing that unique mappers and multi-mappers represent fundamentally different levels of uncertainty:

### Information Quality Hierarchy

| Read Type | Certainty Level | Primary Ambiguity |
|-----------|----------------|-------------------|
| **Unique mappers** | High-confidence genomic location | Only isoform choice at that locus |
| **Multi-mappers** | Uncertain genomic location | Both locus assignment AND isoform choice |

### Processing Strategy

**Phase 1 — Unique Mapper Processing:**

```
θ_unique[t, i+1] = E_step(unique_classes, θ_total[t, i])
```

Process high-confidence genomic assignments first to establish expression patterns at each locus.

**Phase 2 — Multi-mapper Processing (Informed by Phase 1):**

```
priors[t] = θ_unique[t, i+1] + θ_multi[t, i]
θ_multi[t, i+1] = E_step(multi_classes, priors[t])
```

Allocate multi-mapping reads based on established expression patterns from unique mappers.

**Combined Update:**

```
θ_total[t, i+1] = θ_unique[t, i+1] + θ_multi[t, i+1]
```

**Rationale:**

- Respects biological reality (genomic vs. isoform uncertainty are distinct problems).
- Uses high-quality evidence to guide ambiguous assignments.

---

## The E-Step: Read Allocation

**Formula:**

```
P(read | transcript_i) ∝ θ_i
```

**Standard EM Update (per equivalence class):**

```
For transcript t in equivalence class C:
θ[t, i+1] = unique_reads[t] + Σ_classes( reads[C] × θ[t,i] / Σ_j∈C(θ[j,i]) )
```

---

## Evidence-Based Prior Initialization

Before the EM algorithm begins, initial read counts are modulated by a series of independent, evidence-based priors applied sequentially:

```
Initial Counts (from featureCounts)
    ↓
1. Junction Boost (positive evidence from splice junctions)
    ↓
2. TSL Penalty (down-weight low-confidence annotations)
    ↓
3. Junction Completeness Penalty (penalize structural incompleteness)
    ↓
4. Subset Isoform Penalty (resolve fragment vs. complete transcripts)
    ↓
Final Adjusted Priors → EM Algorithm
```

---

## Prior Adjustment Models

### 1. Junction Evidence Boost

Transcripts receive positive evidence based on splice junction support, weighted by junction uniqueness.

**Annotation-Level Uniqueness:** During preprocessing, every junction `j` is labeled with `n_transcripts` (the number of transcripts sharing it).

**Read-Based Evidence Calculation:**

```
evidence[j, t] = count[j] × (1/n_transcripts)^decay_exponent × junction_weight

Where:
- count[j]: Number of reads spanning junction j
- n_transcripts: Number of transcripts containing junction j
- decay_exponent: Penalty for shared junctions (default: 1.0)
- junction_weight: Global importance parameter (default: 3.0)
```

**Example:** A junction shared by 10 transcripts with `decay_exponent=2`:

```
weight = (1/10)^2 = 0.01 (1% of the evidence per transcript)
```

### 2. TSL Penalty (`--use_tsl_penalty`)

Down-weights transcripts based on GENCODE/Ensembl Transcript Support Level annotations.

**Default Penalty Values:**

```
TSL1: 1.0   (highest confidence, no penalty)
TSL2: 0.9
TSL3: 0.7
TSL4: 0.5
TSL5: 0.3
NA:   0.8   (unknown quality)
```

Applied as: `adjusted_count[t] = initial_count[t] × tsl_penalty[t]`.

### 3. Junction Completeness Penalty (`--use_junction_completeness`)

A statistical model that penalizes transcripts showing incomplete junction observation, indicating potential fragmentation or dropout.

**Algorithm:**

```
For each multi-exon transcript:
1. Calculate baseline: median coverage of observed junctions
2. For each junction:
   - Calculate Z-score comparing observed vs. expected coverage
   - Use variance model with overdispersion parameter
3. Identify worst-performing junctions
4. Apply terminal dropout modeling (if --library_type specified):
   - 5' dropout for dT/polyA libraries
   - Apply --terminal_relax to reduce penalties in expected dropout zones
5. Calculate final penalty based on Z-score severity
```

**Final penalty range:** controlled by `--junction_completeness_min_score`.

### 4. Two-Stage Subset Isoform Penalty (`--use_subset_penalty`)

Resolves ambiguity between fragment isoforms and complete transcripts using a hybrid annotation + data-driven approach.

#### Stage 1: Annotation-Based Initial Penalty

```
For each subset transcript t with supersets S:
1. Calculate expected shared junction coverage from superset evidence:
   expected_coverage = f(superset_exclusive_junction_counts)

2. Compare to observed shared junction coverage:
   z_score = (observed - expected) / (sqrt(variance) + 1e-6)

3. If observed coverage cannot exceed expected (low z_score):
   → Apply initial penalty (subset is likely an artifact)
```

The `--subset_evidence_threshold` parameter raises the bar for what counts as "excess evidence".

#### Stage 2: Data-Driven Validation (`--use_subset_coverage_data`)

Uses direct read coverage from pre-defined unique exonic territories to validate or override the initial penalty.

```
1. Measure coverage in unique and comparator territories:
   unique_density = mean_coverage(unique_territory)
   comparator_density = mean_coverage(comparator_territory)

2. Calculate evidence ratio:
   evidence_ratio = unique_density / comparator_density

3. Weight by territory length:
   confidence = f(unique_territory_length)

4. Combine with annotation-based penalty:
   final_penalty = confidence × data_penalty + (1-confidence) × annotation_penalty
```

**Interpretation:**

- `evidence_ratio ≈ 1.0` from long unique region → strong independent evidence → rescue from penalty.
- `evidence_ratio ≈ 0.0` → confirms annotation-based penalty.

---

## Grouped Momentum Acceleration

MAJEC accelerates EM convergence using expression-level grouped momentum, taking larger steps toward convergence based on iteration history.

### Algorithm

**1. Calculate Velocity (after `--momentum_start` iterations):**

```
velocity[t, i] = θ[t, i] - θ[t, i-1]
```

**2. Apply Grouped Momentum:**

```
θ[t, i+1] = θ[t, i] + momentum_factor[expression_group[t]] × velocity[t, i]
```

**3. Expression-Level Grouping:**

Transcripts are binned into expression groups with different momentum factors:

| Expression Group | Default Factor | Rationale |
|-----------------|----------------|-----------|
| Low | 1.5 | Aggressive acceleration (stable, low counts) |
| Medium | 1.0 | Moderate acceleration |
| High | 0.7 | Conservative (volatile, high counts) |

Controlled by `--momentum_scaling` (e.g., `"1.5 1.0 0.7"`).

### Stability Safeguards

- **Oscillation detection:** Monitors velocity correlation between iterations.
- **Maximum change limits:** Caps step size to prevent overshooting.
- **Graceful degradation:** Falls back to standard EM if instability detected.

Result: typically 2–3x faster convergence vs. standard EM.

---

## Convergence Criteria

EM iteration continues until **all** of the following conditions are met (with a minimum of 15 iterations):

```
1. Total count stability: |Σθ[i+1] - Σθ[i]| / Σθ[i] < ε_rel
2. Per-feature stability: max change among expressed features < adaptive threshold
```

Where:

- `ε_rel = 0.0001` (1e-4).
- Adaptive threshold: max(1.0, 10th percentile of expressed feature counts).
- Maximum iterations: 150 (default, via `--em_iterations`).

---

## Distributional Effective Length Model

- `L_eff` is calculated using a **Normal distribution** of fragment lengths (via `--frag_stats_dir`).
- This accurately models the mappable territory of each transcript, not just a point estimate.

---

## Confidence Metrics

MAJEC calculates multi-dimensional reliability scores that provide both absolute confidence and diagnostic information about the source of ambiguity.

### Transcript-Level Metrics

#### 1. Core Components

**Certain Evidence Fraction:**

```
CertainFrac[t] = unique_fraction[t] + unique_multimapper_fraction[t]
```

Reads with unambiguous genomic assignment (unique mappers + unique junction evidence from multi-mappers).

**Uncertain Evidence Fraction:**

```
UncertainFrac[t] = 1 - CertainFrac[t]
```

Reads requiring EM deconvolution (shared across multiple transcripts).

**Ambiguous Fraction Distinguishability:**

```
Dist_ambig[t] = Σ_classes( weight[C] × |allocation[t,C] - max_other[C]| )
```

How well transcript `t` was separated from competitors *within the uncertain fraction*.

#### 2. Final Holistic Distinguishability Score

The user-facing reliability metric combining certain and uncertain evidence:

```
distinguishability_score[t] = (CertainFrac[t] × 1.0) + (UncertainFrac[t] × Dist_ambig[t])
```

**Interpretation:**

- **1.0:** Perfectly distinguishable (all evidence is certain).
- **0.5–1.0:** Moderately distinguishable.
- **0.0–0.5:** Highly ambiguous.

#### 3. Assignment Entropy (Diagnostic)

Shannon entropy of read source distribution:

```
entropy[t] = -Σ_sources( p[source] × log₂(p[source]) )
confidence_score[t] = 1 / (1 + entropy[t])
```

Where `p[source]` is the proportion of transcript `t`'s count from each equivalence class.

### Group-Level Metrics (Gene/TE Family)

For a group `G` with transcripts `{t₁, ..., tₙ}` and expression counts `{c₁, ..., cₙ}`:

#### 1. Group Distinguishability

Expression-weighted average of transcript-level holistic scores:

```
group_distinguishability[G] = Σᵢ(cᵢ × distinguishability_score[tᵢ]) / Σᵢ(cᵢ)
```

Absolute score of total quantification ambiguity for the gene.

#### 2. Inter-Group Competition

Expression-weighted fraction of ambiguity from external genes:

```
inter_group_competition[G] = Σᵢ(cᵢ × inter_gene_frac[tᵢ]) / Σᵢ(cᵢ)
```

**Diagnostic value:**

- **High (>0.5):** "Contamination problem" — ambiguity from paralogs/pseudogenes.
- **Low (<0.5):** "Complex splice" — ambiguity from isoform complexity within the gene.

#### 3. Holistic Group External Distinguishability Score

The final, user-facing metric for filtering or weighting in downstream analyses. Synthesizes the *total amount* of ambiguity within a gene group with the *source* of that ambiguity (internal vs. external).

**Formula:**

```
holistic_group_external_distinguishability[G] =
    (Certain_Fraction × 1.0) + (Ambiguous_Fraction × Ambiguous_External_Distinguishability)
```

**Components:**

- `group_weighted_shared_fraction`: expression-weighted average of the `shared_read_fraction` across transcripts — the proportion of the gene's count from ambiguous read classes requiring EM resolution.
- `group_total_abs_intergene_dist`: expression-weighted average of `ambiguous_fraction_abs_inter_dist` — how well the ambiguous reads were distinguished from transcripts in *other* genes.

This score answers: "How much of this gene's quantification is potentially contaminated by reads from other genes?" A gene with complex internal splicing but no external competitors scores high (reliably quantified). A gene difficult to distinguish from a paralog scores low (potentially unreliable).

**Score Range:**

- **1.0:** Highly reliable (no ambiguity, or all ambiguity is internal to the gene).
- **0.0:** Highly unreliable (high ambiguity from external gene competition).
