# Algorithm Details

## Overview

MAJEC uses an **Expectation-Maximization (EM) algorithm**, enhanced with evidence-based priors and momentum acceleration, to resolve read ambiguity across a joint feature space of genes, transcript isoforms and individual transposable element (TE) loci. A run has four stages:

1. **A single pass over each BAM** builds junction-aware equivalence classes (EqCs). The same pass tallies annotated splice-junction reads and, for paired-end data, measures the fragment-length distribution.
2. **Junction-informed priors** adjust the initial counts using junction evidence, annotation quality and structural completeness.
3. **A two-phase, momentum-accelerated EM** distributes ambiguous fragments.
4. **Aggregation and reporting** sum the counts to genes and TE subfamilies, and optionally compute TPM and confidence metrics.

The method rests on three ideas:

1. **Junction-aware equivalence classes.** Each fragment's candidates are narrowed to the features whose exons contain it and whose annotated junctions match its splicing.
2. **A hierarchical two-phase EM** that respects the information-quality hierarchy between unique and multi-mapping reads.
3. **A multi-layered prior adjustment system** that brings in splice-junction evidence, annotation quality and structural integrity.

---

## Feature Space

`majec_precompute_annotations` combines a gene GTF and an optional TE GTF into one annotation file. Every transcript in the gene GTF and every TE locus in the TE GTF becomes a **feature**. All features share one exon index, so a read that overlaps both a gene's exon and an embedded TE has both as candidates, and the EM decides between them.

- **Transcripts** aggregate to their `gene_id`.
- **TE loci** are single-interval features, one per annotated insertion. They aggregate to a **`name:family:class`** subfamily ID built from the TE GTF's `gene_id`, `family_id` and `class_id` attributes, for example `AluYa5:Alu:SINE` or `L1HS:L1:LINE`. This follows the TEtranscripts convention. Loci without `family_id`/`class_id` fall back to the bare subfamily name.
- `--te_coordinates` writes `{prefix}_TE_locus_coordinates.tsv`, which maps each TE locus to its coordinates and subfamily. See [Output Files](output_files.md).

The precomputed annotation also holds everything later stages need:
- the exon interval index;
- annotated junctions, with how many transcripts share each one;
- each transcript's junctions in 5′→3′ order;
- subset/superset relationships between isoforms;
- splice-site competitors;
- Transcript Support Level (TSL), when the GTF carries it.

---

## Building Equivalence Classes

### Builders

Two builders apply identical rules and produce the same results:

| Builder | Package | Notes |
|---|---|---|
| Compiled builder (majec-core) | `majec-core`, installed with `pip install "majec[rust]"` or `pip install majec-core` | Optional. Roughly 6× faster on the BAM scan. Required for the long-read options. |
| Pure-Python builder (pysam) | `pysam` (core dependency) | Always available. Used automatically when majec-core is not installed. |

The log records which builder ran. For the compiled builder it also records the majec-core version and a SHA-256 hash of the loaded binary. When caching is on, the same information goes into the `eqc_builder` field of the run manifest (`{prefix}_run_manifest.json`). Each sample's BAM scan is single-threaded, so `--threads` parallelises across samples. See [Installation](installation.md).

### Which alignments are used

- **Unmapped and supplementary records** are skipped.
- **Unique vs. multi-mapping** is decided from the `NH` tag:
  - A fragment is **unique** when its alignments have `NH == 1` (both mates, for pairs) and MAPQ ≥ 30. STAR gives `NH == 1` alignments MAPQ 255 by default, so for STAR the MAPQ requirement removes nothing.
  - `NH == 1` alignments with MAPQ < 30 are not counted.
  - A read with `NH > 1` is a **multimapper**. All of its alignments are used, **including secondary alignments**, and no MAPQ filter is applied.
- **Strand.** With `--strandedness 1` (read 1 on the transcript strand) or `--strandedness 2` (read 1 on the opposite strand), only features on the inferred transcript strand are candidates. With `--strandedness 0`, both strands are queried.

STAR and HISAT2 write `NH`. For aligners that do not, see [Long-Read Data](#long-read-data-experimental).

### Candidate set for one alignment

Each alignment's CIGAR is split into aligned blocks (`M`, `=`, `X`) and splice junctions (`N`). Three sets are computed:

| Set | Definition |
|---|---|
| **O** (exon overlap) | Features with an exon overlapping any aligned block. |
| **C** (exon containment) | Features for which *every* aligned block lies entirely inside one of their exons. Different blocks may sit in different exons. |
| **J** (junction compatibility) | Transcripts that contain *all* of the alignment's annotated junctions (the intersection of each junction's transcript set). |

A junction is annotated when its donor and acceptor coordinates exactly match an intron in the annotation, regardless of strand. Strand specificity comes from the exon sets (O and C).

The candidate set is the first non-empty set in this ladder:

| Alignment crosses an annotated junction? | C non-empty? | Candidate set (first non-empty) |
|---|---|---|
| yes | yes | O ∩ C ∩ J → O ∩ J → O ∩ C → O |
| yes | no | O ∩ J → O |
| no | yes | C |
| no | no | O |

If O is empty, the alignment overlaps no annotated feature and is not counted.

**Novel junctions.** An alignment's unannotated junctions are ignored when it also crosses annotated ones. If *all* of its junctions are novel, it has no J constraint and is handled like an unspliced alignment (C, then O).

**Containment example.** Take a read whose aligned block runs from exon 3 into the following intron. It overlaps every isoform containing exon 3. It is contained only in features where that sequence is exonic: a retained-intron isoform, an isoform using a downstream donor, or an overlapping TE locus. It is assigned among those, and isoforms that splice at the end of exon 3 drop out.

TE loci carry no junctions. A read crossing an annotated junction therefore excludes TE loci whenever an overlapping transcript is junction-compatible.

### Pairs and fragments

**Every fragment is counted once.**

- **Pairing.** Mates are paired by read name and `HI` tag, so each alignment pair of a multimapper is paired at its own locus.
- **Mate intersection.** A pair's candidate set is the intersection of the two mates' sets. If the intersection is empty, the union is used.
- **Unique fragments** add one count to their EqC.
- **Multimapping reads.** Each alignment pair is intersected first. The resulting sets are then unioned across all of the read's alignments, giving **one** multimapper EqC entry per read.
- **Orphan mates.** A mate whose partner is unmapped, or overlaps no feature, is counted on its own.
- **Single-end data.** Each read is a fragment.

Pass `--paired_end` for paired-end libraries. Without it, each mate is counted as a separate single-end read and no fragment-length distribution is measured.

> **Counting unit relative to 0.1.5.** In 0.1.5, the featureCounts-based flow counted each mate of a uniquely mapped pair separately. Multimapping pairs were already counted as fragments. On paired-end data, 0.2.0 unique counts are therefore about half of 0.1.5's. Single-end data is unaffected. See [CHANGELOG.md](../CHANGELOG.md).

### Builder outputs

For each sample, the scan produces:
- **unique EqCs** and **multimapper EqCs** (candidate set → fragment count);
- **initial unique counts**: each unique fragment split evenly over its candidate set, which is the starting point for the priors;
- **annotated-junction counts**, tallied per alignment record. Every alignment that crosses an annotated junction adds one to it, whether unique or multimapping, including secondary alignments. The two mates of a pair are tallied separately. These counts drive the priors below.
- **the measured fragment-length distribution** (paired-end only; see [Effective Length and TPM](#effective-length-and-tpm)).

With `--use_cache`, these results are cached per sample. Prior and EM settings can then be changed without rescanning the BAMs.

---

## Two-Phase Hierarchical EM

MAJEC processes reads in two distinct phases. Unique mappers and multi-mappers carry fundamentally different levels of uncertainty:

### Information Quality Hierarchy

| Read Type | Certainty Level | Primary Ambiguity |
|-----------|----------------|-------------------|
| **Unique mappers** | High-confidence genomic location | Only isoform choice at that locus |
| **Multi-mappers** | Uncertain genomic location | Both locus assignment AND isoform choice |

### Processing Strategy

**Phase 1: Unique Mapper Processing**

```
θ_unique[t, i+1] = E_step(unique_classes, θ_total[t, i])
```

High-confidence genomic assignments are processed first to establish expression patterns at each locus.

**Phase 2: Multi-mapper Processing (Informed by Phase 1)**

```
priors[t] = θ_unique[t, i+1] + θ_multi[t, i]
θ_multi[t, i+1] = E_step(multi_classes, priors[t])
```

Multi-mapping reads are allocated according to the expression patterns established from unique mappers.

**Combined Update:**

```
θ_total[t, i+1] = θ_unique[t, i+1] + θ_multi[t, i+1]
```

**Rationale:**

- Genomic uncertainty and isoform uncertainty are distinct problems.
- High-quality evidence guides the ambiguous assignments.

---

## The E-Step: Read Allocation

**Formula:**

```
P(read | transcript_i) ∝ θ_i
```

**Standard EM Update (per equivalence class):**

```
For transcript t in equivalence class C:
θ[t, i+1] = Σ_classes( reads[C] × θ[t,i] / Σ_j∈C(θ[j,i]) )
```

Single-feature classes go entirely to their feature. A small pseudo-count (1e-6) keeps the division stable. A class whose members all have zero abundance is split evenly.

The EM works on counts. Effective lengths do not enter the allocation; they are applied afterwards, when TPM is computed.

---

## Evidence-Based Prior Initialization

Before the EM begins, the initial unique counts are adjusted by a series of independent, evidence-based priors, applied in order:

```
Initial counts (from the EqC builder)
    ↓
1. Junction Boost (positive evidence from splice junctions)        always on; --junction_weight
    ↓
2. TSL Penalty (down-weight low-confidence annotations)            --use_tsl_penalty
    ↓
3. Junction Completeness Penalty (penalize structural incompleteness)   --use_junction_completeness
    ↓
4. Subset Isoform Penalty (resolve fragment vs. complete transcripts)   --use_subset_penalty
    ↓
Final Adjusted Priors → EM Algorithm
```

Priors set the abundances the EM starts from. They do not change the EqCs. They have most influence where the classes alone cannot separate features, for example an isoform whose exons and junctions are all shared with a longer isoform.

With `--verbose_output`, every adjustment is recorded per sample in `{prefix}_{sample}_prior_adjustments.tsv.gz`. See [Output Files](output_files.md).

**Which priors matter in 0.2.0:**
- **Completeness penalty:** remains important.
- **Junction boost:** has much less effect than in 0.1.5, because annotated junctions now constrain each fragment's candidate set when the classes are built.
- **Subset penalty:** has roughly half its 0.1.5 influence.

---

## Prior Adjustment Models

### 1. Junction Evidence Boost

Transcripts receive positive evidence from splice-junction support, weighted by junction uniqueness.

**Annotation-Level Uniqueness:** During preprocessing, every junction `j` is labeled with `n_transcripts`, the number of transcripts sharing it.

**Read-Based Evidence Calculation:**

```
boost[t] = junction_weight × Σ_j∈t( count[j] × (1/n_transcripts[j])^decay_exponent )
adjusted_count[t] = initial_count[t] + boost[t]

Where:
- count[j]: Number of alignment records spanning annotated junction j
- n_transcripts[j]: Number of transcripts containing junction j
- decay_exponent: Penalty for shared junctions (--junction_decay_exponent, default: 1.0)
- junction_weight: Global importance parameter (--junction_weight, default: 3.0)
```

Junction counts include multimapping alignments that carry annotated junctions. `--junction_weight 0` disables the boost.

**Example:** A junction shared by 10 transcripts with `decay_exponent=2`:

```
weight = (1/10)^2 = 0.01 (1% of the evidence per transcript)
```

### 2. TSL Penalty (`--use_tsl_penalty`)

Down-weights transcripts based on GENCODE/Ensembl Transcript Support Level annotations. TSL is read from the GTF's `transcript_support_level` (or `tsl`) attribute.

**Default Penalty Values:**

```
TSL1: 1.0   (highest confidence, no penalty)
TSL2: 0.9
TSL3: 0.7
TSL4: 0.5
TSL5: 0.3
NA:   0.8   (unknown quality)
```

Applied as: `adjusted_count[t] = count[t] × tsl_penalty[t]`.

Features without a TSL get the `NA` value. This includes TE loci, and every transcript in annotations that carry no TSL. Use this prior with annotations that provide TSL. Override any value with a JSON string, for example `--tsl_penalty_values '{"NA":1.0}'`.

### 3. Junction Completeness Penalty (`--use_junction_completeness`)

A statistical model that penalizes multi-exon transcripts whose annotated junctions are unevenly observed, a sign that the reads belong to a different isoform. Single-exon transcripts and TE loci are not penalized.

**Algorithm:**

```
For each multi-exon transcript (junction counts in 5'→3' order):
1. If no junction has reads → penalty = --junction_completeness_min_score
2. Baseline: median of the non-zero junction counts
   (root-mean-square instead if the largest count exceeds 50× the median)
3. Splice competition: for a junction that shares a donor or acceptor with another
   annotated junction, expected = max(baseline, median count of its competitors)
4. Terminal dropout modeling (--terminal_relax, needs --library_type and ≥3 junctions):
   - dT / polyA: expected 5' dropout; random: expected 3' dropout
   - find the first junction from that end with ≥80% of the median and a rising trend;
     inside the dropout zone, expected = median × (1 − exp(−r / distance)),
     r = --terminal_recovery_rate (default: 2)
5. Z-scores for the three lowest-count junctions:
   z = (observed − expected) / sqrt(expected + φ × expected²)
   φ = --junction_completeness_overdispersion (default: 0.05; halved at splice competitors)
6. Combine the z-scores (weights 0.6 / 0.3 / 0.1). Severity is amplified when several
   junctions have z < −2, and the result is never more lenient than the worst junction.
7. If the worst junction is below its expectation:
   penalty = ((observed + 0.1) / (expected + 0.1))^e
   e rises from 0.2 (z ≥ −1.5) to 1.0 (z ≤ −6); otherwise penalty = 1
```

**Final penalty range:** from `--junction_completeness_min_score` (default 0.0001) to 1.0.

Set `--library_type` (`dT`, `polyA`, `random` or `none`) to match the protocol. The run exits if `--terminal_relax` is given without it, and `none` disables terminal modeling.

### 4. Subset Isoform Penalty (`--use_subset_penalty`)

Resolves ambiguity between fragment isoforms and complete transcripts. During preprocessing, a multi-exon transcript is marked as a **subset** when all of its annotated junctions also appear in another transcript that has more junctions (its superset).

**Logic:**

```
For each subset transcript s with reads:
1. For each other transcript sharing s's junctions, estimate a per-junction baseline
   from its independent junctions (those not in s): the mean of the junction reads
   attributable to it, given its share of the current counts at each junction.
2. Expected reads at s's junctions = sum of those baselines at each junction,
   multiplied by --subset_evidence_threshold (default: 1.25).
3. z = (observed − expected) / sqrt(expected + φ_eff × expected²)
   φ_eff scales --subset_penalty_overdispersion (default: 0.1) up where expected
   coverage per junction is low.
4. z < 0       → penalty = --subset_penalty_min (default: 0.001)
   0 ≤ z < 2   → penalty rises linearly to 1
   z ≥ 2       → no penalty (the subset has excess junction evidence of its own)
```

`--subset_evidence_threshold` raises the bar for what counts as excess evidence.

When a transcript has both a completeness penalty `a` and a subset penalty `b`, they are combined as `(min(a, b) + a × b) / 2` rather than multiplied.

> In 0.1.5, the subset penalty had a coverage-based second stage that used reads in a transcript's unique exonic territory (`--use_subset_coverage_data`), and the completeness model had paired-junction rescue. In 0.2.0 both were replaced by exon containment in the EqC builder, and their options were removed. See [CHANGELOG.md](../CHANGELOG.md).

---

## Grouped Momentum Acceleration

After a few standard EM iterations, MAJEC switches to an expression-level grouped momentum phase that extrapolates from the iteration history.

### Algorithm

**1. Calculate Velocity (after `--momentum_start` standard iterations, default 4):**

```
velocity[t, i] = θ[t, i] - θ[t, i-1]
```

**2. Apply Grouped Momentum:**

```
θ[t, i+1] = θ[t, i] + momentum[expression_group[t]] × velocity[t, i]
momentum[g] = 0.3 × momentum_scaling[g], adapted per iteration (see below), capped at 0.6
```

**3. Expression-Level Grouping:**

Features are binned into expression groups, each with its own momentum scaling factor:

| Expression Group | Default Scaling | Rationale |
|-----------------|----------------|-----------|
| Low | 1.5 | Aggressive acceleration (stable, low counts) |
| Medium | 1.0 | Moderate acceleration |
| High | 0.7 | Conservative (volatile, high counts) |

Controlled by `--momentum_scaling LOW MEDIUM HIGH` (e.g. `--momentum_scaling 1.5 1.0 0.7`).

The group boundaries are the 25th and 75th percentiles of the non-zero prior-adjusted counts, pooled across all samples in the run. They are written to `{prefix}_joint_binning_info.json`. To process a cohort in batches with identical groups, pass fixed thresholds with `--use_bins`, using either that file from an earlier run or the output of `majec_calc_thresholds`.

### Stability Safeguards

- **Oscillation detection:** Velocity is compared with the previous iteration's within each group. Momentum is used at full strength when the two agree (correlation > 0.5), reduced to 30% when they oscillate (< −0.5), and set to 60% otherwise.
- **Maximum change limits:** A momentum step that would change any feature by more than 100% is rejected.
- **Graceful degradation:** A rejected or failed momentum step falls back to a standard EM step.

### What `--momentum_start` controls

During the momentum phase the totals advance by the momentum step itself. The E-step on the extrapolated point is used only to split each feature's count into its unique and multimapping parts; a full EM step is taken only when a momentum step is rejected. Each momentum step is a fraction of the one before, so the momentum phase settles the estimates near the point reached by the standard EM iterations from the prior-adjusted start. It does not iterate EM to its fixed point. Iterated to that point, EM on overlapping isoforms tends to concentrate a gene's counts in a single winning isoform.

`--momentum_start` (default 4) is therefore a dial:

- **Higher values:** more standard EM iterations let the data move estimates further from the prior-informed start, toward the maximum-likelihood solution and its winner-take-all tendency.
- **Lower values:** estimates stay closer to the prior-informed start.

With the 15-iteration minimum (see [Convergence Criteria](#convergence-criteria)), a default run takes 4 standard iterations followed by about 11 momentum iterations. The defaults are the settings used for MAJEC's published benchmarks.

---

## Convergence Criteria

After a minimum of 15 iterations, EM stops when **all** of the following hold:

```
1. Total count stability:   |Σθ[i+1] - Σθ[i]| / Σθ[i] < ε
2. Among meaningful features (count ≥ adaptive threshold), relative change per feature:
   max < 5ε,   95th percentile < 2ε,   mean < 0.5ε
```

Where:

- `ε = 0.0001` (1e-4).
- Adaptive threshold: max(1.0, 10th percentile of expressed feature counts).
- Maximum iterations: 150 (default, via `--em_iterations`). The log warns if a sample stops there without converging.

---

## Effective Length and TPM

Counts are the primary output and do not depend on fragment length. `--output_tpm` adds TPM, which needs an effective length for each feature.

### Measured fragment-length distribution

On paired-end data (`--paired_end`), the builder measures the fragment-length distribution during its BAM scan, in transcript (cDNA) coordinates:

1. **Which pairs.** Only pairs whose mates are both uniquely mapped (`NH == 1`, MAPQ ≥ 30) on the same chromosome are measured.
2. **cDNA length.** The pair's outer genomic bounds are converted to cDNA coordinates on each feature in the pair's candidate set, which removes introns. A feature is skipped when either end falls outside its exons, for example in an intron or past the annotated ends. Intron length therefore never enters the measurement.
3. **Weighting.** The pair's weight of 1 is split evenly over the k features that can place it. Each adds its own cDNA length to the histogram with weight 1/k.

Keeping only pairs compatible with a single transcript would be a length-dependent filter. A longer fragment spans more structure and is more often uniquely assignable, so that rule over-represents long fragments. The uniform split matches how Salmon and RSEM seed their own distributions.

The distribution is kept as an untruncated histogram rather than fitted to a Normal. The real distribution is right-skewed, with the mean above the median. In testing, the measured distributions agreed closely with Salmon's and RSEM's estimates for the same libraries. Measuring in the scan removes a separate preprocessing step and keeps intron length out of the estimate. It is a correctness and convenience change, not a claimed gain in TPM accuracy.

The histogram is stored in the per-sample cache, so a cached run reproduces the same effective lengths. When TPM is computed, the log records each sample's fragment-length mean and SD.

### Effective length

Effective length models the number of positions a fragment can start from, using the whole fragment-length distribution rather than a point estimate:

```
L_eff[t] = L[t] − E[d | d ≤ L[t]] + 1
```

The expectation is **conditioned on the feature's own length**: the distribution is renormalized over the fragments the feature is long enough to produce. Features shorter than the shortest observed fragment get `L_eff = 1`, the limit of the formula. Effective lengths are computed per sample from that sample's distribution.

### Overrides and fallbacks

Each sample's effective lengths come from the first source available:

| Source | When | Effective length |
|---|---|---|
| `--frag_stats_dir DIR` | Override | Reads `<BAM name without .bam>_cDNA_frag_stats.json` (mean and SD) written by `majec_calc_frag_len`. Uses a Normal truncated at ±4 SD, renormalized by P(d ≤ L) to give the same conditional quantity. If a sample's file is missing, the measured distribution is used (or annotated lengths, if there is none). |
| `--mean_fragment_length N` | Override | A single fixed length for all samples: `L_eff = max(L − N + 1, 1)`. Intended for single-end data. |
| Measured distribution | Default for paired-end | As above. |
| Annotated length | Single-end data with no override | A warning is logged. |

`--frag_stats_dir` and `--mean_fragment_length` are mutually exclusive. The `majec_calc_frag_len` statistics are estimated from genomic `TLEN`, while the in-scan measurement works in transcript coordinates, so the two can differ on spliced libraries; the override is mainly useful for reproducing runs made with earlier versions. A warning is logged when it overrides a measured distribution.

### TPM

```
TPM[t] = ( count[t] / L_eff[t] ) / Σ_j( count[j] / L_eff[j] ) × 10^6
```

TPM is computed per sample over all features: transcripts and TE loci together. Gene and TE subfamily TPMs (`{prefix}_total_EM_aggregated_TPM.tsv`) are sums of their members' TPMs.

**Caveats:**

- **3′-biased libraries.** In 3′-biased polyA libraries, long transcripts yield fewer reads per molecule than their length implies. Any length correction, MAJEC's or Salmon's, then over-corrects their TPM. Compare long and short transcripts by TPM with care in such libraries.
- **Use counts, not TPM, for TE loci.** TE loci are genomic intervals rather than transcripts, and very short loci get inflated TPM. Use counts for locus- and subfamily-level TE analyses. See [Post-Quantification Workflows](post_quantification.md).

---

## Confidence Metrics

With `--output_confidence`, MAJEC calculates multi-dimensional reliability scores. They give both absolute confidence and diagnostic information about the source of ambiguity. Metrics are computed for features with a final count of at least `--confidence_min_expression` (default 1.0).

### Transcript-Level Metrics

#### 1. Core Components

**Certain Evidence Fraction:**

```
CertainFrac[t] = unique_fraction[t] + unique_multimapper_fraction[t]
```

The fraction of `t`'s reads that come from single-feature classes. These are unique fragments compatible only with `t`, plus multimapping reads whose alignments all resolve to `t` alone.

**Uncertain Evidence Fraction:**

```
UncertainFrac[t] = 1 - CertainFrac[t]
```

Reported as `shared_read_fraction`. These are reads from classes shared with other features, which need EM deconvolution.

**Ambiguous Fraction Distinguishability:**

```
Dist_ambig[t] = Σ_C( w[C] × mean_j∈C,j≠t |a[t,C] − a[j,C]| ) / Σ_C( w[C] )

a[j,C] = θ_j / Σ_k∈C(θ_k)      (share of class C allocated to feature j)
w[C]   = reads[C] / θ_t
```

The sums run over the shared classes containing `t`. The score measures how well `t` was separated from its competitors *within the uncertain fraction*.

#### 2. Final Holistic Distinguishability Score

The user-facing reliability metric combines certain and uncertain evidence:

```
distinguishability_score[t] = (CertainFrac[t] × 1.0) + (UncertainFrac[t] × Dist_ambig[t])
```

**Interpretation:**

- **1.0:** Perfectly distinguishable (all evidence is certain).
- **0.5–1.0:** Moderately distinguishable.
- **0.0–0.5:** Highly ambiguous.

#### 3. Assignment Entropy (Diagnostic)

Shannon entropy of the read source distribution:

```
entropy[t] = -Σ_sources( p[source] × log₂(p[source]) )
confidence_score[t] = 1 / (1 + entropy[t])
```

Where `p[source]` is the proportion of transcript `t`'s count from each equivalence class.

### Group-Level Metrics (Gene/TE Subfamily)

These are computed with `--calculate_group_confidence`, which requires `--output_confidence`. For a group `G` with expressed members `{t₁, ..., tₙ}` and counts `{c₁, ..., cₙ}`, a bar denotes an expression-weighted mean over members, `x̄ = Σᵢ(cᵢ × xᵢ) / Σᵢ(cᵢ)`.

#### 1. Group Distinguishability

```
holistic_group_distinguishability[G] = (1 − S̄) + S̄ × D̄
S̄ = mean shared_read_fraction,   D̄ = mean Dist_ambig
```

An absolute score of the total quantification ambiguity for the group.

#### 2. Inter-Group Competition

The expression-weighted fraction of ambiguity that comes from other genes or TE subfamilies:

```
inter_group_competition[G] = Σᵢ(cᵢ × inter_gene_frac[tᵢ]) / Σᵢ(cᵢ)
```

**Diagnostic value:**

- **High (>0.5):** a "contamination problem": ambiguity from paralogs, pseudogenes or overlapping TEs.
- **Low (<0.5):** a "complex splice" case: ambiguity from isoform complexity within the gene.

#### 3. Holistic Group External Distinguishability Score

The final, user-facing metric for filtering or weighting in downstream analyses. It combines the *total amount* of ambiguity within a group with the *source* of that ambiguity (internal vs. external).

**Formula:**

```
holistic_group_external_distinguishability[G] =
    (Certain_Fraction × 1.0) + (Ambiguous_Fraction × Ambiguous_External_Distinguishability)
```

**Components:**

- `group_weighted_shared_fraction`: the expression-weighted average of `shared_read_fraction` across members. This is the proportion of the group's count that came from ambiguous classes needing EM resolution. `Certain_Fraction` is one minus this.
- `group_total_abs_intergene_dist`: the expression-weighted average of `ambiguous_fraction_abs_inter_dist`. It measures how well the ambiguous reads were distinguished from features in *other* groups. Classes shared only within the group count as fully distinguished.

This score answers: "How much of this gene's quantification is potentially contaminated by reads from other genes?" A gene with complex internal splicing but no external competitors scores high (reliably quantified). A gene that is hard to distinguish from a paralog scores low (potentially unreliable).

**Score Range:**

- **1.0:** Highly reliable (no ambiguity, or all ambiguity is internal to the gene).
- **0.0:** Highly unreliable (high ambiguity from external gene competition).

---

## Long-Read Data (Experimental)

> **Experimental.** Long-read quantification has been checked for correctness on ONT libraries. Its accuracy against dedicated long-read quantifiers has not been established.

MAJEC reads alignment multiplicity from the `NH` tag. minimap2 does not write `NH`, so for such BAMs the compiled builder (majec-core) can **derive multiplicity from read names**:

- **Switching it on.** `--derive_multiplicity auto` (default) turns derive mode on only if none of the first 10,000 mapped records carries `NH`, so short-read runs are unaffected. `on` forces it and `off` disables it.
- **Buffering.** In derive mode, every primary and secondary alignment that overlaps an annotated feature is buffered under its read name. Each is stored with its candidate set (built by the same rules as above), alignment score (`AS`) and MAPQ. The BAM is treated as single-end.
- **Classification.** At the end of the scan, each read is classified:
  - If `--lr_score_ratio` is above 0, only alignments scoring within that fraction of the read's best `AS` are kept. The default, 0, keeps every reported alignment. Raising it hands each read to its top-scoring locus, which on noisy long reads can move reads between a gene and its pseudogene copies.
  - A read with one surviving alignment is unique. It is subject only to `--lr_min_mapq` (default 0, off), not to the short-read MAPQ ≥ 30 rule.
  - A read with several surviving alignments becomes one multimapper EqC entry (the union of their candidate sets), with no MAPQ filter. The EM resolves it.
- **TPM.** No fragment-length distribution is measured, so TPM uses annotated lengths unless `--mean_fragment_length` is given.

These options need majec-core. The pure-Python builder (pysam) refuses `--derive_multiplicity on`, `--lr_score_ratio` and `--lr_min_mapq` rather than ignoring them. If more than 1% of primary alignments lack `NH`, it logs a warning: those reads are counted as unique mappers and their secondary alignments are discarded.
