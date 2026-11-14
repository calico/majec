# Output Files

Every file `majec_run_pipeline` writes starts with `--prefix`. Per-sample files add the sample name: `{prefix}_{sample}_...`. Sample names come from the BAM file names (see [Usage Guide](usage_guide.md#sample-names)).

## Units

Counts are estimated numbers of **fragments** for paired-end data (each read pair counts once) and of **reads** for single-end data.

- **Fractional values.** The EM splits ambiguous fragments between transcripts, so counts are fractional.
- **Totals.** Each sample's counts sum to approximately the number of fragments assigned to equivalence classes (EqCs).
- **Difference from 0.1.5.** In 0.1.5, each mate of a uniquely mapped pair counted separately. Paired-end unique counts are therefore about half of 0.1.5's, so don't mix count matrices from the two versions.

TPM values sum to one million per sample.

## Overview

| File | Written when | Skipped by `--light` |
|---|---|---|
| `{prefix}_total_EM_counts.tsv` | always | |
| `{prefix}_total_EM_aggregated_counts.tsv` | always | |
| `{prefix}_total_EM_TPM.tsv` | `--output_tpm` | |
| `{prefix}_total_EM_aggregated_TPM.tsv` | `--output_tpm` | |
| `{prefix}_TE_locus_coordinates.tsv` | `--te_coordinates` and a TE annotation | |
| `{prefix}_unique_mapper_EM_counts.tsv.gz`, `{prefix}_multimapper_EM_counts.tsv.gz` | `--verbose_output` | yes |
| `{prefix}_pipeline.log` | always | |
| `{prefix}_run_manifest.json` | a cache is used or written (`--use_cache`) | |
| `{prefix}_joint_binning_info.json` | always | |
| `{prefix}_{sample}_prior_adjustments.tsv.gz` | `--verbose_output` | yes |
| `{prefix}_prior_adjustment_summary.tsv` | `--verbose_output` | yes |
| `{prefix}_{sample}_junction_completeness_stats.json` | `--use_junction_completeness` | yes |
| `{prefix}_{sample}_transcript_metrics_SPARSE.tsv.gz` | `--output_confidence` | yes |
| `{prefix}_{sample}_high_confidence_counts.tsv.gz` | `--output_confidence` | yes |
| `{prefix}_transcript_confidence_summary.tsv.gz` | `--output_confidence` | yes |
| `{prefix}_{sample}_counts_with_confidence.tsv.gz` | `--output_confidence` + `--verbose_output` | yes |
| `{prefix}_{sample}_distinguishability_detailed.tsv.gz` | `--output_confidence` + `--verbose_output` | yes |
| `{prefix}_group_confidence_comprehensive.tsv.gz`, `{prefix}_group_confidence_summary.json` | `--calculate_group_confidence` (needs `--output_confidence`) | yes |
| `{prefix}_competition_summary.tsv`, `{prefix}_complex_competition_genes.tsv` | `--analyze_competition` (needs `--output_confidence`) | yes |
| `{prefix}_{sample}_em_classes.pkl` | `--dump_em_classes` | yes |

`--light` turns off every option marked "yes" and skips prior tracking. Counts and TPM are unchanged.

---

## Count and TPM matrices

| File | Rows | Values |
|---|---|---|
| `_total_EM_counts.tsv` | Every transcript and TE locus in the annotation, including zeros | Final EM counts, rounded to 6 decimals |
| `_total_EM_TPM.tsv` | Same as above | TPM |
| `_total_EM_aggregated_counts.tsv` | Genes (by GTF `gene_id`), then TE subfamilies (`name:family:class`) | Sum of the counts of the group's transcripts or loci |
| `_total_EM_aggregated_TPM.tsv` | Same as above | Sum of the group's TPM |

All four are tab-separated. The first column is the feature ID, and there is one column per sample, in `--bams` order. Transcript and TE locus IDs are the `transcript_id` values from the GTFs.

**Unique/multimapper split** (`--verbose_output`). `_unique_mapper_EM_counts.tsv.gz` and `_multimapper_EM_counts.tsv.gz` have the same layout as `_total_EM_counts.tsv`. They split each final count into the part from uniquely mapped fragments (NH = 1, MAPQ ≥ 30) and the part from multimapping fragments; the two add up to approximately the total.

**TPM effective lengths.** On paired-end data, effective lengths use the fragment-length distribution measured during the BAM scan, unless `--frag_stats_dir` or `--mean_fragment_length` is given; single-end data uses annotated lengths. Two caveats:

- On 3′-biased poly(A) libraries, length correction over-corrects long transcripts.
- For TE loci, use counts rather than TPM: very short loci get inflated TPM.

See [Usage Guide](usage_guide.md#tpm-and-effective-lengths).

### TE Subfamily Naming

TE counts are aggregated using a **`name:family:class`** composite identifier
(e.g. `AluYa5:Alu:SINE`, `L1HS:L1:LINE`), matching the TEtranscripts convention.
The composite ID disambiguates subfamily names that recur across classes. For
example, `MER121` exists in both the `TcMar` and `hAT` families, and a bare
subfamily name would merge the two into a single, incorrect row.

> **Note:** This naming is baked into the precomputed annotation, and 0.2.0
> annotation files are needed in any case. **Regenerate annotations with
> `majec_precompute_annotations`** if they were built with an older MAJEC
> version. The TE GTF must carry `family_id` and `class_id` attributes (as in
> the standard TEtranscripts `rmsk_TE.gtf`); loci lacking them fall back to the
> bare subfamily name.

### TE Locus Coordinates (`_TE_locus_coordinates.tsv`)

Written only when `--te_coordinates` is passed and the annotation includes a TE
GTF. It has one row per individual TE locus, so each quantified TE instance can
be mapped back to the genome. **This table is large**: one row per TE locus in
the annotation (~4.7M rows / ~260 MB for the hg38 rmsk annotation), so it is off
by default.

| Column | Description |
|:-------|:------------|
| `locus_id` | Individual TE locus ID; matches a row index in `_total_EM_counts.tsv` (e.g. `L1HS_dup1234`). |
| `chr` | Chromosome. |
| `start` | Locus start (1-based, inclusive; genomic bounds taken from the GTF). |
| `end` | Locus end (1-based, inclusive). |
| `strand` | `+` or `-`. |
| `subfamily` | The `name:family:class` subfamily this locus aggregates into; matches a row in `_total_EM_aggregated_counts.tsv`. |

Rows are sorted by `chr`, `start`, `end`. Join on `locus_id` to the locus-level
counts, or on `subfamily` to the aggregated counts.

---

## Run records

### Pipeline log (`_pipeline.log`)

The log is everything printed to the console, with timestamps. It is overwritten on each run with the same prefix. In order, it records:

1. **Version banner.** For example `MAJEC version: 0.2.0`.
2. **EqC builder.**
   - With the compiled builder (majec-core): `EqC builder: Rust majec_core 0.2.0 (sha256 a0a66b9495f48f67)`, giving the majec-core version and the start of the binary's SHA-256.
   - With the pure-Python builder (pysam): a warning that the fallback is in use, gives the same results and scans roughly 6× slower.
3. **The full command line.**
4. **Run settings.**
   - Number of BAMs and threads, strandedness, paired-end mode and prefix.
   - Whether light mode is on.
   - With `--output_tpm`, which fragment-length source TPM will use.
5. **The annotation.** Transcript count, GTF names, creation date, TSL distribution and junction counts.
6. **Per-sample cache status.** `No cache found`, `Valid cache found`, `Cache invalidated due to <key> change`, or `Cache saved to <path>`.
7. **Per-sample scan summary.** With the compiled builder:

   ```
   [sample1] Rust BAM scan: 118,542,500 reads in 185.3s (640K/s)
   [sample1] Junction stats: 32,547,358 spliced reads — 21,181,789 refined (65.1%), 362,796 novel (1.1%), 9,029 incompatible (0.0%)
   [sample1] Containment stats: 49,014,677 assigned reads — 15,886,030 refined (32.4%), 28,407 incompatible (0.1%)
   [sample1] Fragment length: n=45,584,872, mean=214.2, sd=109.9, median=194, p90=321, p99=504
   [sample1]   capped at 1000bp: mean=212.1, sd=84.5 (56,466 longer, 608,379 unique-pair spans unplaceable in exons)
   [sample1] EqC builder complete: 507,793 unique classes (47,847,331 reads), 57,221 multi classes (1,352,936 reads), 163,451 transcripts with junction metrics
   ```

   - **Junction stats** show how spliced alignments fared:
     - *refined*: annotated-junction compatibility narrowed the candidate transcripts;
     - *novel*: none of the read's junctions is annotated;
     - *incompatible*: the junctions fit none of the read's exon-overlap candidates, so the junction filter was not applied to that read.
   - **Containment stats** show the same for exon containment (reads overhanging an exon boundary drop transcripts they don't fit).
   - Treat these percentages as rough diagnostics. Refinements are tallied per alignment, so on paired-end data they can exceed 100%.
   - **Fragment length** summarises the measured fragment-length distribution. The capped line limits it to 1,000 bp, the range Salmon and RSEM model. It is absent for single-end data.
   - The **EqC builder complete** line gives the number of unique and multimapping EqCs. Its "reads" are fragments for paired-end data.
   - **Pure-Python builder.** It logs `BAM scan complete` and the EqC summary, but not the junction, containment and fragment-length lines.
   - **Long reads.** In long-read derive mode, a `Multiplicity derived by read name (no NH tag)` line reports how many reads were unique and how many multimapping.
8. **Priors.** Junction boosts, TSL penalties by level, the completeness model summary (terminal recovery, penalty severity) and subset penalty strategies.
9. **Momentum bins.** The low and high expression thresholds, or the file given to `--use_bins`.
10. **EM progress.** Iterations, convergence checks, `Converged after N iterations` and momentum statistics, for each sample.
11. **Effective lengths.** For each sample, the source used. For example: `Using measured fragment distribution (n=... pairs, mean=..., sd=..., ... bins) for effective lengths`. Otherwise, the fallback and why.
12. **Results.** Confidence summaries, the path of every output written, and the total run time.

### Run manifest (`_run_manifest.json`)

The run manifest records the MAJEC version and which EqC builder ran (`eqc_builder`), and indexes the run's files for `majec_build_db`. It is written on every run. Without `--use_cache`, `cache_files` is empty, and `majec_build_db` then leaves its junction-coverage tables empty.

```json
{
    "run_prefix": "results/my_experiment",
    "majec_version": "0.2.0",
    "eqc_builder": {
        "builder": "rust (majec_core)",
        "majec_core_version": "0.2.0",
        "majec_core_binary": "/path/to/site-packages/majec_core/majec_core.cpython-311-x86_64-linux-gnu.so",
        "majec_core_sha256": "a0a66b94..."
    },
    "run_timestamp_utc": "2026-11-06T21:10:44.170168+00:00",
    "annotation_file": "/abs/path/ann_annotations.pkl.gz",
    "cache_files": { "sample1": "/abs/path/pipeline_cache/sample1_<key>.pkl" },
    "summary_files": {
        "transcript_counts": ".../my_experiment_total_EM_counts.tsv",
        "transcript_tpm": ".../my_experiment_total_EM_TPM.tsv",
        "aggregated_counts": ".../my_experiment_total_EM_aggregated_counts.tsv",
        "aggregated_tpm": ".../my_experiment_total_EM_aggregated_TPM.tsv",
        "group_confidence": ".../my_experiment_group_confidence_comprehensive.tsv.gz",
        "te_locus_coordinates": ".../my_experiment_TE_locus_coordinates.tsv"
    },
    "per_sample_files": {
        "sample1": {
            "priors": ".../my_experiment_sample1_prior_adjustments.tsv.gz",
            "metrics_sparse": ".../my_experiment_sample1_transcript_metrics_SPARSE.tsv.gz",
            "confidence_dense": ".../my_experiment_sample1_counts_with_confidence.tsv.gz",
            "high_confidence": ".../my_experiment_sample1_high_confidence_counts.tsv.gz"
        }
    }
}
```

- **`majec_version` and `eqc_builder`** record what produced the run. With the pure-Python builder, `eqc_builder` is `{"builder": "python (pysam fallback)", "pysam_version": "..."}`.
- **`summary_files`** always lists the four matrices and the group-confidence file. `te_locus_coordinates` is added with `--te_coordinates`. Paths are absolute, and the log warns about any listed summary file that was not produced.
- **`per_sample_files`** is filled only with `--output_confidence`; otherwise it is `{}`.

### Momentum binning (`_joint_binning_info.json`)

This file holds the expression thresholds used to group transcripts for momentum acceleration (`global_thresholds.low_threshold` and `high_threshold`, the 25th and 75th percentiles across samples), together with the samples processed and the EM settings. It can be passed to `--use_bins` in a later run.

---

## Prior reports

### Priors & Penalties Report (`_prior_adjustments.tsv.gz`)

Written per sample with `--verbose_output` (not in `--light` mode). This is a detailed log of every adjustment made to a transcript's initial count before the EM, showing the effect of each penalty and boost.

- **Rows:** transcripts with a non-zero initial count, indexed by transcript ID.
- **Columns:** they appear only for the models that ran:
  - TSL columns need `--use_tsl_penalty`;
  - completeness columns need `--use_junction_completeness`;
  - subset columns need `--use_subset_penalty`.
- **No territory columns.** The 0.1.5 `territory_evidence_ratio` and `territory_confidence` columns no longer exist.

**Counts and boosts**

| Column | Description |
|:--|:--|
| `initial_count` | Count from uniquely mapped fragments before any adjustment. A fragment compatible with *k* transcripts contributes 1/*k* to each. |
| `raw_junction_evidence` | Junction evidence before `--junction_weight`: junction reads, each down-weighted by how many transcripts share the junction (`--junction_decay_exponent`). |
| `junction_weight` | The `--junction_weight` used. |
| `junction_boost` | Additive boost: `raw_junction_evidence × junction_weight`. |
| `post_junction_count` | Count after the junction boost. |

**TSL** (`--use_tsl_penalty`)

| Column | Description |
|:--|:--|
| `tsl` | Transcript Support Level from the annotation (`1`–`5` or `NA`). |
| `tsl_penalty` | Multiplicative factor for that level. It is 1.0 when TSL penalties are off. |

**Junction completeness** (`--use_junction_completeness`)

| Column | Description |
|:--|:--|
| `n_expected_junctions` / `n_observed_junctions` | Annotated junctions of the transcript, and how many were seen in the data. |
| `junction_observation_rate` | `n_observed_junctions / n_expected_junctions`. |
| `completeness_penalty` | Multiplicative factor (0–1); below 1.0 means some junctions were missing or under-covered. It is 1.0 when the model is off. |
| `completeness_model` | How the expectation was set. Values: `median`, `median_with_competition` (alternative splice sites raised the expectation), `RMS` (very uneven coverage), `terminal_recovery_<library_type>` (terminal relaxation applied), `simple_complete`, `single_exon`, `no_junctions`, `no_data`, `none`. |
| `terminal_recovery_used` | True if terminal relaxation was applied. |
| `recovery_point` | Junction index (5′→3′) where coverage recovers from terminal drop-off; -1 if none. |
| `RMS_used` | True if the root-mean-square junction coverage replaced the median as the expectation. |
| `median_coverage` | The junction coverage used as the expectation. |
| `z_score` | Combined z-score of the worst-covered junctions. |
| `worst_junction_position` | Index (5′→3′) of the worst-covered junction; -1 if none. |
| `n_worst_junctions` | Number of worst junctions considered (up to 3). |
| `worst_junction_positions`, `worst_junction_counts`, `worst_junction_z_scores` | Comma-separated positions, read counts and z-scores of those junctions. |
| `worst_expected_vs_actual` | Comma-separated `actual/expected` read counts for those junctions. |
| `pure_median_1pct` | True if a junction had under 1% of the median coverage. |
| `used_splice_competition`, `n_competing_junctions` | Whether competing splice sites informed the expectation, and how many of the transcript's junctions have competitors. |
| `post_completeness_count` | Count after the completeness penalty. |

**Subset isoforms** (`--use_subset_penalty`)

| Column | Description |
|:--|:--|
| `is_subset` | True if the transcript's junction chain is contained in another isoform's. |
| `original_subset_penalty` | Factor (0–1) from the subset model, based on the subset's junction evidence relative to its superset partners. |
| `subset_z_score` | z-score behind `original_subset_penalty`. |
| `adjusted_subset_penalty` | Final factor for subset isoforms. It equals `original_subset_penalty` when there was no completeness penalty; otherwise it is the two combined as (min(*a*, *b*) + *a*·*b*) / 2, applied in place of the completeness penalty. |
| `penalty_combination_strategy` | `comb_completeness_subset` or `subset_only`. |

**Summary**

| Column | Description |
|:--|:--|
| `final_count` | Count after every boost and penalty: the prior that starts the EM. |
| `total_penalty` | Product of `tsl_penalty`, `completeness_penalty` and `subset_penalty`. |
| `subset_penalty` | Always 1.0 in this release. The subset factor actually applied is `adjusted_subset_penalty`, and its effect is included in `final_count`. |
| `total_prior_adjustment` | `final_count / initial_count`. Above 1.0, the transcript was boosted overall; below 1.0, penalized overall. |
| `had_junction_boost` | `junction_boost > 0`. |
| `had_penalty` | `total_penalty < 1`. |
| `penalty_types` | Underscore-joined list of the penalties below 1.0 among `tsl`, `completeness` and `subset` (for example `tsl_completeness`), or `none`. |

### Prior adjustment summary (`_prior_adjustment_summary.tsv`)

Written with `--verbose_output`, one row per sample:

- `sample`
- `transcripts_total`, `transcripts_with_reads`, `transcripts_boosted`, `transcripts_penalized`
- `tsl_penalties`, `completeness_penalties`, `subset_penalties` (counted from `subset_penalty`, so 0 in this release)
- `total_initial_reads`, `total_final_reads`
- `median_boost`, `median_penalty`

The file holds one row per sample and is rewritten on each run with the same prefix.

### Junction completeness statistics (`_junction_completeness_stats.json`)

Written per sample with `--use_junction_completeness` (not in `--light` mode). It holds transcript counts from the completeness model:

- `complete`, `partial`, `none`, `single_exon`
- `mild_penalty`, `moderate_penalty`, `severe_penalty`
- `simple_complete`, `complex_analyzed`, `terminal_recovery_used`, `pure_median_1pct`
- `splice_competition_used` and `rms_used`, when they occur

---

## Confidence reports

These need `--output_confidence`. Metrics are computed for transcripts with a final count of at least `--confidence_min_expression` (default 1.0). See [Algorithm Details](algorithm_details.md#confidence-metrics) for the definitions.

### Transcript metrics (`_transcript_metrics_SPARSE.tsv.gz`)

One file per sample, with one row per transcript that has a non-zero count. This is the file `majec_build_db` loads. It has the columns described under [Transcript Confidence Report](#transcript-confidence-report-_counts_with_confidencetsvgz), plus:

| Column | Description |
|:--|:--|
| `ambiguous_fraction_distinguishability` | Distinguishability within the shared (ambiguous) part of the count only. It is 1.0 for transcripts with no shared reads. |
| `abs_inter_dist` | Holistic score (0–1) for distinguishability from **other genes**: `(1 − shared_read_fraction) + shared_read_fraction × ambiguous_fraction_abs_inter_dist`. Low = a near "dead heat" with another gene. |

### Transcript Confidence Report (`_counts_with_confidence.tsv.gz`)

One file per sample, written with `--output_confidence` and `--verbose_output`. It has one row per transcript in the annotation, sorted by count. Transcripts that are not expressed get placeholder values: `distinguishability_score` -1, and `competition_type` / `overall_confidence` set to `not_expressed`.

**Key columns**

| Column | Description | Interpretation |
|:--|:--|:--|
| **`count`** | Final count for the sample. | Primary expression estimate. |
| **`distinguishability_score`** | Holistic reliability (0–1): `(1 − shared_read_fraction) + shared_read_fraction × ambiguous_fraction_distinguishability`. | ~1.0: reliable. Below 0.5: ambiguous. -1: not computed. |
| **`strong_evidence_fraction`** | Fraction of the count from unambiguous evidence: unique fragments, unique multimapper classes and dominant shares of shared classes. | High = anchored by solid evidence. |
| **`shared_read_fraction`** | Fraction of the count from shared (ambiguous) classes. | High = heavy reliance on the EM. |
| **`inter_gene_competition_frac`** | Fraction of competing weight that comes from **other genes**. | High = warning flag for paralog or pseudogene contamination. |
| `intra_gene_competition_frac` | Fraction of competing weight from isoforms of the same gene. | High = isoform complexity rather than contamination. |
| **`hardest_competitor_gene`** | Gene (or TE subfamily) of the strongest competitor. | Pinpoints the locus causing ambiguity. |
| `competition_type` | Relation to the strongest competitor: `unique`, `intra-genic`, `inter-genic`, `gene-to-TE`, `TE-to-gene`, `intra-TE`, `inter-TE`. | |
| `has_unique_junction_support` | True if at least one observed splice junction belongs to this transcript alone in the annotation. | Strong independent evidence for the transcript. |
| `overall_confidence` | Category: `gold_standard`, `good`, `medium` or `low`. It combines `confidence_score`, distinguishability, unique junction support and `shared_read_fraction`. | Quick filter. |

**Assignment entropy**

| Column | Description |
|:--|:--|
| `entropy` | Shannon entropy (bits) of where the transcript's reads came from (classes). |
| `confidence_score` | `1 / (1 + entropy)`; 1.0 = all reads from one source. |
| `normalized_entropy` | Entropy divided by its maximum for the number of sources. |
| `n_sources` | Number of classes contributing reads. |
| `unique_fraction` | Fraction of the count from fragments unique to this transcript. |
| `unique_multimapper_fraction` | Fraction from multimapping fragments whose class contains only this transcript. |
| `total_count` | Count used for these metrics. |

**Competition**

| Column | Description |
|:--|:--|
| `n_competitive_classes` | Shared classes the transcript appears in. |
| `n_unique_competitors` | Distinct transcripts it shares classes with. |
| `min_distinguishability` | Weighted mean of the smallest allocation gap to any competitor. |
| `hardest_competitor` | Strongest competing transcript. In this file and in `_distinguishability_detailed.tsv.gz` it is an internal integer index; `_transcript_metrics_SPARSE.tsv.gz` gives the transcript ID. Use `hardest_competitor_gene` here. |
| `hardest_competitor_weight_frac` | Share of competing weight held by the hardest competitor. |
| `ambiguous_fraction_abs_inter_dist` | Distinguishability from other genes within the shared part of the count. |
| `dominant_reads`, `dominant_fraction` | Reads credited from shared classes in which this transcript takes more than half the allocation (scaled by how far above half), and that as a fraction of `count`. |

**Junctions and discord**

| Column | Description |
|:--|:--|
| `junction_evidence_fraction` | Junction reads relative to the count (can exceed 1). |
| `junction_confidence_category` | `unique_junction_validated`, `specific_shared_junctions`, `ambiguous_shared_junctions` or `no_junctions`. |
| `discord_score` | 0–4. How far the EM moved the count down from its prior. 0 = no reduction; higher = the EM took reads away from a transcript the priors supported. |

### High-confidence subset (`_high_confidence_counts.tsv.gz`)

One file per sample, with the same columns as `_transcript_metrics_SPARSE.tsv.gz`. Rows are the expressed transcripts with `confidence_score` > 0.8, `distinguishability_score` > 0.8, **or** unique junction support. If none qualify, the file falls back to transcripts with count > 10, and the log warns.

### Distinguishability details (`_distinguishability_detailed.tsv.gz`)

One file per sample, written with `--output_confidence` and `--verbose_output`. It has one row per expressed transcript with distinguishability data, sorted by count. Columns:

- `distinguishability_score`: here, the score within the shared part only, before the holistic combination;
- `min_distinguishability`, `ambiguous_fraction_abs_inter_dist`
- `n_competitive_classes`, `n_unique_competitors`
- `hardest_competitor`, `hardest_competitor_gene`, `strongest_external_competitor_gene`, `hardest_competitor_weight_frac`
- `intra_gene_competition_frac`, `inter_gene_competition_frac`, `competition_type`
- `my_gene` (the transcript's own gene or TE subfamily), `transcript_type` (`gene` or `TE`)
- `dominant_reads`, `count`

### Transcript Confidence Summary (`_transcript_confidence_summary.tsv.gz`)

Consolidates key confidence metrics from all samples into a single wide-format matrix, for cohort-level quality control and cross-sample comparison.

- **Rows:** transcripts expressed in at least one sample.
- **Columns:** four per sample:
  - `{sample}_count`;
  - `{sample}_confidence`: `confidence_score`;
  - `{sample}_distinguishability`: holistic `distinguishability_score`;
  - `{sample}_junction_support`: 1.0 = unique junction support, 0.5–0.9 = mostly specific shared junctions, 0.25 = ambiguous shared junctions, 0 = none.

### Group Confidence Report (`_group_confidence_comprehensive.tsv.gz`)

Metrics aggregated to genes and TE subfamilies, written with `--calculate_group_confidence`.

- **Rows:** one per sample and group (long format).
- **Expressed:** a transcript counts as expressed if its count is at least `--confidence_min_expression`.
- **Companion file:** `_group_confidence_summary.json` records the number of gene and TE groups and the parameters used.

| Column | Description | Interpretation |
|:--|:--|:--|
| `sample` | Sample name. | |
| **`group_id`** | Gene ID or TE subfamily. | |
| `group_type` | `gene` or `te`. | |
| **`aggregated_count`** | Sum of the group's transcript counts. | Total expression estimate. |
| **`holistic_group_distinguishability`** | Expression-weighted reliability for the whole group, from internal and external ambiguity. | ~1.0: easy to quantify. Below 0.5: high ambiguity. |
| **`inter_group_competition`** | Expression-weighted fraction of ambiguity coming from other groups. | Above 0.5: contamination problem (paralogs). Below 0.5: isoform complexity. |
| **`holistic_group_external_distinguishability`** | Score (0–1) combining total ambiguity with its external source. | Single actionable score for filtering or variance weighting. 1.0 = reliable; 0.0 = unreliable due to external conflict. |
| `intra_group_competition` | Expression-weighted fraction of ambiguity from within the group. | |
| `ambiguous_fraction_group_distinguishability`, `ambiguous_fraction_external_distinguishability` | The two distinguishability scores above, restricted to the shared part of the count. | |
| `main_external_competitor` | Most significant expression-weighted external competitor. | Pinpoints the gene causing inter-group ambiguity. |
| `n_expressed_transcripts` / `n_total_transcripts` | Expressed and annotated transcripts (or loci) in the group. | Isoform usage complexity. |
| `dominant_transcript`, `dominant_fraction` | Most highly expressed member and its share of the group. | Above 0.8: one member dominates. About 1/n: spread evenly. |
| `is_dominated` | `dominant_fraction > 0.5`. | |
| `effective_copies` | Effective number of expressed members, 2^(entropy of the expression shares). | |
| `gini_coefficient` | Inequality of expression across expressed members. | |
| `strong_evidence_fraction` | `weighted_unique_fraction + weighted_dominant_reads_fraction`. | Proportion anchored by unambiguous evidence. |
| `weighted_unique_fraction`, `weighted_dominant_reads_fraction`, `weighted_shared_fraction` | Expression-weighted transcript fractions. | |
| `weighted_confidence`, `high_conf_fraction` | Expression-weighted `confidence_score`, and the weighted share of members with `confidence_score` > 0.8. | |
| `has_junction_validation` | True if any expressed member has unique junction support. | |

### Competition analysis (`_competition_summary.tsv`, `_complex_competition_genes.tsv`)

Written with `--analyze_competition`.

- **`_competition_summary.tsv`** gives the mean, median, standard deviation and count of distinguishability for each `competition_type`, pooled across samples.
- **`_complex_competition_genes.tsv`** lists genes with more than 10 transcript-sample competition records. It has one column per competition type, sorted by intra-genic competition.

---

## Debugging: EM classes (`_em_classes.pkl`)

Written per sample with `--dump_em_classes`. It is a Python pickle of a dictionary, with transcript IDs decoded to strings:

| Key | Contents |
|:--|:--|
| `sample_name` | Sample name. |
| `unique_mapper_groups` | Unique EqCs: tuple of transcript IDs → fragment count. |
| `multimapper_classes` | Multimapping EqCs: frozenset of transcript IDs → fragment count. |
| `initial_unique_counts`, `initial_multimapper_counts` | Initial per-transcript counts. |
| `boosted_unique_counts` | Unique counts after the priors. |
| `junction_evidence`, `junction_boosted_transcripts` | Junction evidence per transcript, and the transcripts it boosted. |
| `global_expression_thresholds` | Momentum bin thresholds. |

```python
import pickle
with open("results/my_experiment_sample1_em_classes.pkl", "rb") as f:
    classes = pickle.load(f)
```

---

## Other files

**Annotation precompute.** `majec_precompute_annotations --output PREFIX` writes:

- `{PREFIX}_annotations.pkl.gz`: the annotation used by `--annotation`;
- `{PREFIX}_annotations_summary.json`: source GTF paths and checksums, creation date, transcript, gene, TE and junction counts, TSL distribution, and per-transcript junction statistics.

It no longer writes `_subset_coverage_features.bed`.

**Cache directory** (`--cache_dir`, default `pipeline_cache`):

- `{sample}_{key}.pkl` per sample: the EqCs, junction counts and measured fragment-length histogram from the BAM scan;
- `cache_metadata.json`: what each cache was built from.

Caches can be deleted at any time; they are rebuilt on the next `--use_cache` run. See [Usage Guide](usage_guide.md#caching).
