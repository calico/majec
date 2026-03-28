# Output Files

## Main Outputs

- `{prefix}_total_EM_counts.tsv` — Final transcript/locus-level expression estimates
- `{prefix}_total_EM_aggregated_counts.tsv` — Gene and TE subfamily aggregated counts

---

## Priors & Penalties Report (`_prior_adjustments.tsv.gz`)

A detailed log of every adjustment made to a transcript's initial count before the EM algorithm. Provides complete transparency into the evidence-based priors system, showing the effect of each penalty and boost.

| Column | Description | Interpretation |
|:-------|:------------|:---------------|
| **`initial_count`** | Raw fractional count before any adjustments. | Baseline expression evidence from read counting. |
| `raw_junction_evidence` | Raw weighted evidence score from splice junctions before applying `--junction_weight`. | Pure data-driven evidence from observed splice junctions, weighted by uniqueness. |
| **`junction_boost`** | Final additive count boost from all splice junction evidence. | Positive value = transcript prior was increased due to junction support. |
| **`tsl_penalty`** | Multiplicative penalty factor (0–1) from Transcript Support Level. | < 1.0 = down-weighted due to poor annotation support. |
| **`completeness_penalty`** | Multiplicative penalty factor (0–1) from the Junction Completeness model. | < 1.0 = penalized for missing or under-represented splice junctions. |
| `original_subset_penalty` | Initial annotation-based penalty factor (0–1) for subset isoforms. | Based on junction evidence of the subset vs. its superset partners. |
| **`adjusted_subset_penalty`** | Final combined multiplicative penalty factor (0–1) from the Subset Penalty model. | After modulation by data-driven `territory_evidence_ratio`. |
| **`final_count`** | Final adjusted count after all boosts and penalties. | The prior value (θ) used to start the EM algorithm. |
| `n_observed_junctions` / `n_expected_junctions` | Splice junctions observed vs. expected from annotation. | Raw data for the `completeness_penalty`. |
| `completeness_model` | Statistical model used for completeness penalty (e.g., 'median', 'terminal_recovery'). | Context for how the completeness penalty was derived. |
| `territory_evidence_ratio` | Ratio of read density in unique exonic regions vs. comparator regions. | Data-driven component of subset penalty. > 1.0 = evidence for the subset, potentially rescuing it from penalty. |
| `total_prior_adjustment` | Total fold-change from `initial_count` to `final_count`. | > 1.0: boosted overall. < 1.0: penalized overall. |
| `penalty_types` | Semicolon-separated list of applied penalties (e.g., `tsl;completeness`). | Quick summary of why a transcript's count was adjusted. |

---

## Transcript Confidence Report (`_counts_with_confidence.tsv.gz`)

Final quantification for each transcript with detailed reliability metrics. Generated when `--output_confidence` is enabled.

| Column | Description | Interpretation |
|:-------|:------------|:---------------|
| **`count`** | Final fractional count after all priors and EM steps. | Primary expression estimate. |
| **`distinguishability_score`** | Holistic reliability score (0–1) combining all evidence. | ~1.0: reliable. < 0.5: ambiguous. |
| **`strong_evidence_fraction`** | Fraction of counts from unambiguous evidence (unique + dominant reads). | Foundation of reliability — high value = anchored by solid evidence. |
| **`shared_read_fraction`** | Fraction of count from ambiguous, shared read groups. | High value = heavy reliance on EM for quantification. |
| **`inter_gene_competition_frac`** | Fraction of the ambiguous portion from transcripts of **other genes**. | High value = warning flag for paralog/pseudogene contamination. |
| **`abs_inter_dist`** | Score (0–1) measuring distinguishability against external competitors. | Low score = "dead heat" with another gene. |
| **`hardest_competitor_gene`** | Gene ID of the most significant competitor. | Pinpoints the gene causing ambiguity. |
| `has_unique_junction_support` | Boolean: supported by at least one unique splice junction. | TRUE = strong independent evidence for the transcript's existence. |
| `ambiguous_fraction_distinguishability` | Distinguishability score for only the shared read fraction. | Diagnostic for understanding ambiguous reads in isolation. |

---

## Group Confidence Report (`_group_confidence_comprehensive.tsv.gz`)

Aggregated metrics at the gene or TE family level. Generated when `--calculate_group_confidence` is enabled.

| Column | Description | Interpretation |
|:-------|:------------|:---------------|
| **`group_id`** | Gene or TE family name. | |
| **`aggregated_count`** | Sum of counts across all transcripts in the group. | Total expression estimate. |
| **`holistic_group_distinguishability`** | Expression-weighted reliability score for the entire gene. | ~1.0: easy to quantify. < 0.5: high ambiguity. |
| **`inter_group_competition`** | Expression-weighted fraction of ambiguity from other genes. | > 0.5: "contamination problem" (paralogs). < 0.5: "complex splice" (isoform complexity). |
| **`holistic_group_external_distinguishability`** | Final score (0–1) combining total ambiguity and its external source. | Single actionable score for filtering or variance weighting. 1.0 = reliable. 0.0 = unreliable due to external conflict. |
| `n_expressed_transcripts` | Transcripts in the group with count > 1.0. | Isoform usage complexity. |
| `dominant_transcript` | Most highly expressed transcript in the group. | Major isoform or TE copy. |
| `dominant_fraction` | Dominant transcript as a fraction of total group expression. | > 0.8: one transcript dominates. ~1/n: spread evenly. |
| `main_external_competitor` | Most significant expression-weighted external competitor. | Pinpoints the gene causing inter-group ambiguity. |
| `strong_evidence_fraction` | Expression-weighted average of strong evidence across expressed transcripts. | Proportion of quantification anchored by unambiguous evidence. |

---

## Transcript Confidence Summary (`_transcript_confidence_summary.tsv.gz`)

Consolidates key confidence metrics from all samples into a single wide-format matrix for cohort-level quality control and cross-sample comparison.

**Structure:** Transcripts as rows, columns named `SampleName_MetricName`. Provides cohort-wide context for assessing consistency and dynamics of quantification reliability across your run.
