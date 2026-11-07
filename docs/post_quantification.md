# Post-Quantification Workflows

After `majec_run_pipeline`, four tools take results from one or more runs through to differential expression and per-gene reports:

1. `majec_build_db` consolidates runs and sample metadata into one SQLite database.
2. `majec_add_norm_factors` adds library size factors to the database.
3. `majec_prepare_deseq2` writes a ready-to-run DESeq2 analysis package.
4. `majec_visualize` draws interactive per-gene splicing and quantification reports.

## Requirements

- **Python extras.**
  - `majec_build_db` needs polars, from the `database` extra.
  - `majec_visualize` needs the `viz` extra (plotly, python-kaleido, pypdf2, xlsxwriter).
  - `full_analysis` installs both. See [installation.md](installation.md).
- **sqlite3.** `majec_build_db` calls the `sqlite3` command-line program, so it must be on your `PATH`.
- **R.** Running the generated DESeq2 script needs the Bioconductor packages DESeq2, apeglm and genefilter:

  ```r
  BiocManager::install(c("DESeq2", "apeglm", "genefilter"))
  ```

- **Pipeline flags.** The database is built from files that `majec_run_pipeline` writes only with certain flags:

  | Flag | Why it is needed |
  |------|------------------|
  | `--use_cache` | The per-sample junction-coverage tables are read from the cache files listed in the run manifest. Without caching the database builds, but those tables are empty. |
  | `--output_confidence` | Needed for the per-sample confidence and prior-adjustment tables, which transcript-level confidence modes and `majec_visualize` use. The manifest lists these files only when this flag is set. |
  | `--calculate_group_confidence` | Writes the gene/TE-level confidence used by `majec_prepare_deseq2 --level gene` with `--confidence_mode append` or `variance`. |
  | `--output_tpm` | Optional. Without it, the count tables are loaded with their `tpm` column left empty (NULL). |

  Do not use `--light`: it turns off confidence metrics and prior tracking. The manifest records absolute paths to the annotation file, cache files and outputs, so keep them in place until the database is built. If you move them, edit the paths in the manifest.

**Counting unit.** For paired-end data, 0.2.0 counts each fragment once. 0.1.5 counted each mate of a uniquely mapped pair separately, so 0.2.0 unique counts are about half of 0.1.5's (single-end data is unaffected). `majec_build_db` does not check versions, so quantify every sample in one database with the same MAJEC version. See [`CHANGELOG.md`](../CHANGELOG.md).

---

## Step 1: Build the Analysis Database (`majec_build_db`)

Consolidate output files from one or more pipeline runs into a single, queryable SQLite database. This becomes the input for all downstream tools.

```bash
majec_build_db \
    --run_manifests chunk1_run_manifest.json chunk2_run_manifest.json \
    --metadata_file sample_metadata.tsv \
    --output_db my_project.db \
    --force
```

| Flag | Description |
|------|-------------|
| `--run_manifests` | One or more `_run_manifest.json` files from `majec_run_pipeline`. Required in practice. Manifests from chunked runs are merged. They must name the same annotation file, and each sample may appear in only one manifest. |
| `--output_db` | Path for the SQLite database file (required). |
| `--metadata_file` | Tab-separated sample metadata (recommended). Without it, the database holds only sample names, so groups can be selected only by `sample_id`. |
| `--force` | Delete and rebuild an existing database. Without it, an existing file is reused and tables that already contain data are skipped. To add runs, rebuild from all manifests with `--force`. |

### Metadata File Format

```tsv
#experimental_variables: treatment, cell_line
#id_variables: rep
sample_id	cell_line	treatment	rep	batch
MCF7_Control_R1	MCF7	Control	1	Day1
MCF7_Control_R2	MCF7	Control	2	Day2
MCF7_DrugA_R1	MCF7	DrugA	1	Day1
MCF7_DrugA_R2	MCF7	DrugA	2	Day2
```

- The file is tab-separated. Lines starting with `#` are not part of the table.
- The first column must be named `sample_id`. Its values must match the sample names MAJEC derived from the BAM file names (see [bam_preparation.md](bam_preparation.md#sample-names)).
  - Every sample in the runs must have a row, or `majec_build_db` stops.
  - Rows for samples that are not in the runs are ignored with a warning.
- The `#experimental_variables:` and `#id_variables:` lines record which columns are factors and which are identifiers (stored in the `metadata_schema` table).
  - Every variable they name must be a column of the table (`sample_id` included).
  - Group criteria in the downstream tools can use any metadata column.
- All values are stored as text, and matching is exact and case-sensitive.
- Include a batch column if applicable. It enables batch correction with `--batch_column` in `majec_prepare_deseq2`.

### What the Database Contains

| Table | Contents | Source |
|-------|----------|--------|
| `sample_metadata`, `metadata_schema` | Sample metadata and variable types | `--metadata_file` |
| `aggregated_counts_tpm` | Gene and TE subfamily counts and TPM (`group_id`, `count`, `tpm`) | `_total_EM_aggregated_counts.tsv`, `_total_EM_aggregated_TPM.tsv` |
| `final_counts_tpm` | Transcript and TE locus counts and TPM | `_total_EM_counts.tsv`, `_total_EM_TPM.tsv` |
| `counts_with_confidence` | Per-sample transcript confidence metrics | `_{sample}_transcript_metrics_SPARSE.tsv.gz` |
| `priors` | Per-sample prior adjustments (junction boost, TSL, completeness and subset penalties) | `_{sample}_prior_adjustments.tsv.gz` |
| `group_confidence` | Per-sample gene/TE-level confidence metrics | `_group_confidence_comprehensive.tsv.gz` |
| `junction_counts`, `transcript_junction_summary` | Per-sample annotated-junction counts, including multimapping alignments that span annotated junctions | Cache files |
| `transcript_annotations`, `transcript_junctions`, `junction_annotations`, `subset_supersets`, `transcript_id_map`, `junction_id_map` | Annotation context | Annotation file |
| `sample_normalization` | Size factors | Added by `majec_add_norm_factors` |

The database is plain SQLite, so you can query it from R, Python or the command line:

```bash
sqlite3 -header -separator $'\t' my_project.db \
    "SELECT sm.sample_id, a.group_id, a.count
     FROM aggregated_counts_tpm a JOIN sample_metadata sm USING (sample_id_int)
     WHERE a.group_id = 'MYH9';"
```

The territory tables of 0.1.5 (`territory_regions`, `transcript_territories`, `territory_coverage`) are no longer created. Subset/superset discrimination is now handled when equivalence classes (EqCs) are built (see [algorithm_details.md](algorithm_details.md)).

---

## Step 2: Add Normalization Factors (`majec_add_norm_factors`)

Calculate median-of-ratios size factors from the gene and TE subfamily counts, and store them in a `sample_normalization` table. `majec_visualize` requires this table, so run this step after every database build.

```bash
majec_add_norm_factors --db my_project.db
```

`majec_prepare_deseq2` does not use these factors: DESeq2 estimates its own.

---

## Step 3: DESeq2 Analysis (`majec_prepare_deseq2`)

Generate a ready-to-run DESeq2 analysis package from the database.

```bash
majec_prepare_deseq2 \
    --db my_project.db \
    --comparisons_file comparisons.txt \
    --output_dir deseq2_analysis \
    --level gene \
    --confidence_mode append

cd deseq2_analysis && Rscript gene_run_deseq2.R
```

### Defining Comparisons

Each line in the comparisons file defines one contrast:

```
# Format: Name; Case_Criteria; Control_Criteria
MCF7_DrugA_vs_Ctrl; cell_line=MCF7;treatment=DrugA; cell_line=MCF7;treatment=Control
All_Drugs_vs_Ctrl; treatment=DrugA,DrugB; treatment=Control
Specific_Samples; sample_id=Sample_A01,Sample_A04; sample_id=Sample_A02,Sample_A03
```

- Separate the three fields with a semicolon followed by a space (`; `).
- Within a group, join criteria with a semicolon and **no** space (`cell_line=MCF7;treatment=DrugA`). A sample must match every criterion.
- Separate several values for one key with commas (`treatment=DrugA,DrugB`).
- Keys are metadata column names. `sample_id` selects samples directly.
- Lines starting with `#` are ignored. Lines that do not split into three fields are skipped without an error, so check the group sizes the tool prints.
- Comparison names are made R-safe: characters other than letters, digits, `.` and `_` become `_`, and a leading digit gets a `G_` prefix.

### Cohort

`--cohort_str` (a criteria string such as `"cell_line=MCF7"`) or `--cohort_file` (criteria, one per line) adds samples beyond those named in the comparisons.

Cohort samples are written to the count matrix and coldata. They also count toward the expression pre-filters: `--min_cpm`/`--min_samples`, and for `delta_psi` the junction and gene filters. Each comparison's DESeq2 model, however, is fit on that comparison's case and control samples only, because the generated script drops all other samples before calling DESeq2. If you give no cohort, the package contains just the samples in your comparisons.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | (required) | The MAJEC database. |
| `--comparisons_file` | (required) | Contrast definitions (above). |
| `--output_dir` | (required) | Directory for the analysis package. |
| `--output_prefix` | the level | Prefix for all output files. |
| `--level` | `gene` | `gene`, `transcript`, `junction` or `delta_psi` (below). |
| `--confidence_mode` | `append` | `none`, `append` or `variance` (below). Forced to `none` for `junction` and `delta_psi`. |
| `--variance_base` | 5 | Maximum dispersion multiplier in `variance` mode. |
| `--min_cpm`, `--min_samples` | 1.0, 3 | Keep features with at least this CPM in at least this many samples (all levels except `delta_psi`). |
| `--junction_min_average_count`, `--junction_min_samples` | 10, 3 | `delta_psi`: keep junctions averaging at least this many reads per sample and seen in at least this many samples. |
| `--min_gene_cpm`, `--min_gene_samples` | 2, 3 | `delta_psi`: consider only junctions of genes with at least this CPM in at least this many samples. |
| `--batch_column` | none | Metadata column added to the design (`~ batch + comparison`). |

### Analysis Levels

| `--level` | Features |
|-----------|----------|
| `gene` | Genes and TE subfamilies (`name:family:class`, for example `L1HS:L1:LINE`), from the aggregated counts. |
| `transcript` | Isoforms and individual TE loci. |
| `junction` | Annotated-junction read counts. |
| `delta_psi` | Differential splicing. Each feature is one junction within one gene (`gene_id:chr:start-end:strand`). Its value is the junction's share of the gene's junction reads, `(junction reads + 1) / (gene junction reads + 10) × 10,000`, rounded so that DESeq2 can model it. A significant change means the junction is used more or less relative to the gene's other junctions. A junction shared by several genes appears once per gene, and `{prefix}_junction_metadata.tsv` links each feature back to its genes. |

### Confidence Modes

| `--confidence_mode` | Description |
|---------------------|-------------|
| `none` | Standard DESeq2 on the counts. |
| `append` | Standard DESeq2, then annotate the results with MAJEC confidence metrics and a `MAJEC_flags` column. Recommended for exploration and filtering. |
| `variance` | Bring confidence into the model. Each feature's DESeq2 dispersion is multiplied by `variance_base^(1 − confidence)`, averaged over the comparison's samples in which the feature is expressed. The multiplier runs from 1 (fully confident) to `--variance_base` (confidence 0), which down-weights ambiguous features in the test. Confidence is `holistic_group_external_distinguishability` at gene level and `final_confidence` (below) at transcript level. The results carry the same metric and flag columns as `append`. |

### Output

The package directory contains (with the default prefix, which is the level name):

- `gene_counts_matrix.tsv`: the pre-filtered count matrix.
- `gene_coldata.tsv`: sample metadata, plus one `comp_<name>` column per comparison holding `case`, `control` or `other`.
- `gene_run_deseq2.R`: the R script. Run it from inside the package directory.
- `gene_<metric>_matrix.tsv`: one matrix per confidence metric, read by the script (`append` and `variance`).
- `gene_variance_matrix.tsv`: per-sample dispersion multipliers (`variance`).
- `delta_psi_junction_metadata.tsv`: junction-to-gene links (`delta_psi` only).

The script writes one `<prefix>_<comparison>_DESeq2_results.csv` per comparison, sorted by p-value. Each file has `feature_id`, the apeglm-shrunken DESeq2 columns (`baseMean`, `log2FoldChange`, `lfcSE`, `pvalue`, `padj`) and, in `append` and `variance` modes, the MAJEC columns below.

The pipeline's counts are fractional EM estimates. `majec_prepare_deseq2` rounds them to the nearest integer for DESeq2.

### Gene-Level Confidence Columns and Flags

Each metric is the mean over the comparison's case and control samples.

| Column | Meaning |
|--------|---------|
| `MAJEC_strong_evidence_frac` | Share of the gene's count anchored by unambiguous evidence (`strong_evidence_fraction`). |
| `MAJEC_inter_gene_competition` | The gene's `holistic_group_external_distinguishability`. Despite the column name, **higher is better**. 1 means the gene's count is not confounded with other genes. Values near 0 mean most of its count comes from reads it shares with other genes (paralogs, pseudogenes, TE copies) that MAJEC could not separate. |
| `MAJEC_gini` | Gini coefficient of isoform expression within the gene: 0 means evenly spread, 1 means a single isoform. |
| `MAJEC_flags` | Semicolon-separated flags, or `PASS` if none apply. |

| Flag | Condition | Meaning |
|------|-----------|---------|
| `LOW_EVIDENCE` | `padj` < 0.05 and `MAJEC_strong_evidence_frac` < 0.3 | A significant result resting mostly on ambiguous reads. |
| `HIGH_PARALOG_AMBIGUITY` | `padj` < 0.05 and `MAJEC_inter_gene_competition` < 0.3 | A significant result for a gene whose count is largely shared with other genes. Some of the change may belong to a paralog or pseudogene. |
| `DOMINATED_BY_ONE_ISOFORM` | `padj` < 0.05 and `MAJEC_gini` > 0.90 | One isoform carries the gene. Check the transcript-level results. |
| `AMBIGUOUS_NULL` | `padj` ≥ 0.05 or `NA`, `MAJEC_inter_gene_competition` < 0.3, and `baseMean` > 50 | A well-expressed gene with no significant change, whose count is largely shared with other genes. Here the lack of significance can reflect reads MAJEC could not assign between the gene and its competitors, rather than an absence of change. Read it as "not determined", not "unchanged". |

- The first three flags apply only to significant results, and `AMBIGUOUS_NULL` only to non-significant ones, so `AMBIGUOUS_NULL` never appears together with the others.
- `PASS` means that no flag condition was met. It does not mean that the result is significant.
- `AMBIGUOUS_NULL` is gated on the inter-gene score only. Many genes have a low strong-evidence fraction for structural reasons, which says little about whether a null result can be trusted.

### Transcript-Level Confidence Columns and Flags

| Column | Meaning |
|--------|---------|
| `MAJEC_final_confidence` | `strong_evidence_fraction + shared_read_fraction × ambiguous_fraction_distinguishability`, capped at 1. Unambiguous evidence counts in full; the ambiguous part counts in proportion to how well it was resolved. |
| `MAJEC_strong_evidence_frac` | Share of the count from unambiguous evidence. |
| `MAJEC_distinguish_score` | `distinguishability_score`, the transcript's overall reliability (0–1). |
| `MAJEC_discord_score` | How strongly the EM reduced the transcript's count below its pre-EM count, scaled by expression (0–4). |
| `MAJEC_completeness_penalty_case_mean`, `_control_mean`, `MAJEC_completeness_pval` | Mean junction-completeness penalty per group on a −log10 scale (0 = no penalty), and a t-test between groups. |
| `MAJEC_subset_penalty_case_mean`, `_control_mean`, `MAJEC_subset_pval` | The same for the subset penalty. |

Every transcript-level flag requires `padj` < 0.05:

| Flag | Condition |
|------|-----------|
| `LOW_FINAL_CONFIDENCE` | `MAJEC_final_confidence` < 0.3 |
| `LOW_POSITIVE_EVIDENCE` | `MAJEC_strong_evidence_frac` < 0.3 |
| `POOR_DISTINGUISHABILITY` | `MAJEC_distinguish_score` < 0.3 |
| `HIGH_EM_DISCORDANCE` | `MAJEC_discord_score` > 1.5 |
| `COMPLETENESS_PENALTY_CHANGED` | `MAJEC_completeness_pval` < 0.05: the completeness penalty differs between groups, so part of the fold change may come from the prior. |
| `SUBSET_PENALTY_CHANGED` | `MAJEC_subset_pval` < 0.05: the same for the subset penalty. |

`AMBIGUOUS_NULL` is gene-level only. For the underlying metrics, see [output_files.md](output_files.md) and [algorithm_details.md](algorithm_details.md).

### Transposable Elements and TPM

- `--level gene` tests TE subfamilies alongside genes. `--level transcript` tests individual TE loci alongside isoforms.
- To place TE loci in the genome, run the pipeline with `--te_coordinates`. Then join `{prefix}_TE_locus_coordinates.tsv` to the results on `locus_id` = `feature_id`. `majec_build_db` does not load that file.
- Use counts, not TPM, for TE loci. TE loci are genomic intervals, and very short loci get inflated TPM. DESeq2 works on counts. If you rank or filter loci by expression outside DESeq2, use counts or DESeq2-normalized counts too.
- TPM caveat for genes: on 3′-biased polyA libraries, long transcripts receive fewer reads than their length implies, so any length correction (MAJEC's or another tool's) over-corrects them in TPM.

---

## Step 4: Visualization (`majec_visualize`)

Generate interactive, multi-panel HTML reports for one gene, comparing two groups of samples. Run `majec_add_norm_factors` first.

```bash
majec_visualize \
    --db my_project.db \
    --gene MYH9 \
    --group_A_str "cell_line=MCF7;treatment=DrugA" \
    --group_B_str "cell_line=MCF7;treatment=Control" \
    --output_dir myh9_report \
    --export_excel \
    --save_svg
```

| Flag | Description |
|------|-------------|
| `--db` | The MAJEC database (required). |
| `--gene` | A gene ID as it appears in the annotation GTF's `gene_id` attribute (for example `MYH9` for a RefSeq-style GTF, or an Ensembl ID for GENCODE), or a transcript ID, which selects its gene (required). |
| `--output_dir` | Output directory (required). |
| `--group_A_str` / `--group_A_file` | Group A, as a criteria string or a file. One of the two is required. |
| `--group_B_str` / `--group_B_file` | Group B, as above. |
| `--delta_psi_bootstrap` | Bootstrap iterations for ΔPSI confidence intervals (default 1000). |
| `--save_svg` | Also save SVG files. Needs kaleido. |
| `--save_pdf` | Also save PDF files. Needs kaleido, plus PyPDF2 for the combined multi-page PDFs. |
| `--export_excel` | Write a companion spreadsheet with the raw and aggregated data. Needs xlsxwriter. |
| `--include_related_genes` | Also run the report for every gene that shares a junction with the query gene. |

Criteria use the same `key=value` syntax as the comparisons file. A group file should hold the criteria on one line, separated by semicolons.

### Report Contents

Files are named after the resolved gene ID:

- `<gene>_aggregate_junction_arc_plot.html`: all isoforms, with arc thickness showing junction usage in each group.
- `<gene>_aggregate_junction_heatmap.html`: ΔPSI (log2, B vs A) for every junction across every isoform.
- `<gene>_all_samples_penalty_heatmaps.html`: one panel per sample showing each isoform's TSL, subset and completeness penalties, initial and final counts, and junction coverage. Use the left and right arrow keys to step between samples.
- `<gene>_all_subsets_junction_plots.html`: written only when the gene has subset/superset isoform pairs. For each subset isoform it shows the junction evidence and penalties MAJEC used to weigh it against its supersets.
- `<gene>_analysis_data.xlsx`: with `--export_excel`.
- SVG files (`.svg`, or a `.zip` of SVGs for the multi-panel reports) with `--save_svg`, and PDF files with `--save_pdf`.
- With `--include_related_genes`: the same files for each related gene, a `<gene>_related_genes_summary.json` listing the shared junctions, and `<related gene>_error.txt` for any gene that failed.

The multi-panel reports load their panels from a sibling `<report name>_figures/` directory, so move the two together.

The territory coverage panels of 0.1.5 are no longer drawn.

---

## Related Helper Tools

Two helper tools run on BAMs before quantification rather than after. See [usage_guide.md](usage_guide.md) for full usage.

- **`majec_calc_thresholds`** computes cohort-wide expression thresholds (25th and 75th percentiles of expressed counts) and writes a JSON file for `majec_run_pipeline --use_bins`. This gives consistent momentum grouping across separately run chunks.
  - It runs featureCounts, so it needs Subread ≥ 2.0.2. Paired-end data is counted as fragments (`--countReadPairs`), the same unit `majec_run_pipeline` uses.
  - Its `--strandedness` defaults to `2`; set it to the value you use for the pipeline.
- **`majec_calc_frag_len`** writes per-BAM fragment-length statistics (`<BAM file name without .bam>_cDNA_frag_stats.json`) for the optional `majec_run_pipeline --frag_stats_dir` override. It needs samtools.
  - On paired-end data, 0.2.0 measures the fragment-length distribution during the BAM scan and uses it for TPM by default, so this tool is optional.
  - `majec_calc_frag_len` estimates from genomic template length (TLEN), while the in-scan measurement works in transcript coordinates, so the two can differ on spliced libraries. When `--frag_stats_dir` is given, it takes precedence, and the pipeline logs a warning.
  - The statistics files are named after the BAM file name, not the sample name. BAMs that share a file name, such as STAR's default `Aligned.sortedByCoord.out.bam` in per-sample directories, would share one statistics file.
