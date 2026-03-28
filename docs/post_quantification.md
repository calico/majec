# Post-Quantification Workflows

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
| `--run_manifests` | One or more `_run_manifest.json` files from `majec_run_pipeline`. Automatically merged and validated. |
| `--metadata_file` | Tab-separated sample metadata (recommended). Enables downstream statistical comparisons. |
| `--output_db` | Path for the SQLite database file. |
| `--force` | Overwrite an existing database. |

### Metadata File Format

```tsv
#experimental_variables: treatment, cell_line
#id_variables: sample_id, rep
sample_id	cell_line	treatment	rep	batch
MCF7_Control_R1	MCF7	Control	1	Day1
MCF7_DrugA_R1	MCF7	DrugA	1	Day1
```

- Header lines starting with `#` define variable types for downstream tools.
- `experimental_variables`: factors for comparison (e.g., in DESeq2).
- `id_variables`: sample identifiers and replicates.
- `sample_id` must match sample names from the pipeline run.
- Include batch columns if applicable — enables batch correction via `--batch_column` in `majec_prepare_deseq2`.

---

## Step 2: DESeq2 Analysis (`majec_prepare_deseq2`)

Generates a ready-to-run DESeq2 analysis package from the database.

```bash
majec_prepare_deseq2 \
    --db my_project.db \
    --comparisons_file comparisons.txt \
    --output_dir deseq2_analysis \
    --level gene \
    --confidence_mode variance
```

### Defining Comparisons

Each line in the comparisons file defines one contrast:

```
# Format: Name; Case_Criteria; Control_Criteria
MCF7_DrugA_vs_Ctrl; cell_line=MCF7;treatment=DrugA; cell_line=MCF7;treatment=Control
All_Drugs_vs_Ctrl; treatment=DrugA,DrugB; treatment=Control
```

Criteria use `key=value` syntax. Multiple values for one key are comma-separated; multiple keys are semicolon-separated. You can also select specific samples with `sample_id=Sample_A01,Sample_A04`.

### Cohort

The `--cohort_str` or `--cohort_file` flag defines the superset of samples included in the DESeq2 model. All samples in your comparisons are automatically included; the cohort flag adds extra samples to serve as background for variance estimation. If unspecified, the cohort is just the samples in your comparisons.

### Analysis Levels

| `--level` | Description |
|-----------|-------------|
| `gene` | Standard gene-level differential expression. |
| `transcript` | Transcript-level differential expression. |
| `junction` | Junction-level counts. |
| `delta_psi` | Differential splicing — generates a normalized junction usage matrix and tests for changes in relative splice site utilization between conditions. |

### Confidence Modes

| `--confidence_mode` | Description |
|---------------------|-------------|
| `none` | Standard DESeq2 on raw counts. |
| `append` | Standard DESeq2, then annotate results with MAJEC confidence metrics (distinguishability, discord, evidence fractions). Recommended for exploration and filtering. |
| `variance` | Incorporate reliability scores directly into the DESeq2 model by inflating variance for low-confidence features. Down-weights ambiguous genes/transcripts during statistical testing. |

### Output

The command creates a directory containing:

- `_counts_matrix.tsv` — count matrix
- `_coldata.tsv` — sample metadata for DESeq2
- `_variance_matrix.tsv` — variance inflation factors (if `--confidence_mode variance`)
- `_run_deseq2.R` — ready-to-run R script

Run with `Rscript _run_deseq2.R`.

---

## Step 3: Visualization (`majec_visualize`)

Generate interactive, multi-panel HTML reports for individual genes.

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

### Report Contents

- **Junction arc plot**: all isoforms with arc thickness representing junction usage per condition.
- **Differential junction heatmap**: delta PSI for every junction across every isoform.
- **Subset analysis plots**: junction and exonic territory evidence used to resolve subset/superset ambiguity.
- **Per-sample penalty heatmaps**: raw counts, applied penalties, and final corrected counts per isoform per sample. Combined into a single HTML with keyboard navigation.
- **Excel export** (`--export_excel`): companion spreadsheet with all raw and aggregated data.
