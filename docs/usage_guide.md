# Usage Guide

## Gene vs. Isoform Quantification

MAJEC's most advanced features — Junction Completeness and Subset Isoform Penalties — are designed to resolve ambiguity at the **isoform level**. Their primary impact is on individual transcript quantification accuracy.

When counts are aggregated to the gene level, the effects of these penalties are often less pronounced. If your analysis is solely focused on differential **gene** expression, the benefits may be subtle. For any analysis involving isoforms or splicing, they are essential.

---

## Configuration Files

Most command-line options can be provided in a JSON configuration file via `--config`. CLI flags override file settings.

### Baseline Gene-Level Analysis

Fast configuration for gene-level counts without isoform-specific penalties:

```json
{
  "threads": 16,
  "prefix": "baseline_gene_run",
  "strandedness": 2,
  "paired_end": true,
  "use_subset_penalty": false,
  "use_junction_completeness": false,
  "use_tsl_penalty": false,
  "output_tpm": true,
  "output_confidence": false,
  "use_cache": true
}
```

### Publication-Quality Isoform Analysis

Recommended configuration with all evidence-based priors and confidence metrics enabled:

```json
{
  "threads": 24,
  "prefix": "publication_run",
  "strandedness": 2,
  "paired_end": true,
  "use_subset_penalty": true,
  "use_subset_coverage_data": true,
  "use_junction_completeness": true,
  "use_tsl_penalty": true,
  "library_type": "dT",
  "terminal_relax": true,
  "output_confidence": true,
  "calculate_group_confidence": true,
  "verbose_output": true,
  "use_cache": true,
  "cache_dir": "./majec_cache"
}
```

`library_type` must match your experimental protocol for the completeness model to work correctly. Options: `dT`, `polyA`, `random`, `none`.

---

## Utility Tools

### `majec_calc_frag_len` — Fragment Length Statistics

Calculates cDNA fragment length distributions by sampling intra-exonic read pairs from BAMs. Outputs per-sample JSON files used by `--frag_stats_dir` for effective length calculation and TPM.

```bash
# From a file listing BAM paths (one per line)
majec_calc_frag_len -o frag_stats/ bam_list.txt

# Or pipe paths directly
ls *.bam | majec_calc_frag_len -o frag_stats/
```

The tool samples 1M intra-exonic pairs per BAM by default (`-n`), skipping the first 10% to avoid positional bias in coordinate-sorted files. Use `-t` to process multiple BAMs in parallel.

### `majec_add_norm_factors` — Library Size Normalization

Calculates library size normalization factors and stores them in the MAJEC database. Required before running `majec_visualize`.

```bash
majec_add_norm_factors --db my_project.db
```

### `majec_calc_thresholds` — Cohort Expression Binning

Pre-calculates global expression thresholds from a cohort of BAMs for consistent momentum grouping across runs. Outputs a JSON file consumed by `--use_bins`.

```bash
majec_calc_thresholds \
    --bams *.bam \
    --gene_gtf gencode.v44.annotation.gtf \
    --te_gtf hg38_rmsk_TE.gtf \
    --output global_bins.json \
    --paired_end \
    --strandedness 2
```

---

## Resource Requirements

Memory and runtime depend primarily on annotation complexity and BAM count/size. The following are from actual benchmarking runs on an HPC cluster.

| Scenario | Annotation | Features | BAMs | Peak RSS | Wall Time |
|----------|-----------|----------|------|----------|-----------|
| Gene-only (LongBench) | GENCODE v44 | ~250K transcripts | 8 x ~4.5 GB | 33-42 GB | 7-21 min |
| Gene + TE | refGene + RMSK | 4.77M features | 6 x ~2.5 GB | 47.1 GB | 12.5 min |
| Cache build (gene-only) | GENCODE v44 | ~250K transcripts | 8 x ~4.5 GB | 34.6 GB | 35 min |

**Key observations:**

- Annotation complexity is the primary memory driver. Gene+TE annotations (millions of features) require significantly more memory than gene-only.
- Subsequent runs with caching (`--use_cache --read_only_cache`) skip featureCounts and are much faster.
- Annotation precompute (`majec_precompute_annotations`) takes ~3 min for gene+TE (6.2M GTF lines) and uses modest memory.

### Memory Recommendations

- **Gene-only**: 40-48 GB RAM
- **Gene + TE**: 64-96 GB RAM
- **Threads**: 1 thread per BAM for multi-BAM runs; 4-8 threads for a single BAM

---

## HPC Deployment

### SLURM Array Job

```bash
#!/bin/bash
#SBATCH --array=1-100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

SAMPLE_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
majec_run_pipeline \
    --bams /data/project/sample_${SAMPLE_ID}/*.bam \
    --threads 8 \
    --annotation /ref/annotations.pkl.gz \
    --prefix sample_${SAMPLE_ID} \
    --use_cache
```

- Each array task processes one sample independently.
- Use `--use_cache` on the first run, then `--read_only_cache` for parameter sweeps.
- After all jobs complete, merge with `majec_build_db --run_manifests sample_*_run_manifest.json`.
