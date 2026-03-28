# MAJEC

<img width="2816" height="1536" alt="Gemini_Generated_Image_7e14am7e14am7e14" src="https://github.com/user-attachments/assets/c3a50e27-7bf0-45b5-9e23-8aa55d3551a3" />

**Momentum-Accelerated Junction-Enhanced Counting** — unified gene, isoform, and locus-level transposable element quantification from RNA-seq.

MAJEC jointly quantifies genes, transcript isoforms, and individual TE loci from standard BAM alignments in a single pass. By operating a probabilistically resolved Expectation-Maximization (EM) algorithm on a **joint gene+TE feature space**, MAJEC eliminates the systematic signal misattribution that plagues existing TE quantification tools.

## The Gene-TE Overlap Problem

Approximately 45% of the human genome consists of TEs, many of which are embedded directly within gene bodies.

* **TE-only tools:** Operate blind to genes. When a host gene is transcribed, its reads are falsely attributed to the overlapping TE, driving massive false-positive TE reactivation calls.

**The MAJEC Solution:** Genes and TEs compete for reads probabilistically. MAJEC uses empirical splice junction evidence to heavily penalize unsupported transcript isoforms, preventing them from stealing genuine TE reads, while correctly assigning spliced genic reads to their host genes.

## Key Features

- **Joint gene+TE feature space**: genes and TE loci compete for reads probabilistically, preventing systematic misattribution at overlapping loci
- **Junction-informed priors**: splice junction evidence from the BAM drives isoform-level accuracy and helps distinguish genic from TE-derived signal
- **Locus-level TE resolution**: individual TE insertions are quantified — not just subfamily aggregates
- **Confidence metrics**: per-transcript distinguishability scores, assignment entropy, and discord scores for transparent quality assessment
- **Fast**: momentum-accelerated EM typically converges in ~15 iterations; multiprocessing across samples

## Installation

### From source

```bash
mamba env create -f majec.yml
conda activate majec
pip install -e .
```

### Dependencies

Installed automatically via the conda environment:

- [Subread](http://subread.sourceforge.net/) (featureCounts) >= 2.0
- [samtools](http://www.htslib.org/) >= 1.20
- [bedtools](https://bedtools.readthedocs.io/) >= 2.31

## Quick Start

### 1. Precompute Annotations

Combine your gene and TE GTFs into a unified, mathematically optimized index.

```bash
majec_precompute_annotations \
    --gene_gtf gencode.v44.annotation.gtf \
    --te_gtf hg38_rmsk_TE.gtf \
    --output my_annotations
```

This produces `my_annotations_annotations.pkl.gz` (and optionally `_subset_coverage_features.bed` if `--generate_rescue_features` is used).

### 2. Run the Pipeline

Quantify all your samples jointly in a single command.

```bash
majec_run_pipeline \
    --annotation my_annotations_annotations.pkl.gz \
    --bams sample1.bam sample2.bam \
    --prefix my_experiment \
    --paired_end \
    --strandedness 2 \
    --use_subset_penalty \
    --use_junction_completeness \
    --library_type dT \
    --terminal_relax \
    --output_confidence \
    --use_cache
```

Input BAMs should be coordinate-sorted and produced by a splice-aware aligner such as STAR. For TE quantification, use `--outFilterMultimapNmax 100` (or similar) during alignment to retain multimapping reads.

## Outputs

MAJEC automatically aggregates locus-level estimates and provides DESeq2-ready matrices at multiple resolutions:

- `_total_EM_aggregated_counts.tsv` — Gene and subfamily-level counts
- `_total_EM_counts.tsv` — Isoform and locus-level counts
- `_transcript_metrics_SPARSE.tsv.gz` — Confidence scores and junction evidence

## How It Works

1. **Read assignment**: featureCounts assigns reads to features from the joint gene+TE annotation, forming equivalence classes of reads that map to the same set of transcripts.

2. **Junction extraction**: splice junctions reported by featureCounts (`.jcounts`) are matched to annotated transcript structures, providing isoform-discriminating evidence.

3. **Prior construction**: junction evidence, completeness scores, subset relationships, and (optionally) TSL annotations are combined into per-transcript priors that seed the EM.

4. **EM with momentum**: the Expectation-Maximization algorithm iteratively refines transcript abundance estimates. Momentum acceleration speeds convergence, typically reaching stable estimates within ~15 iterations.

5. **Output**: final gene-level, transcript-level, and (if applicable) TE locus-level count matrices, plus optional confidence metrics.

## Optional Workflow Tools

MAJEC includes additional tools for downstream analysis. See the full documentation for details.

- **`majec_build_db`** — Consolidate results from one or more pipeline runs into a single, queryable SQLite database. Supports merging chunked runs, attaching sample metadata, and serves as the input for all downstream tools.
- **`majec_prepare_deseq2`** — Generate ready-to-run DESeq2 analysis packages directly from the database. Supports flexible sample group definitions, confidence-weighted variance modeling, gene/transcript/differential-splicing analysis levels, and batch correction.
- **`majec_visualize`** — Generate interactive, multi-panel HTML reports for individual genes. Includes junction arc plots, differential splicing heatmaps, per-sample penalty diagnostics, and optional Excel export.

## Key Parameters

Run `majec_run_pipeline --help` or `majec_precompute_annotations --help` for the full list of options. The most important flags are shown in the Quick Start above. A few others worth noting:

| Flag | Description |
|------|-------------|
| `--use_subset_coverage_data` | Use read coverage to inform subset penalties (requires `--generate_rescue_features` during annotation precompute) |
| `--output_tpm` | Output TPM values in addition to counts |
| `--light` | Skip prior tracking and confidence metrics for faster, leaner runs |
| `--use_cache` | Cache featureCounts results to speed up reruns with different parameters |
| `--config` | Load settings from a JSON file (CLI flags override) |

## Documentation

- [BAM Preparation](docs/bam_preparation.md) — STAR alignment parameters for TE-aware quantification
- [Algorithm Details](docs/algorithm_details.md) — Two-phase EM, prior adjustment models, momentum acceleration, confidence metrics
- [Output Files](docs/output_files.md) — Column descriptions for all output tables and confidence reports
- [Post-Quantification Workflows](docs/post_quantification.md) — Database building, DESeq2 integration, and visualization
- [Usage Guide](docs/usage_guide.md) — Configuration templates, utility tools, resource requirements, and HPC deployment

## Citation

If you use MAJEC in your research, please cite:

> Lim, T.-Y. & Firestone, A.J. (2026) MAJEC: unified gene, isoform, and locus-level transposable element quantification from RNA-seq. *bioRxiv*.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
