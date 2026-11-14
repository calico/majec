# MAJEC

<img width="2816" height="1536" alt="Gemini_Generated_Image_7e14am7e14am7e14" src="https://github.com/user-attachments/assets/c3a50e27-7bf0-45b5-9e23-8aa55d3551a3" />

**Momentum-Accelerated Junction-Enhanced Counting**: unified gene, isoform, and locus-level transposable element quantification from RNA-seq.

MAJEC quantifies genes, transcript isoforms, and individual transposable element (TE) loci together from standard BAM alignments. It builds equivalence classes in a single pass over each BAM, then runs an Expectation-Maximization (EM) algorithm on a **joint gene+TE feature space**. Because genes and TEs are modelled together, reads from a transcribed host gene are not misattributed to the TEs embedded in it.

## The Gene–TE Overlap Problem

About 45% of the human genome consists of TEs, and many of them sit inside gene bodies.

* **TE-only tools** do not model genes. When a host gene is transcribed, its reads can be attributed to the overlapping TE, producing false-positive TE reactivation calls.

**The MAJEC approach:** genes and TEs compete for reads probabilistically. Splice-junction evidence from the BAM restricts and penalises transcript isoforms the data do not support, so they cannot absorb genuine TE reads, while spliced genic reads are assigned to their host genes.

## Key Features

- **Joint gene+TE feature space**: genes and TE loci compete for reads probabilistically, which prevents systematic misattribution at overlapping loci.
- **Junction-aware equivalence classes and priors**: annotated splice junctions and exon containment narrow each fragment's candidate transcripts while the BAM is read, and junction evidence also informs the per-transcript priors.
- **Locus-level TE resolution**: individual TE insertions are quantified, as well as subfamily aggregates (`name:family:class` IDs).
- **Confidence metrics**: per-transcript distinguishability scores, assignment entropy, and discord scores for transparent quality assessment.
- **Fast**: a single BAM pass per sample with the compiled builder (majec-core); the momentum EM typically finishes in about 15 iterations; samples are processed in parallel.

## What's new in 0.2.0

0.2.0 builds on 0.1.5. See [CHANGELOG.md](https://github.com/calico/majec/blob/master/CHANGELOG.md) for the full list and migration notes, and the [0.2.0 white paper](https://github.com/calico/majec/blob/master/docs/whitepaper.md) for a benchmark comparison of the two versions.

- **Single-pass, junction-aware equivalence classes (EqCs).** MAJEC reads the BAM directly. Each fragment's candidate transcripts come from exon overlap, annotated-junction compatibility, and exon containment, and the two mates' candidates are intersected. This replaces the featureCounts-based read assignment.
- **Two interchangeable builders.** The compiled builder (majec-core, `pip install "majec[rust]"`) and the pure-Python builder (pysam, always installed) give the same results; the pure-Python one is roughly 6× slower on the BAM scan. The log and the run manifest record which one ran.
- **Paired-end data is counted per fragment.** Each fragment is counted once. Unique counts on paired-end data are therefore about half of 0.1.5's, so do not mix counts from the two versions in one analysis. Single-end data is unaffected.
- **TPM effective lengths from the measured fragment-length distribution.** On paired-end data, fragment lengths are measured during the BAM scan, so no separate prepass is needed. `--frag_stats_dir` is now an optional override.
- **Fewer external dependencies.** `majec_run_pipeline` no longer calls featureCounts, samtools, or bedtools, and annotation files no longer need the source GTFs at their original paths.
- **Experimental support for aligners without NH tags** (for example minimap2 long-read BAMs) through `--derive_multiplicity`. Accuracy for long reads has not been established.
- **Removed options.** Coverage-based subset rescue and paired-junction rescue are no longer part of the pipeline; their roles are now covered by exon containment, junction compatibility and mate intersection during EqC construction. Re-run `majec_precompute_annotations` after upgrading.

## Installation
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/majec/README.html)

MAJEC needs Python 3.11 or newer. See [docs/installation.md](https://github.com/calico/majec/blob/master/docs/installation.md) for details, extras, and building majec-core from source.

### pip

```bash
pip install "majec[rust]"    # with the compiled builder (majec-core)
# or
pip install majec            # pure-Python builder only: same results, slower BAM scan
```

### From source

```bash
git clone https://github.com/calico/majec.git
cd majec
mamba env create -f majec.yml
conda activate majec
pip install -e ".[rust]"
```

### bioconda

```bash
conda install bioconda::majec
pip install majec-core       # optional: adds the compiled builder
```

### External tools

`majec_run_pipeline` reads BAMs itself and needs no external programs. Two optional helper tools do: `majec_calc_frag_len` uses [samtools](http://www.htslib.org/), and `majec_calc_thresholds` uses featureCounts from [Subread](http://subread.sourceforge.net/). Both are included in `majec.yml` and the bioconda package.

## Quick Start

### 1. Precompute Annotations

Combine your gene and TE GTFs into a single precomputed annotation index.

```bash
majec_precompute_annotations \
    --gene_gtf gencode.v44.annotation.gtf \
    --te_gtf hg38_rmsk_TE.gtf \
    --output my_annotations
```

This produces `my_annotations_annotations.pkl.gz`. The file is self-contained: the GTFs do not need to stay at their original paths. Build it once per annotation, with the same MAJEC version you quantify with.

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

Input BAMs should be coordinate-sorted and produced by a splice-aware aligner such as STAR. For TE quantification, use `--outFilterMultimapNmax 100` (or similar) during alignment to retain multimapping reads. See [docs/bam_preparation.md](https://github.com/calico/majec/blob/master/docs/bam_preparation.md). For single-end data, omit `--paired_end`.

## Outputs

MAJEC aggregates locus-level estimates and writes DESeq2-ready matrices at several resolutions:

- `{prefix}_total_EM_aggregated_counts.tsv`: gene and TE subfamily counts. TE subfamilies use `name:family:class` IDs (for example `L1HS:L1:LINE`).
- `{prefix}_total_EM_counts.tsv`: isoform and TE locus counts.
- `{prefix}_{sample}_transcript_metrics_SPARSE.tsv.gz`: per-sample confidence scores and junction evidence (with `--output_confidence`).
- `{prefix}_total_EM_TPM.tsv` and `{prefix}_total_EM_aggregated_TPM.tsv`: TPM values (with `--output_tpm`).
- `{prefix}_TE_locus_coordinates.tsv`: chromosome, start, end, strand, and subfamily of every TE locus (with `--te_coordinates`).
- `{prefix}_run_manifest.json`: input for `majec_build_db`; records the MAJEC version and which EqC builder ran.
- `{prefix}_pipeline.log`: run log.

See [docs/output_files.md](https://github.com/calico/majec/blob/master/docs/output_files.md) for column descriptions.

## How It Works

1. **Equivalence classes, in one BAM pass**: for each fragment, the candidate transcripts are those whose exons it overlaps, whose annotated splice junctions match its own, and whose exons contain its aligned blocks (a read that overhangs an exon boundary drops the transcripts it does not fit). If no transcript passes every test, MAJEC falls back to the less strict sets. The two mates' candidate sets are intersected and each fragment is counted once. Fragments with the same candidate set form an equivalence class (EqC). The NH tag separates unique alignments (NH = 1, kept if MAPQ ≥ 30) from multimappers (NH > 1).

2. **Junction evidence and fragment lengths, in the same pass**: annotated splice junctions observed in the BAM, including those in multimapping alignments, are counted and matched to transcript structures. On paired-end data, the fragment-length distribution is measured in transcript coordinates and used for TPM effective lengths.

3. **Prior construction**: junction evidence, junction completeness, subset relationships, and (optionally) transcript support levels (TSL) are combined into per-transcript priors that seed the EM. Because junction compatibility already acts when classes are built, the junction boost matters less than in 0.1.5; the completeness penalty remains important.

4. **EM with momentum**: a two-phase EM resolves unique mappers first and then allocates multimappers using those estimates. After a few standard EM iterations (`--momentum_start`, default 4), a grouped momentum phase settles the estimates near the prior-informed solution rather than iterating to a winner-take-all fixed point; runs typically finish in about 15 iterations.

5. **Output**: final gene-level, transcript-level, and (if applicable) TE locus-level count matrices, plus optional TPM and confidence metrics.

See [docs/algorithm_details.md](https://github.com/calico/majec/blob/master/docs/algorithm_details.md) for details.

## Optional Workflow Tools

MAJEC includes additional tools for downstream analysis. See [docs/post_quantification.md](https://github.com/calico/majec/blob/master/docs/post_quantification.md) for details.

- **`majec_build_db`**: consolidates results from one or more pipeline runs (via their `_run_manifest.json` files) into a single, queryable SQLite database. Supports merging chunked runs and attaching sample metadata, and serves as the input for the downstream tools.
- **`majec_add_norm_factors`**: calculates normalization size factors and adds them to the database.
- **`majec_prepare_deseq2`**: generates ready-to-run DESeq2 analysis packages from the database. Supports flexible sample group definitions, confidence-weighted variance modeling, gene/transcript/junction/delta-PSI analysis levels, and batch correction.
- **`majec_visualize`**: generates interactive, multi-panel HTML reports for individual genes, including junction arc plots, differential splicing heatmaps, per-sample penalty diagnostics, and optional Excel export.

Two helper tools are also installed: `majec_calc_thresholds` (cohort-wide expression thresholds for `--use_bins`) and `majec_calc_frag_len` (fragment-length statistics for the optional `--frag_stats_dir` override). See [docs/usage_guide.md](https://github.com/calico/majec/blob/master/docs/usage_guide.md).

## Key Parameters

Run `majec_run_pipeline --help` or `majec_precompute_annotations --help` for the full list of options. The most important flags are shown in the Quick Start above. A few others worth noting:

| Flag | Description |
|------|-------------|
| `--threads` | Number of worker processes (default: 8). Samples are processed in parallel; each sample's BAM scan is single-threaded, so threads beyond the number of samples do not speed up the scan. |
| `--output_tpm` | Output TPM values in addition to counts. On paired-end data, effective lengths use the measured fragment-length distribution. Use counts, not TPM, for TE loci: they are genomic intervals, and very short loci get inflated TPM. On 3′-biased polyA libraries, long transcripts receive fewer reads than their length implies, so any length correction (MAJEC's or another tool's) over-corrects them in TPM. |
| `--mean_fragment_length` | A single global mean fragment length for effective lengths. Use this for single-end data, where fragment lengths cannot be measured from the BAM. |
| `--te_coordinates` | Write `{prefix}_TE_locus_coordinates.tsv`, mapping each TE locus to its genomic coordinates and subfamily. Off by default because the table is large (one row per TE locus). |
| `--light` | Skip prior tracking and confidence metrics for faster, leaner runs. |
| `--use_cache` | Cache each sample's EqCs so reruns with different prior or EM parameters skip the BAM scan. Add `--read_only_cache` for parallel parameter sweeps. |
| `--config` | Load settings from a JSON file (CLI flags override). Keys that do not match a current option are ignored without an error, so remove options that 0.2.0 no longer has. |

## Documentation

- [Installation](https://github.com/calico/majec/blob/master/docs/installation.md): pip, conda, and bioconda installs; the compiled and pure-Python builders; optional external tools
- [BAM Preparation](https://github.com/calico/majec/blob/master/docs/bam_preparation.md): STAR alignment parameters for TE-aware quantification
- [Algorithm Details](https://github.com/calico/majec/blob/master/docs/algorithm_details.md): EqC construction, two-phase EM, prior adjustment models, momentum acceleration, confidence metrics
- [Output Files](https://github.com/calico/majec/blob/master/docs/output_files.md): column descriptions for all output tables and confidence reports
- [Post-Quantification Workflows](https://github.com/calico/majec/blob/master/docs/post_quantification.md): database building, DESeq2 integration, and visualization
- [Usage Guide](https://github.com/calico/majec/blob/master/docs/usage_guide.md): configuration templates, utility tools, resource requirements, and HPC deployment
- [0.2.0 White Paper](https://github.com/calico/majec/blob/master/docs/whitepaper.md): what changed from 0.1.5, and how the two versions compare on spike-in, long-read and TE benchmarks
- [Changelog](https://github.com/calico/majec/blob/master/CHANGELOG.md): changes in each release and notes on migrating from 0.1.5

## Citation

If you use MAJEC in your research, please cite:

> Lim, T.-Y. & Firestone, A.J. (2026) MAJEC: unified gene, isoform, and locus-level transposable element quantification from RNA-seq. *Bioinformatics*, 42(10). https://doi.org/10.1093/bioinformatics/btag737

## License

Apache License 2.0. See [LICENSE](https://github.com/calico/majec/blob/master/LICENSE) for details.
