# MAJEC

**Momentum-Accelerated Junction-Enhanced Counting** — unified gene, isoform, and locus-level transposable element quantification from RNA-seq.

MAJEC jointly quantifies genes, transcript isoforms, and individual TE loci from BAM alignments in a single pass using a junction-informed Expectation-Maximization algorithm operating on a joint gene+TE feature space.

## Installation

### From source (recommended for now)

```bash
mamba env create -f majec.yml
conda activate majec
pip install -e .
```

### Dependencies

MAJEC requires the following command-line tools (installed automatically via the conda environment):

- [Subread](http://subread.sourceforge.net/) (featureCounts) ≥2.0
- [samtools](http://www.htslib.org/) ≥1.20
- [bedtools](https://bedtools.readthedocs.io/) ≥2.31

## Quick Start

### 1. Precompute annotations

```bash
majec_precompute_annotations \
    --gene_gtf gencode.v44.annotation.gtf \
    --te_gtf hg38_rmsk_TE.gtf \
    --output my_annotations
```

### 2. Run the pipeline

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

## Key Features

- **Joint gene+TE feature space**: genes and TE loci compete for reads probabilistically, preventing systematic misattribution at overlapping loci
- **Junction-informed priors**: splice junction evidence improves isoform accuracy and provides the basis for distinguishing genic from TE-derived signal
- **Locus-level TE resolution**: individual TE insertions are quantified, not just subfamily aggregates
- **Confidence metrics**: per-transcript distinguishability scores, assignment entropy, and discord scores for quality assessment
- **Fast**: processes all samples jointly via multiprocessing; momentum-accelerated EM converges in ~15 iterations

## Citation

If you use MAJEC in your research, please cite:

> Firestone, A.J. et al. (2026) MAJEC: unified gene, isoform, and locus-level transposable element quantification from RNA-seq. *In preparation.*

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
