# MAJEC

![MAJEC Joint Gene and TE Quantification Workflow](/images/<img width="2816" height="1536" alt="Gemini_Generated_Image_7e14am7e14am7e14" src="https://github.com/user-attachments/assets/c3a50e27-7bf0-45b5-9e23-8aa55d3551a3" />
)

**Momentum-Accelerated Junction-Enhanced Counting** — unified gene, isoform, and locus-level transposable element quantification from RNA-seq.

MAJEC jointly quantifies genes, transcript isoforms, and individual TE loci from standard BAM alignments in a single pass. By operating a probabilistically resolved Expectation-Maximization (EM) algorithm on a **joint gene+TE feature space**, MAJEC eliminates the systematic signal misattribution that plagues existing TE quantification tools.

## 🛑 The Gene-TE Overlap Problem

Approximately 45% of the human genome consists of TEs, many of which are embedded directly within gene bodies. 

* **TE-only tools:** Operate blind to genes. When a host gene is transcribed, its reads are falsely attributed to the overlapping TE, driving massive false-positive TE reactivation calls.

**The MAJEC Solution:** Genes and TEs compete for reads probabilistically. MAJEC uses empirical splice junction evidence to heavily penalize unsupported transcript isoforms, preventing them from stealing genuine TE reads, while correctly assigning spliced genic reads to their host genes.

*Dependencies (installed automatically via conda): Subread (featureCounts) ≥2.0, samtools ≥1.20, bedtools ≥2.31.*

### 2. Precompute Annotations

Combine your gene and TE GTFs into a unified, mathematically optimized index.

Bash

```
majec_precompute_annotations \
    --gene_gtf gencode.v44.annotation.gtf \
    --te_gtf hg38_rmsk_TE.gtf \
    --output my_annotations
```

### 3. Run the Pipeline

Quantify all your samples jointly in a single command.

Bash

```
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

------

## 📊 Outputs

MAJEC automatically aggregates locus-level estimates and provides DESeq2-ready matrices at multiple resolutions:



- `_total_EM_aggregated_counts.tsv` (Gene & Subfamily level)
- `_total_EM_counts.tsv` (Isoform & Locus level)
- `_transcript_metrics_SPARSE.tsv.gz` (Confidence & Junction evidence)

------

## 📖 Citation

If you use MAJEC in your research, please cite:

> Lim, T.-Y. & Firestone, A.J. (2026) MAJEC: unified gene, isoform, and locus-level transposable element quantification from RNA-seq. *bioRxiv*.

## ⚖️ License

Apache License 2.0. See [LICENSE](https://www.google.com/search?q=LICENSE) for details.

> Firestone, A.J. et al. (2026) MAJEC: unified gene, isoform, and locus-level transposable element quantification from RNA-seq. *In preparation.*

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
