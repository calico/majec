# BAM Preparation for MAJEC

MAJEC reads genome-aligned BAM files directly. It reads each BAM once and builds equivalence classes (EqCs) from the alignments, so no featureCounts, samtools or bedtools step is needed before `majec_run_pipeline`. How you align affects the result, most of all for transposable elements (TEs) and gene families, which depend on multimapping reads.

In short:

- Use a splice-aware aligner that writes the `NH` tag (STAR or HISAT2), and align to the **genome**. STAR's `Aligned.toTranscriptome.out.bam` is not a valid input.
- **Keep secondary alignments.** They tell MAJEC which other loci a multimapping read could have come from.
- Write `HI` as well for paired-end data.
- Use coordinate-sorted BAMs, and give each BAM a name that resolves to a distinct sample name (see [Sample names](#sample-names)).
- Pass `--paired_end` and `--strandedness` to `majec_run_pipeline` to match the library.

## STAR Alignment

### Recommended Parameters

The following STAR command was used for the benchmarks in the MAJEC paper. It works unchanged with 0.2.0:

```bash
STAR \
    --genomeDir /path/to/STAR_index \
    --readFilesIn sample_R1.fastq.gz sample_R2.fastq.gz \
    --readFilesCommand zcat \
    --outFileNamePrefix sample_ \
    --outSAMtype BAM SortedByCoordinate \
    --outSAMunmapped Within \
    --outSAMattributes NH HI AS NM MD \
    --outFilterType BySJout \
    --outFilterMultimapNmax 100 \
    --outSAMmultNmax 100 \
    --outMultimapperOrder Random \
    --winAnchorMultimapNmax 100 \
    --outFilterMismatchNmax 999 \
    --outFilterMismatchNoverLmax 0.04 \
    --alignIntronMin 20 \
    --alignIntronMax 1000000 \
    --alignMatesGapMax 1000000 \
    --alignSJoverhangMin 8 \
    --alignSJDBoverhangMin 3 \
    --outFilterIntronMotifs RemoveNoncanonical \
    --runThreadN 16 \
    --genomeLoad NoSharedMemory
```

Build the STAR index with the same gene GTF you give to `majec_precompute_annotations` (`--sjdbGTFfile`), so that STAR knows the annotated junctions. MAJEC uses only annotated junctions.

### Critical Parameters for TE Quantification

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--outSAMattributes NH HI AS NM MD` | Write the NH, HI and AS tags | MAJEC needs `NH` and uses `HI` for paired-end data (see [Alignment tags](#alignment-tags-nh-hi-as)). `NM` and `MD` are not used by MAJEC. STAR's `Standard` set (`NH HI AS nM`) also works. |
| `--outFilterMultimapNmax 100` | Allow up to 100 alignments per read | Essential for TE quantification; many TE reads map to 10–50+ locations. STAR reports reads with more loci than this as unmapped, and its default of 10 discards many TE reads. |
| `--outSAMmultNmax 100` | Write up to 100 alignments per read to the BAM | MAJEC builds a multimapping read's EqC from the alignments present in the BAM. A locus that is not written cannot be a candidate. Keep this equal to `--outFilterMultimapNmax`, or at STAR's default of `-1` (write all of them). |
| `--winAnchorMultimapNmax 100` | Allow alignment seeds (anchors) that map to many loci | Lets STAR find alignments for reads from high-copy repeats and repetitive genes. |
| `--outMultimapperOrder Random` | Randomize alignment order | Avoids systematic bias toward specific TE copies when not every alignment is reported. |
| `--outFilterMismatchNmax 999` | Disable the hard mismatch cap | Used together with the fraction-based filter below. This is not a typo. |
| `--outFilterMismatchNoverLmax 0.04` | Allow mismatches up to 4% of read length | This is the parameter that actually controls mismatch stringency. It scales with read length (about 6 mismatches for 150 bp reads), so it adapts better than a hard cap. |
| `--outFilterIntronMotifs RemoveNoncanonical` | Remove non-canonical splice junctions | Reduces spurious junction calls that can confuse isoform resolution. |

**Note:** STAR parameters are dataset-dependent. The values above are a reasonable starting point, but you may need to adjust for your read length, genome, and library type. For TE work, the multimapping parameters (`100`) matter most.

### Other Aligners

- **HISAT2** writes `NH`. It reports at most `-k` alignments per read (default 5 for a standard index), so raise `-k` for TE work. HISAT2 writes SAM, so sort it into a BAM, for example `hisat2 ... | samtools sort -o sample.bam`. Use the [tag check](#check-the-alignment-tags) to see which tags your BAM carries.
- **Short-read aligners that do not write NH** are not a supported input. Without NH, the pure-Python builder (pysam) counts every read as unique. With the compiled builder (majec-core), the experimental long-read mode switches on (see [Long reads](#long-reads-minimap2-experimental)).

## What MAJEC Reads From a BAM

### Alignment tags (NH, HI, AS)

| Tag | Used for | If missing |
|-----|----------|------------|
| `NH` | Multiplicity. An alignment with `NH` = 1 and MAPQ ≥ 30 is unique. An alignment with `NH` > 1 belongs to a multimapper, whatever its MAPQ. Unique alignments with MAPQ < 30 are not counted. | With majec-core and the default `--derive_multiplicity auto`, MAJEC derives multiplicity from read names if none of the first 10,000 mapped records carries `NH` (experimental, see below). With the pure-Python builder, reads without `NH` are counted as unique, their secondary alignments are discarded, and the log warns when more than 1% of primary alignments lack the tag. |
| `HI` | Pairing the two mates of each alignment of a multimapping paired-end fragment, so that mates are matched at the same locus. | All alignments of a fragment share one key, so a mate at one locus can be paired with a mate at another. |
| `AS` | Only the experimental no-NH mode, and only when `--lr_score_ratio` is above 0. | Not needed for short reads. |

### Which records are used

- **Secondary alignments** (flag 0x100) are used. They carry the other loci of multimapping reads. Do not remove them (for example with `samtools view -F 0x100`), and do not filter the BAM by MAPQ before running MAJEC. Either step removes multimappers or hides their alternative loci.
- **Supplementary alignments** (flag 0x800) are skipped.
- **Unmapped records** are counted in the scan statistics and otherwise ignored.
- **Duplicate and QC-fail flags** (0x400, 0x200) are not checked, so those records are counted. Remove them beforehand if you want them excluded.
- Reads on contigs that are absent from the annotation contribute nothing. Check that [chromosome names match](#verify-chromosome-naming).

### Sorting and indexing

`majec_run_pipeline` streams each BAM from start to end. It does not read the index and does not depend on sort order. A multimapping read's alignments are collected by read name across the whole file, and mates are matched by read name and `HI`.

Coordinate-sorted BAMs, as written by STAR with `--outSAMtype BAM SortedByCoordinate`, are the tested input. MAJEC itself does not need an index, but samtools and genome browsers use one, so indexing is still worthwhile.

### Strandedness

Set `--strandedness` to match the library. The values follow featureCounts `-s`:

| `--strandedness` | Library | HTSeq / TopHat equivalent |
|------------------|---------|---------------------------|
| `0` | Unstranded | `--stranded no` / `fr-unstranded` |
| `1` | Stranded; read 1 is on the transcript strand | `--stranded yes` / `fr-secondstrand` |
| `2` | Reverse-stranded; read 1 is antisense to the transcript (dUTP-based kits such as Illumina TruSeq Stranded) | `--stranded reverse` / `fr-firststrand` |

The `majec_run_pipeline` default is `0`, so set it explicitly for stranded libraries. `majec_calc_thresholds` defaults to `2`. Use the same value for both. If you are unsure, check one BAM with a tool such as RSeQC's `infer_experiment.py`. Single-end reads are treated as read 1.

### Single-end and paired-end data

- **Paired-end:** pass `--paired_end`. Each fragment is counted once.
  - The two mates' candidate transcript sets are intersected (if the intersection is empty, the union is used).
  - A mate whose partner never appears, for example because it is unmapped or lies outside annotated exons, is counted on its own.
  - The fragment-length distribution is measured from uniquely mapped pairs during the same scan and used for TPM effective lengths.
- **Single-end:** omit `--paired_end`. Fragment lengths cannot be measured, so TPM uses annotated transcript lengths unless you pass `--mean_fragment_length`.
- If you omit `--paired_end` on paired-end data, each mate is counted as a separate fragment. Counts roughly double, the mates are not intersected, and no fragment lengths are measured.
- `--paired_end` and `--strandedness` apply to every BAM in a run, so quantify libraries of different layouts or strandedness in separate runs.

The counting unit for paired-end data is the fragment. In 0.1.5, each mate of a uniquely mapped pair was counted separately, so 0.2.0 unique counts are about half of 0.1.5's. Single-end counts are unaffected. See [`CHANGELOG.md`](../CHANGELOG.md).

## Sample Names

MAJEC names each sample after its BAM file. It removes a chain of trailing suffixes separated by `.` or `_`: `Aligned`, `sortedByCoord`, `toTranscriptome`, `out`, `sortedByName`, `sorted`, `dedup`, `mkdup`, `pe`, `se`, `bam`, `sam` and `cram` (case-insensitive). It then removes one leading `sorted_` or `dedup_`. If nothing informative is left (an empty name, `aligned`, `output` or `results`), the name of the BAM's parent directory is used instead.

| BAM path | Sample name |
|----------|-------------|
| `star/S1_Aligned.sortedByCoord.out.bam` | `S1` |
| `star/S1/Aligned.sortedByCoord.out.bam` | `S1` (parent directory) |
| `bams/ctrl_rep1.sorted.dedup.bam` | `ctrl_rep1` |
| `bams/sorted_A.bam` | `A` |
| `bams/sample_se.bam` | `sample` |

- Every BAM in a run must resolve to a different name. When you merge runs with `majec_build_db`, a sample may appear in only one run.
- These names must match the `sample_id` column of the metadata file used by `majec_build_db` (see [post_quantification.md](post_quantification.md)).
- In 0.1.5, STAR's default `Aligned.sortedByCoord.out.bam` was named `Aligned`. Update existing metadata files if your BAMs use that name.

## Long Reads (minimap2, Experimental)

minimap2 does not write `NH`. With the compiled builder (majec-core) installed, MAJEC detects this and derives each read's multiplicity from how many alignments carry its name, so multimapping long reads reach the EM. The pure-Python builder cannot do this: it stops with an error if you pass `--derive_multiplicity on`, `--lr_score_ratio` or `--lr_min_mapq`. See [installation.md](installation.md) for majec-core.

Long-read quantification is **experimental**. It has been checked for correctness on ONT libraries, but its accuracy against dedicated long-read quantifiers has not been established.

```bash
minimap2 -ax splice genome.fa sample.fastq.gz | samtools sort -o sample.bam
```

- Keep secondary alignments: do not pass `--secondary=no`. minimap2 writes up to `-N` secondary alignments per read (default 5).
- In this mode, reads are processed as single-end, even if `--paired_end` is passed.
- `--lr_score_ratio` (default 0) keeps every reported alignment. Raising it assigns each read to whichever locus scores highest, which on noisy long reads can move reads between a gene and its pseudogene copies.
- `--lr_min_mapq` (default 0, off) sets a MAPQ floor for reads that resolve as unique. See [usage_guide.md](usage_guide.md).

## Post-Alignment Checks

### Index and verify the BAM

```bash
samtools index sample_Aligned.sortedByCoord.out.bam

# Quick integrity check
samtools quickcheck sample_Aligned.sortedByCoord.out.bam

# Alignment statistics
samtools flagstat sample_Aligned.sortedByCoord.out.bam

# Sort order (expect SO:coordinate)
samtools view -H sample_Aligned.sortedByCoord.out.bam | grep '^@HD'
```

### Check the alignment tags

Count how many of the first 10,000 mapped records carry each tag:

```bash
samtools view -F 0x4 sample_Aligned.sortedByCoord.out.bam | head -n 10000 | awk '
    { for (i = 12; i <= NF; i++) { t = substr($i, 1, 3); if (t == "NH:" || t == "HI:" || t == "AS:") n[t]++ } }
    END { printf "records: %d  NH: %d  HI: %d  AS: %d\n", NR, n["NH:"], n["HI:"], n["AS:"] }'
```

For a STAR BAM aligned as above, all three counts equal the number of records.

### Check the multimapper rate

STAR reports this in its final log:

```bash
grep -E "Uniquely mapped reads %|mapped to multiple loci|mapped to too many loci" sample_Log.final.out
```

A large "% of reads mapped to too many loci" means `--outFilterMultimapNmax` is discarding reads; raise it. For any aligner, you can count primary alignments by `NH`:

```bash
samtools view -F 0x904 sample_Aligned.sortedByCoord.out.bam | awk '
    { nh = 1; for (i = 12; i <= NF; i++) if ($i ~ /^NH:i:/) { nh = substr($i, 6); break }
      if (nh == 1) unique++; else multi++ }
    END { printf "Unique: %d\nMulti: %d\nMulti fraction: %.3f\n", unique, multi, multi / (unique + multi) }'
```

The multimapper fraction depends on the genome, the library type (total RNA vs polyA) and the aligner settings. Compare it with similar libraries rather than with a fixed threshold.

### Verify chromosome naming

Chromosome names in the BAM must match your GTF annotations (gene and TE):

```bash
# BAM chromosomes (from the header)
samtools view -H sample_Aligned.sortedByCoord.out.bam | awk '$1 == "@SQ" { sub(/^SN:/, "", $2); print $2 }' | head

# GTF chromosomes
grep -v '^#' your_annotation.gtf | cut -f1 | sort -u | head
```

## Expected Files

After alignment, you should have:

```
sample_Aligned.sortedByCoord.out.bam      # Main input for MAJEC
sample_Aligned.sortedByCoord.out.bam.bai  # BAM index (optional for MAJEC)
sample_Log.final.out                      # Alignment statistics
sample_SJ.out.tab                         # Splice junction table (not used by MAJEC)
```

Next: run `majec_run_pipeline` as described in [usage_guide.md](usage_guide.md).
