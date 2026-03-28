# BAM Preparation for MAJEC

MAJEC requires coordinate-sorted, indexed BAM files as input. Alignment strategy significantly impacts quantification accuracy, particularly for transposable elements that rely on multi-mapping reads.

## STAR Alignment

### Recommended Parameters

The following STAR command was used for the benchmarks in the MAJEC paper:

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

### Critical Parameters for TE Quantification

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--outFilterMultimapNmax 100` | Allow up to 100 alignments per read | Essential for TE quantification; many TE reads map to 10–50+ locations |
| `--outSAMmultNmax 100` | Report up to 100 alignments in BAM | featureCounts needs to see all alignments for correct equivalence classes |
| `--winAnchorMultimapNmax 100` | Allow multimappers in splice junction detection | Enables proper isoform detection for genes with repetitive sequences |
| `--outMultimapperOrder Random` | Randomize alignment order | Avoids systematic bias toward specific TE copies when hitting alignment limits |
| `--outFilterMismatchNmax 999` | Disable hard mismatch cap | Used in conjunction with the fraction-based filter below; not a typo |
| `--outFilterMismatchNoverLmax 0.04` | Allow mismatches up to 4% of read length | More adaptive than a hard cap — scales with read length (e.g., ~6 mismatches for 150bp reads). This is the parameter actually controlling mismatch stringency |
| `--outFilterIntronMotifs RemoveNoncanonical` | Remove non-canonical splice junctions | Reduces spurious junction calls that can confuse isoform resolution |

**Note:** STAR parameters are dataset-dependent. The values above are a reasonable starting point, but you may need to adjust for your read length, genome, and library type. The multimap parameters (`100`) are the most important to get right for TE work.

## Post-Alignment

### Index the BAM

```bash
samtools index sample_Aligned.sortedByCoord.out.bam
```

### Verify the BAM

```bash
# Quick integrity check
samtools quickcheck sample_Aligned.sortedByCoord.out.bam

# Alignment statistics
samtools flagstat sample_Aligned.sortedByCoord.out.bam
```

### Check Multi-mapper Rate

```bash
samtools view sample_Aligned.sortedByCoord.out.bam | \
    awk '$5 < 255 {multi++} $5 == 255 {unique++} END {
        print "Unique mappers:", unique;
        print "Multi-mappers:", multi;
        print "Multi-mapper fraction:", multi/(unique+multi)
    }'
```

A multi-mapper fraction of 15–25% is typical for human RNA-seq with TE capture. If below 10%, consider increasing `--outFilterMultimapNmax`.

### Verify Chromosome Naming

Chromosome names in the BAM must match your GTF annotation:

```bash
# BAM chromosomes
samtools view sample_Aligned.sortedByCoord.out.bam | head -1000 | cut -f3 | sort -u

# GTF chromosomes
grep "^[^#]" your_annotation.gtf | cut -f1 | sort -u
```

## Expected Files

After alignment, you should have:

```
sample_Aligned.sortedByCoord.out.bam      # Main input for MAJEC
sample_Aligned.sortedByCoord.out.bam.bai  # BAM index
sample_Log.final.out                      # Alignment statistics
sample_SJ.out.tab                         # Splice junction table
```
