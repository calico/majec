# Usage Guide

This guide covers running MAJEC 0.2.0: building the annotation, running `majec_run_pipeline`, choosing flags, caching, configuration files and cluster deployment. It ends with a migration note for 0.1.5 users and a full parameter reference.

Related documents:

- [Installation](installation.md): installing MAJEC and the compiled builder (majec-core).
- [BAM Preparation](bam_preparation.md): aligner settings, especially for TEs.
- [Output Files](output_files.md): every file the pipeline writes, with column descriptions.
- [Algorithm Details](algorithm_details.md): how equivalence classes, priors, the EM and TPM work.
- [Post-Quantification Workflows](post_quantification.md): database, DESeq2 and visualization tools.

A MAJEC analysis has two required steps:

1. Build the annotation once with `majec_precompute_annotations`.
2. Quantify all samples with `majec_run_pipeline`.

---

## 1. Precompute the annotation

`majec_precompute_annotations` parses a gene GTF and, optionally, a TE GTF into a single annotation file. Build it once per annotation and reuse it for every run.

```bash
majec_precompute_annotations \
    --gene_gtf gencode.v44.annotation.gtf \
    --te_gtf hg38_rmsk_TE.gtf \
    --output ref/gencode_v44_rmsk
```

`--output` is a **prefix**. MAJEC appends fixed suffixes to it and writes two files:

| File | Contents |
|---|---|
| `ref/gencode_v44_rmsk_annotations.pkl.gz` | The annotation. Pass this to `majec_run_pipeline --annotation`. |
| `ref/gencode_v44_rmsk_annotations_summary.json` | Metadata and statistics: source GTF paths and checksums, creation date, transcript/gene/TE counts, junction counts and TSL distribution. |

Notes:

- The default prefix is `majec`, which gives `majec_annotations.pkl.gz`.
- Don't end the prefix in `_annotations`, or you get `..._annotations_annotations.pkl.gz`.
- The output directory must already exist.
- For a gene-only analysis, leave out `--te_gtf`.

### GTF requirements

- **Uncompressed GTFs.** The parser reads plain text, so decompress `.gtf.gz` files first.
- **Gene GTF.**
  - `exon` lines carry `transcript_id` and `gene_id`.
  - Genes are aggregated by `gene_id`. With GENCODE, that gives Ensembl gene IDs; with a refGene GTF, it gives gene symbols.
  - Transcript Support Level is read from the `transcript_support_level` (or `tsl`) attribute of `transcript` lines, as in GENCODE and Ensembl GTFs. Other annotations have no TSL, and `--use_tsl_penalty` is then skipped with a warning.
- **TE GTF.** Use a TEtranscripts-style RepeatMasker GTF such as `hg38_rmsk_TE.gtf`:
  - `transcript_id` is one TE locus (for example `L1HS_dup1234`);
  - `gene_id` is the subfamily;
  - `family_id` and `class_id` give the family and class.
  - TE subfamilies are aggregated as `name:family:class` (for example `AluYa5:Alu:SINE`). Loci that lack `family_id` or `class_id` fall back to the bare subfamily name.
- **No shared transcript IDs.** Transcript IDs must not repeat between the gene and TE GTFs. Precompute stops with an error if they do.

### Portability and validation

The annotation file holds everything quantification needs, so you can move or delete the GTFs after precompute. To check that an annotation file still matches a set of GTFs (by checksum):

```bash
majec_precompute_annotations \
    --validate ref/gencode_v44_rmsk_annotations.pkl.gz \
    --gene_gtf gencode.v44.annotation.gtf \
    --te_gtf hg38_rmsk_TE.gtf
echo $?   # 0 = matches, 1 = a GTF has changed
```

`--gene_gtf` is required even in validate mode.

**Re-run precompute when you upgrade from 0.1.x.** Older annotation files still load, but the pipeline then rebuilds part of the index at run time by re-reading the GTFs from their original paths, which is slow.

---

## 2. Run the pipeline

### Minimal run

```bash
majec_run_pipeline \
    --annotation ref/gencode_v44_rmsk_annotations.pkl.gz \
    --bams sample1.bam sample2.bam sample3.bam \
    --prefix results/my_experiment \
    --paired_end \
    --strandedness 2
```

- **Required:** `--annotation` and `--bams`. Either can come from a `--config` file instead.
- **Prefix:** `--prefix` defaults to `MAJEC_output`. It may include a directory, which MAJEC creates if needed.
- **BAMs:**
  - must be coordinate-sorted;
  - need no index (`.bai`);
  - should come from a splice-aware aligner that writes the `NH` tag, such as STAR or HISAT2. See [BAM Preparation](bam_preparation.md).
- **Run all samples of an experiment together.** One joint run shares the momentum binning across samples and writes one matrix per output type.

### Recommended run (isoform-level analysis)

```bash
majec_run_pipeline \
    --annotation ref/gencode_v44_rmsk_annotations.pkl.gz \
    --bams sample1.bam sample2.bam sample3.bam \
    --prefix results/my_experiment \
    --paired_end \
    --strandedness 2 \
    --threads 3 \
    --use_junction_completeness \
    --library_type dT \
    --terminal_relax \
    --use_subset_penalty \
    --use_tsl_penalty \
    --output_tpm \
    --output_confidence \
    --use_cache
```

- **Junction boosts** are on by default (`--junction_weight 3.0`). Set `--junction_weight 0` to turn them off.
- **Other priors** are opt-in: `--use_junction_completeness`, `--use_subset_penalty` and `--use_tsl_penalty`.
- **`--use_cache`** saves each sample's equivalence classes (EqCs). Later runs with different prior or EM settings then skip the BAM scan, and the run writes `{prefix}_run_manifest.json`, which `majec_build_db` reads (see [Caching](#caching)).

### Gene vs. isoform quantification

The junction completeness and subset penalties resolve ambiguity at the **isoform** level. Aggregated to genes, their effect is usually small. If you only need differential gene expression, the baseline configuration below is enough. For anything involving isoforms or splicing, use the recommended settings.

In 0.2.0, junction evidence already narrows each fragment's candidate transcripts when the EqCs are built, so some of the work the priors did in 0.1.5 now happens earlier. In testing:

- completeness penalties remained important;
- junction boosts mattered much less than in 0.1.5;
- subset penalties mattered about half as much as in 0.1.5.

The recommended configuration still enables all of them.

### Library settings

**Paired-end.** Pass `--paired_end` for paired-end libraries. Mates are then combined into one fragment: the fragment's candidate set is the intersection of the two mates', and each fragment counts once. Leave the flag off for single-end data.

**Strandedness** (`--strandedness`, default `0`):

| Value | Meaning | Example |
|---|---|---|
| `0` | Unstranded | |
| `1` | Stranded: read 1 lies on the transcript strand | |
| `2` | Reverse-stranded: read 1 lies opposite the transcript strand | dUTP-based kits such as Illumina TruSeq Stranded |

**Library type and terminal relaxation.** With `--use_junction_completeness`, transcripts are penalized when some of their junctions have far fewer reads than the others. Library preparation leaves predictable coverage drop-off at one end of the transcript. `--terminal_relax` relaxes the penalty there, and `--library_type` says which end:

| `--library_type` | End relaxed by `--terminal_relax` |
|---|---|
| `dT`, `polyA` | 5′ end (oligo-dT priming and poly(A) selection lose coverage toward the 5′ end) |
| `random` | 3′ end |
| `none` | No relaxation |

`--terminal_relax` without `--library_type` is an error. `--terminal_recovery_rate` (default 2) sets how quickly the expected coverage recovers away from the terminus; higher values are more lenient.

### Threads and memory

The per-sample BAM scan is single-threaded with either builder, and so is the per-sample EM.

- **What `--threads` does.** `--threads` sets how many samples are processed in parallel, capped at the number of BAMs. A single-sample run uses one core no matter what `--threads` says.
- **Matching threads to samples.** Set `--threads` to the number of BAMs, or lower it to cap memory. Each parallel worker holds its own sample's data, so peak memory grows with the number of samples processed at once.
- **Scan speed.** The compiled builder (majec-core) scans BAMs roughly 6× faster than the pure-Python builder (pysam), and the two give the same results. The log's first lines say which one ran. See [Installation](installation.md#choosing-a-builder).

### Sample names

Sample names (the column headers and the `{sample}` part of per-sample file names) come from the BAM file name, after removing common suffixes: `Aligned`, `sortedByCoord`, `out`, `sorted`, `dedup`, `mkdup`, `pe`, `se`, `.bam` and so on.

- `sample1_Aligned.sortedByCoord.out.bam` becomes `sample1`.
- `s3.sorted.bam` becomes `s3`.
- If nothing informative is left, the parent directory name is used. STAR's default `sample_01/Aligned.sortedByCoord.out.bam` becomes `sample_01`.

Make sure every BAM in a run gets a distinct name.

---

## TPM and effective lengths

`--output_tpm` writes TPM matrices next to the counts. TPM divides each transcript's count by its effective length and rescales each sample to one million. The effective length is the number of positions in the transcript where a fragment of the observed length distribution could start.

**Paired-end data (default).**

- MAJEC measures the fragment-length distribution during the same BAM scan that builds the EqCs.
- It uses uniquely mapped pairs, measured in transcript coordinates, and uses the measured fragment-length distribution for the effective lengths.
- No separate prepass is needed.
- The measured lengths agree closely with Salmon's and RSEM's estimates. The gain is correctness and convenience, not a claimed accuracy improvement.

The log records the distribution for each sample:

```
[sample1] Fragment length: n=45,584,872, mean=214.2, sd=109.9, median=194, p90=321, p99=504
[sample1] Using measured fragment distribution (n=45,584,872 pairs, mean=214.19, sd=109.89, 6,673 bins) for effective lengths.
```

**Overrides.** You can pass one of these, but not both:

- **`--mean_fragment_length N`** uses one mean fragment length (no SD) for all samples. Use it for single-end data, where fragment lengths can't be measured.
- **`--frag_stats_dir DIR`** uses per-sample mean and SD from `majec_calc_frag_len`, modelled as a truncated normal.
  - Those statistics come from the genomic template length (TLEN), which overstates the spread on spliced libraries, and the pipeline warns when the override is used.
  - Use it only to reproduce a TPM run made before 0.2.0.
  - Files are matched by BAM file name: `{BAM name without .bam}_cDNA_frag_stats.json`.

**Single-end data** without `--mean_fragment_length` falls back to annotated transcript lengths, with a warning.

**Caveats when reading TPM:**

- **3′-biased poly(A) libraries.** Long transcripts receive fewer reads per molecule than their length implies. Every length correction (MAJEC's, Salmon's or any other) then over-corrects them in TPM. Counts are not affected.
- **TE loci: use counts, not TPM.** TE loci are genomic intervals, and very short loci get inflated TPM. Use the count matrices for locus- and subfamily-level TE analysis.

See [Algorithm Details](algorithm_details.md#effective-length-and-tpm) for the effective-length formula.

---

## TE quantification

With a TE GTF in the annotation, genes and individual TE loci compete for reads in the same EM.

- **Locus level:** each TE locus gets a row in `{prefix}_total_EM_counts.tsv`.
- **Subfamily level:** loci are summed into `name:family:class` subfamilies in `{prefix}_total_EM_aggregated_counts.tsv`.
- **Coordinates:** add `--te_coordinates` to write `{prefix}_TE_locus_coordinates.tsv`, which maps each locus to its chromosome, start, end, strand and subfamily.
  - It is off by default because it is large: about 4.7M rows and ~260 MB for hg38 rmsk.
  - See [Output Files](output_files.md#te-locus-coordinates-_te_locus_coordinatestsv).

Align with multimapping reads retained (see [BAM Preparation](bam_preparation.md)), and analyse TE loci with counts rather than TPM.

---

## Caching

The BAM scan is the slowest step. With `--use_cache`, MAJEC saves each sample's scan results in `--cache_dir` (default `pipeline_cache`): the EqCs, junction counts and the measured fragment-length histogram. Priors, the EM, TPM and confidence metrics are recomputed from the cache on every run. You can therefore change any prior, EM or output option and rerun without rescanning.

| Flag | Effect |
|---|---|
| `--use_cache` | Use a valid cache if one exists; otherwise scan the BAM and save the result. |
| `--cache_dir DIR` | Where cache files live (default `pipeline_cache`). |
| `--rebuild_cache` | Ignore existing caches, rescan, and save new ones. |
| `--read_only_cache` | Use valid caches but never write. Combine it with `--use_cache`. Use it when several jobs share one cache directory, for example parameter sweeps. |

**What invalidates a cache.** A cache is reused only when these match:

- the BAM's size and modification time;
- the annotation's checksums and creation date (so re-running precompute invalidates every cache);
- `--strandedness`;
- `--paired_end`.

The long-read options (`--derive_multiplicity`, `--lr_score_ratio`, `--lr_min_mapq`) join the cache key when they are set away from their defaults, so changing them builds a separate cache.

**Caches and versions.** Caches written by 0.1.x are not reused (the cache format is now 2.0). The first 0.2.0 run with `--use_cache` rebuilds them, and you can delete old cache directories.

**Run manifest.** `{prefix}_run_manifest.json` is written on every run. `majec_build_db` reads it; its per-sample junction-coverage tables come from the cache files, so use `--use_cache` for runs you intend to load into a database.

A typical parameter sweep:

```bash
# 1. Build the cache once
majec_run_pipeline --annotation ann_annotations.pkl.gz --bams *.bam \
    --paired_end --strandedness 2 --threads 8 \
    --use_cache --cache_dir cache --prefix sweep/baseline

# 2. Vary priors against the cache (safe to run in parallel)
majec_run_pipeline --annotation ann_annotations.pkl.gz --bams *.bam \
    --paired_end --strandedness 2 --threads 8 \
    --use_cache --read_only_cache --cache_dir cache \
    --junction_weight 0 --prefix sweep/no_boost
```

---

## Configuration files

Any option can be given in a JSON file with `--config`.

- **Keys** are the option names without the leading `--`.
- **Values** use the option's type: `true`/`false` for switches, lists for `bams` and `momentum_scaling`.
- **Command-line options override the file.**

Things to know:

- **Switches can't be turned off from the command line.** If the file sets a switch to `true`, no command-line flag turns it off. Edit the file instead.
- **Unknown keys are silently ignored.** That includes `_comment` keys and options removed in 0.2.0 (`use_subset_coverage_data`, `use_paired_rescue`, `no_paired_rescue`, `paired_rescue_decay`). Remove those so a saved config doesn't appear to enable something it no longer does.
- **`tsl_penalty_values` must be a JSON string**, as on the command line: `"tsl_penalty_values": "{\"1\": 1.0, \"2\": 0.8}"`.

### Baseline gene-level analysis

Junction-aware gene counts without the isoform penalties:

```json
{
  "_comment": "Fast baseline for gene-level counts.",
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

### Isoform-level analysis (recommended)

All evidence priors and confidence metrics:

```json
{
  "_comment": "Recommended settings for isoform-level analysis.",
  "threads": 24,
  "prefix": "isoform_run",
  "strandedness": 2,
  "paired_end": true,
  "use_junction_completeness": true,
  "library_type": "dT",
  "terminal_relax": true,
  "use_subset_penalty": true,
  "use_tsl_penalty": true,
  "output_tpm": true,
  "output_confidence": true,
  "calculate_group_confidence": true,
  "verbose_output": true,
  "use_cache": true,
  "cache_dir": "./majec_cache"
}
```

Set `library_type` to match your protocol. `use_tsl_penalty` has an effect only with an annotation that carries TSL (GENCODE or Ensembl).

Run with:

```bash
majec_run_pipeline --config isoform_run.json \
    --annotation ann_annotations.pkl.gz --bams *.bam
```

---

## Output options

### Confidence metrics

`--output_confidence` computes per-transcript reliability metrics: assignment entropy, distinguishability from competing transcripts, junction support and discord between prior and EM estimates. It writes per-sample and cohort-level tables.

| Flag | Requires | Adds |
|---|---|---|
| `--output_confidence` | | Per-sample `_transcript_metrics_SPARSE.tsv.gz` and `_high_confidence_counts.tsv.gz`, and the cohort `_transcript_confidence_summary.tsv.gz`. |
| `--verbose_output` | | `_counts_with_confidence.tsv.gz` and `_distinguishability_detailed.tsv.gz` per sample (when `--output_confidence` is on), plus the prior-adjustment reports and unique/multimapper count splits. |
| `--calculate_group_confidence` | `--output_confidence` | Gene and TE-subfamily metrics in `_group_confidence_comprehensive.tsv.gz`. Without `--output_confidence`, it is switched off with a warning. |
| `--analyze_competition` | `--output_confidence` | Competition-type summaries. |
| `--confidence_min_expression` | | Minimum count for a transcript to get metrics (default 1.0). |

See [Output Files](output_files.md#confidence-reports) for columns and [Algorithm Details](algorithm_details.md#confidence-metrics) for definitions.

### Light mode

`--light` skips everything that isn't a count or TPM matrix:

- prior tracking (no `_prior_adjustments.tsv.gz`, `_prior_adjustment_summary.tsv` or `_junction_completeness_stats.json`);
- confidence metrics;
- verbose outputs.

It also switches off `--verbose_output`, `--output_confidence`, `--calculate_group_confidence`, `--analyze_competition` and `--dump_em_classes` if they were given, and logs a warning.

The priors are still applied, so counts and TPM are the same as in a full run. Use it for large gene+TE runs where memory or time is tight. `--output_tpm`, `--te_coordinates` and caching work as usual.

### Debugging

`--dump_em_classes` writes each sample's EqCs and initial counts to `{prefix}_{sample}_em_classes.pkl` (see [Output Files](output_files.md#debugging-em-classes-_em_classespkl)).

---

## Long-read data (experimental)

> **Experimental.** Long-read quantification has been checked for correctness on ONT libraries. Its accuracy against dedicated long-read quantifiers has not been established.

MAJEC reads alignment multiplicity from the `NH` tag. STAR and HISAT2 write it; minimap2 doesn't. When `NH` is absent, MAJEC can instead derive multiplicity from read names: it counts the alignments carrying each read name. Multimapping reads then reach the EM rather than being counted as unique. This needs the compiled builder (majec-core).

| Flag | Default | Effect |
|---|---|---|
| `--derive_multiplicity {auto,on,off}` | `auto` | `auto` switches on only if none of the first 10,000 mapped records carries `NH`, so short-read runs are unaffected. `on` forces it; `off` disables it. |
| `--lr_score_ratio R` | `0` (off) | Keep only alignments scoring within this fraction of the read's best `AS`. Raising it hands each read to whichever locus scores highest. On noisy long reads, that can move reads between a gene and its pseudogene copies. |
| `--lr_min_mapq Q` | off | MAPQ floor for reads that resolve as unique in derive mode. minimap2 lowers MAPQ when near-equal alternatives exist, so a floor removes those reads instead of letting the EM resolve them. |

- **pysam builder.** The pure-Python builder (pysam) can't derive multiplicity. If any of these options is turned on, the pipeline stops with an error rather than ignoring it. If a BAM without `NH` reaches the pure-Python builder anyway, the log warns that every read was counted as unique.
- **Keep secondary alignments.** Multiplicity is counted from the alignments present in the BAM, so keep the aligner's secondary alignments.
- **Single-end only.** Derive mode processes reads as single-end. Don't pass `--paired_end`.
- **TPM** uses annotated transcript lengths.

A minimal long-read run:

```bash
majec_run_pipeline \
    --annotation ref/gencode_v44_annotations.pkl.gz \
    --bams ont_sample1.minimap2.sorted.bam \
    --prefix results/ont_run \
    --strandedness 0 \
    --derive_multiplicity on
```

See [Algorithm Details](algorithm_details.md#long-read-data-experimental).

---

## Utility tools

### `majec_calc_frag_len`: fragment-length statistics (optional)

This tool is no longer needed for TPM: the pipeline measures fragment lengths during the scan. It is kept so you can reproduce TPM from runs made before 0.2.0 via `--frag_stats_dir`. It samples intra-exonic read pairs with `samtools` (which must be on your `PATH`) and writes one `{BAM name without .bam}_cDNA_frag_stats.json` per BAM.

```bash
# From a file listing BAM paths (one per line)
majec_calc_frag_len -o frag_stats/ bam_list.txt

# Or pipe paths in
ls *.bam | majec_calc_frag_len -o frag_stats/
```

- `-n` sets the number of pairs sampled per BAM (default 1,000,000).
- `--burn_in_fraction` skips the start of each coordinate-sorted file (default 0.1).
- `-t` processes BAMs in parallel (default 4).
- `--min_frag_len` and `--max_frag_len` bound the lengths considered (defaults 50 and 1000).

### `majec_calc_thresholds`: cohort expression binning

Momentum acceleration groups transcripts into low, medium and high expression bins (see [Algorithm Details](algorithm_details.md#grouped-momentum-acceleration)). By default, each run computes the bin thresholds from its own samples and records them in `{prefix}_joint_binning_info.json`. To use the same thresholds across separate runs, precompute them from the whole cohort and pass the JSON to `--use_bins`. This tool runs featureCounts (Subread), which must be on your `PATH`.

```bash
majec_calc_thresholds \
    --bams *.bam \
    --gene_gtf gencode.v44.annotation.gtf \
    --te_gtf hg38_rmsk_TE.gtf \
    --output global_bins.json \
    --paired_end \
    --strandedness 2
```

`majec_calc_thresholds --strandedness` defaults to `2`, unlike the pipeline's `0`, so set it explicitly. `--use_bins` also accepts the `{prefix}_joint_binning_info.json` written by an earlier joint run, which carries the same `global_thresholds` key.

### `majec_add_norm_factors`: library-size normalization

`majec_add_norm_factors` adds size factors to a MAJEC database built with `majec_build_db`, which `majec_visualize` needs. See [Post-Quantification Workflows](post_quantification.md).

```bash
majec_add_norm_factors --db my_project.db
```

---

## Resources and HPC deployment

### Memory and time

Memory depends mainly on annotation size and on how many samples run in parallel.

- **Annotation size.** A gene+TE annotation (millions of loci) needs far more memory than a gene-only one (hundreds of thousands of transcripts).
- **Parallel samples.** Each parallel worker holds its own sample's data. Lowering `--threads` reduces peak memory, and runtime grows to match.

Starting points for a request:

| Analysis | Memory to request |
|---|---|
| Gene-only (GENCODE-scale) | 40–48 GB |
| Gene + TE (RepeatMasker-scale) | 64–96 GB |

- **Cache reuse.** Runs that reuse a cache skip the BAM scan and are much faster.
- **`--light`** reduces memory and time for large gene+TE runs.
- **Precompute time.** Annotation precompute takes a few minutes for gene+TE and needs modest memory.

### Single node

Give one thread per BAM:

```bash
N_BAMS=$(ls *.bam | wc -l)
majec_run_pipeline \
    --annotation ann_annotations.pkl.gz \
    --bams *.bam \
    --threads "$N_BAMS" \
    --paired_end --strandedness 2 \
    --use_cache \
    --prefix results/my_experiment
```

### SLURM: one joint job

The simplest cluster setup is one job that quantifies all samples:

```bash
#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00

majec_run_pipeline \
    --config isoform_run.json \
    --annotation /ref/ann_annotations.pkl.gz \
    --bams /data/project/*/Aligned.sortedByCoord.out.bam \
    --threads 12 \
    --prefix results/my_experiment
```

### SLURM array: one sample per task

For large cohorts, process each sample as an array task and merge afterwards:

```bash
#!/bin/bash
#SBATCH --array=1-100
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00

SAMPLE_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
majec_run_pipeline \
    --config isoform_run.json \
    --annotation /ref/ann_annotations.pkl.gz \
    --bams /data/project/sample_${SAMPLE_ID}.bam \
    --threads 1 \
    --use_bins global_bins.json \
    --use_cache \
    --cache_dir cache/sample_${SAMPLE_ID} \
    --prefix results/sample_${SAMPLE_ID}
```

- **Rerunning.** Each task is independent, so failed samples are easy to rerun.
- **Same momentum bins.** `--use_bins` gives every task the same bins (see [`majec_calc_thresholds`](#majec_calc_thresholds-cohort-expression-binning)).
- **One cache directory per task.** Every task that writes a cache also updates `cache_metadata.json` in the cache directory, so concurrent writers should not share one.
- **Merging.** Keep `--use_cache` on, so each task writes a run manifest. When all tasks finish, merge them into one database:

  ```bash
  majec_build_db --run_manifests results/sample_*_run_manifest.json --output_db my_project.db
  ```

- **Parameter sweeps.** Rerun against the existing caches with `--use_cache --read_only_cache`. Read-only jobs never write to the cache directory, so any number can share one.

---

## Migrating from 0.1.5

[CHANGELOG.md](../CHANGELOG.md) lists every change. For an existing workflow:

**Flags that no longer exist.**

- On the command line, they make the tool exit with `unrecognized arguments`.
- In a JSON `--config`, they are silently ignored.

| 0.1.5 flag | What to do |
|---|---|
| `majec_run_pipeline --use_subset_coverage_data` | Remove it. Coverage-based subset rescue has been replaced by exon containment, which is applied when EqCs are built. `--use_subset_penalty` and its parameters are unchanged. |
| `majec_run_pipeline --use_paired_rescue`, `--no_paired_rescue`, `--paired_rescue_decay` | Remove them. Paired-junction rescue (on by default in 0.1.5) has been replaced: junction evidence is now gathered during the BAM scan, and mates are intersected when EqCs are built. |
| `majec_precompute_annotations --generate_rescue_features`, `--genome_file_path` | Remove them. `{output}_subset_coverage_features.bed` is no longer written. |

**Other steps.**

- **Rebuild annotations.** Re-run `majec_precompute_annotations` with 0.2.0.
- **Rebuild caches.** 0.1.x caches are not reused. The first run with `--use_cache` rebuilds them. Don't add `--read_only_cache` to that run, or the new caches won't be saved.
- **External tools.** `majec_run_pipeline` no longer needs featureCounts, samtools or bedtools. `majec_calc_thresholds` still uses featureCounts, and `majec_calc_frag_len` still uses samtools.
- **Counting unit.** Paired-end counts are now fragments. In 0.1.5, each mate of a uniquely mapped pair counted separately, so unique counts on paired-end data are about half of 0.1.5's. Single-end data is unaffected. Quantify all samples of one analysis with the same version.
- **TPM.** On paired-end data, TPM now uses the measured fragment-length distribution by default. To reproduce a 0.1.5 TPM run that used `--frag_stats_dir`, keep passing the same directory.
- **Sample names.** BAMs named only `Aligned.sortedByCoord.out.bam` are now named after their parent directory; in 0.1.5 they were all named `Aligned`. Update any `majec_build_db --metadata_file` accordingly.
- **Removed outputs.**
  - `_prior_adjustments.tsv.gz` no longer has the `territory_evidence_ratio` and `territory_confidence` columns.
  - `majec_build_db` no longer creates territory tables.
  - `majec_visualize` no longer draws territory coverage panels.
- **Threads.** Spare threads no longer speed up a single sample (see [Threads and memory](#threads-and-memory)). Size `--threads` to the number of BAMs.

**Unchanged:**

- output file names;
- TE subfamily IDs;
- `--te_coordinates`;
- the defaults of every option present in both versions.

---

## Parameter reference

### `majec_run_pipeline`

This table lists every option shown by `majec_run_pipeline --help` in 0.2.0.

**General**

| Option | Default | Description |
|---|---|---|
| `-h`, `--help` | | Show help and exit. |
| `--version` | | Print the MAJEC version and exit. |

**Required input/output**

| Option | Default | Description |
|---|---|---|
| `--annotation FILE` | required | Annotation file (`*_annotations.pkl.gz`) from `majec_precompute_annotations`. |
| `--bams BAM [BAM ...]` | required | Coordinate-sorted input BAMs. |
| `--prefix PREFIX` | `MAJEC_output` | Prefix for every output file. May include a directory. |

**Core settings**

| Option | Default | Description |
|---|---|---|
| `--strandedness {0,1,2}` | `0` | 0 = unstranded, 1 = stranded, 2 = reverse-stranded. |
| `--threads N` | `8` | Number of samples processed in parallel, capped at the number of BAMs. |
| `--paired_end` | off | Treat data as paired-end, and count fragments. |
| `--tempdir DIR` | system temp | Directory for temporary files. |
| `--use_bins FILE` | | Expression-bin thresholds JSON from `majec_calc_thresholds`, or an earlier run's `_joint_binning_info.json`, used instead of computing them from this run's samples. |
| `--config FILE` | | JSON configuration file. Command-line options override it. |

**EM**

| Option | Default | Description |
|---|---|---|
| `--em_iterations N` | `150` | Maximum EM iterations. |
| `--momentum_start N` | `4` | Number of standard EM iterations before the momentum phase. Higher values let the data move estimates further from the prior-informed start (toward the winner-take-all maximum-likelihood solution); lower values keep them closer. See [algorithm_details.md](algorithm_details.md#what---momentum_start-controls). |
| `--momentum_scaling LOW MEDIUM HIGH` | `1.5 1.0 0.7` | Momentum scaling factors for the low, medium and high expression groups. |

**Aligners without NH: minimap2 / long reads (experimental; need majec-core)**

| Option | Default | Description |
|---|---|---|
| `--derive_multiplicity {auto,on,off}` | `auto` | Derive multiplicity from read names instead of `NH`. `auto` switches on only if none of the first 10,000 mapped records carries `NH`. |
| `--lr_score_ratio R` | `0` | Keep only alignments scoring within this fraction of the read's best `AS`; 0 keeps every reported alignment. |
| `--lr_min_mapq Q` | off | MAPQ floor for reads that resolve as unique in derive mode. |

**Evidence priors and penalties**

| Option | Default | Description |
|---|---|---|
| `--junction_weight W` | `3.0` | Weight applied to junction evidence to form the additive junction boost. 0 disables boosts. |
| `--junction_decay_exponent E` | `1.0` | Exponent that down-weights junctions shared by several transcripts. |
| `--use_tsl_penalty` | off | Penalize transcripts by Transcript Support Level. Skipped with a warning if the annotation has no TSL. |
| `--tsl_penalty_values JSON` | see note | Override TSL penalty factors as a JSON string, for example `'{"1":1.0,"2":0.8}'`. Defaults: TSL1 1.0, TSL2 0.9, TSL3 0.7, TSL4 0.5, TSL5 0.3, NA 0.8. |
| `--use_junction_completeness` | off | Penalize transcripts whose junctions are unevenly or incompletely observed. |
| `--junction_completeness_min_score S` | `0.0001` | Floor for the completeness penalty, so no transcript is zeroed. |
| `--junction_completeness_overdispersion D` | `0.05` | Overdispersion of the junction-count variance model. |
| `--library_type {dT,polyA,random,none}` | unset | Library preparation; tells `--terminal_relax` which end to relax. Required with `--terminal_relax`. |
| `--terminal_relax` | off | Relax the completeness penalty at the transcript end where the library type predicts coverage drop-off. |
| `--terminal_recovery_rate R` | `2` | Steepness of the coverage-recovery curve for terminal relaxation; higher is more lenient. |
| `--use_subset_penalty` | off | Penalize subset isoforms (whose junctions are contained in another isoform's) based on superset evidence. |
| `--subset_evidence_threshold F` | `1.25` | Factor on the evidence a subset isoform needs to avoid the penalty. |
| `--subset_penalty_min P` | `0.001` | Minimum (strongest) subset penalty factor. |
| `--subset_penalty_overdispersion D` | `0.1` | Overdispersion of the subset-penalty variance model. |

**Length scaling and TPM**

| Option | Default | Description |
|---|---|---|
| `--output_tpm` | off | Write TPM matrices. Effective lengths use the measured fragment-length distribution on paired-end data. |
| `--frag_stats_dir DIR` | | Override: per-sample fragment statistics from `majec_calc_frag_len` (truncated normal). For reproducing runs made before 0.2.0. |
| `--mean_fragment_length N` | | Override: one mean fragment length for all samples. Useful for single-end data. Mutually exclusive with `--frag_stats_dir`. |

**Confidence and output**

| Option | Default | Description |
|---|---|---|
| `--output_confidence` | off | Compute per-transcript confidence metrics. |
| `--confidence_min_expression C` | `1.0` | Minimum count for a transcript to get confidence metrics, and for a transcript to count as expressed in group metrics. |
| `--analyze_competition` | off | Summarize competition types across samples. Requires `--output_confidence`. |
| `--calculate_group_confidence` | off | Aggregate confidence metrics to genes and TE subfamilies. Requires `--output_confidence`. |
| `--verbose_output` | off | Write detailed per-sample reports (prior adjustments, dense confidence tables, distinguishability details) and unique/multimapper count matrices. |
| `--dump_em_classes` | off | Save each sample's EqCs to a pickle, for debugging. |
| `--te_coordinates` | off | Write `{prefix}_TE_locus_coordinates.tsv`. |
| `--light` | off | Skip prior tracking, confidence metrics and verbose outputs to save memory and time. |

**Caching**

| Option | Default | Description |
|---|---|---|
| `--use_cache` | off | Use a valid cached scan if available; otherwise scan and save. |
| `--cache_dir DIR` | `pipeline_cache` | Cache directory. |
| `--rebuild_cache` | off | Rescan and overwrite caches even if valid. |
| `--read_only_cache` | off | Read caches but never write them. Use with `--use_cache`. |

### `majec_precompute_annotations`

| Option | Default | Description |
|---|---|---|
| `--gene_gtf FILE` | required | Gene annotation GTF (uncompressed). |
| `--te_gtf FILE` | | TE annotation GTF (optional). |
| `--output PREFIX` | `majec` | Output prefix. Writes `{PREFIX}_annotations.pkl.gz` and `{PREFIX}_annotations_summary.json`. |
| `--validate FILE` | | Check an existing annotation file against the GTFs given with `--gene_gtf`/`--te_gtf`. Exits 0 if they match, 1 if not. |
