# Changelog

All notable changes to MAJEC are listed here. Version numbers follow the `majec` package on PyPI.

## 0.2.0

0.2.0 builds on 0.1.5 with a new way of forming equivalence classes. The EM, the prior models and the output layout carry over. Counts and TPM from the two versions are not directly comparable (see [Changed behaviour](#changed-behaviour)), so quantify all samples of one analysis with the same version. The [0.2.0 white paper](docs/whitepaper.md) compares the two versions on spike-in, long-read and TE benchmarks.

### Highlights

- **Single-pass, junction-aware equivalence classes (EqCs).** MAJEC reads each BAM once. Each fragment's candidate transcripts are the intersection of exon overlap, annotated-junction compatibility, and exon containment; the two mates' candidate sets are intersected, and every fragment is counted once. This replaces the three-pass featureCounts flow used in 0.1.x.
- **Two builders with the same results.**
  - The compiled builder (majec-core) is a separate Rust package, installed with `pip install "majec[rust]"`.
  - The pure-Python builder (pysam) is always installed and is roughly 6× slower on the BAM scan.
  - The pipeline log and the run manifest record which builder ran.
- **Measured fragment-length distribution.** On paired-end data, fragment lengths are measured during the BAM scan, in transcript coordinates, and used for TPM effective lengths. No separate prepass is needed.
- **No external tools in the quantification path.** `majec_run_pipeline` no longer calls featureCounts, samtools, or bedtools.
- **Portable annotations.** Annotation files built by `majec_precompute_annotations` no longer need the source GTFs at their original paths.
- **Experimental: aligners without NH tags.** BAMs from aligners such as minimap2 can be quantified, with read multiplicity derived from read names.

### Changed behaviour

These changes apply without any change to your command line.

- **Counting unit for paired-end data.** Each fragment is counted once. In 0.1.5, each mate of a uniquely mapped pair was counted separately, while multimapping pairs were counted per fragment. Unique counts on paired-end data are therefore about half of 0.1.5's, and the balance between unique and multimapping counts shifts accordingly. Single-end data is unaffected.
- **TPM effective lengths.** With `--output_tpm` on paired-end data, effective lengths now come from the measured fragment-length distribution by default. In 0.1.5 they came from annotated transcript lengths unless `--frag_stats_dir` was given.
  - `--frag_stats_dir` (from `majec_calc_frag_len`) still works and takes precedence when given. `majec_calc_frag_len` estimates from genomic template length (TLEN), while the in-scan measurement works in transcript coordinates, so the two can differ on spliced libraries. The pipeline logs a warning when the override is used.
  - The truncated-normal model used with `--frag_stats_dir` is now renormalised by P(fragment length ≤ transcript length). With the same fragment statistics, transcripts too short to hold every modelled fragment length get larger effective lengths, and so lower TPM, than in 0.1.5. The difference is large only for transcripts shorter than about the mean fragment length.
  - Single-end data falls back to annotated lengths with a warning. Pass `--mean_fragment_length` to use a fragment-length model.
  - Caveats for interpreting TPM: on 3′-biased polyA libraries, long transcripts receive fewer reads than their length implies, so any length correction (MAJEC's or another tool's) over-corrects them. For TE loci, use counts rather than TPM; TE loci are genomic intervals, and very short loci get inflated TPM.
- **Junction compatibility and exon containment act when classes are built.** A spliced fragment is assigned only to transcripts that contain its annotated junctions. A read that overhangs an exon boundary drops the transcripts whose exons it does not fit. If no transcript passes every test, the builder falls back to the less strict candidate sets. In 0.1.5, featureCounts assigned reads by exon overlap, and junction evidence entered through the priors.
- **Effect on priors.** The recommended prior flags are unchanged. Because junction evidence now acts directly when classes are built, the junction boost (`--junction_weight`) has much less influence than in 0.1.5, and subset penalties somewhat less. Completeness penalties remain important.
- **Junction counts for the priors are tallied in the BAM scan.** As in 0.1.5 (featureCounts `-J`), every alignment that crosses an annotated junction adds one, including multimapping and secondary alignments, and the two mates of a pair are counted separately. New in 0.2.0: an alignment counts only if it overlaps an annotated exon on the strand inferred from `--strandedness`; featureCounts counted junctions regardless of strand.
- **Unique/multimapper split.** The split is read from the NH tag. Alignments with NH = 1 are unique and are kept if MAPQ ≥ 30; alignments with NH > 1 are multimappers. For STAR output this matches 0.1.5.
- **Sample names.** A BAM whose file name reduces to nothing informative is now named after its parent directory. This applies to STAR's default `Aligned.sortedByCoord.out.bam`, and to names that reduce to `output` or `results`. In 0.1.5 such files were named `Aligned`. Names such as `sample1_Aligned.sortedByCoord.out.bam` still give `sample1`.
- **Threading.** The per-sample BAM scan is single-threaded with either builder. `--threads` sets how many samples are processed in parallel, so a single-sample run uses one core for the scan. In 0.1.5, spare threads were passed to featureCounts and samtools within each sample.

### Added

- `majec_run_pipeline --version`.
- The `rust` extra (`pip install "majec[rust]"`), which installs majec-core.
- Builder provenance:
  - The log reports `EqC builder: Rust majec_core <version> (sha256 <hash prefix>)`, or warns that the pure-Python pysam builder is in use.
  - The run manifest has new `majec_version` and `eqc_builder` fields. For majec-core, `eqc_builder` records the version, the path of the loaded binary, and its SHA-256; for the pure-Python builder, it records the pysam version.
  - The run manifest is now written on every run; in 0.1.5 it was written only when a cache was used. Without caching, its `cache_files` entry is empty.
- The measured fragment-length histogram is cached per sample with `--use_cache`. The log reports the number of pairs and the mean and SD used for effective lengths.
- **Experimental** option group for aligners without NH tags (minimap2 / long reads). Turning them on needs majec-core; with the pure-Python builder the pipeline stops with an error rather than ignoring them (`--derive_multiplicity auto` then behaves as `off`). Long-read quantification has been checked for correctness on ONT libraries, but its accuracy against dedicated long-read quantifiers has not been established.
  - `--derive_multiplicity {auto,on,off}` (default `auto`): derive multiplicity from read names instead of NH. `auto` switches on only when none of the first 10,000 mapped records carries NH, so short-read runs are unaffected.
  - `--lr_score_ratio`: keep only alignments scoring within this fraction of the read's best AS (default 0, keep all).
  - `--lr_min_mapq`: MAPQ floor for reads that resolve as unique in derive mode (default off).
  - These options join the cache key when set away from their defaults, so changing them builds a separate cache.

### Removed

| Removed | Notes |
|---|---|
| `majec_run_pipeline --use_subset_coverage_data` | Coverage-based ("territory") subset rescue. Subset/superset discrimination is now handled when classes are built, by exon containment and junction compatibility. `--use_subset_penalty` and its parameters are unchanged. |
| `majec_run_pipeline --use_paired_rescue`, `--no_paired_rescue`, `--paired_rescue_decay` | Paired-junction rescue (on by default in 0.1.5). Junction evidence is now gathered directly during the BAM scan, and mates are intersected when classes are built. |
| `majec_precompute_annotations --generate_rescue_features`, `--genome_file_path` | Only needed for coverage-based subset rescue. |
| `{output}_subset_coverage_features.bed` | Was written by `--generate_rescue_features`. |
| featureCounts, samtools and bedtools in `majec_run_pipeline` | Replaced by the built-in EqC builders. featureCounts is still used by `majec_calc_thresholds`, and samtools by `majec_calc_frag_len`. |

### Fixed

- `majec_calc_thresholds` could not run: its featureCounts command lacked a space before the BAM path. Paths are now quoted in `majec_calc_thresholds` and `majec_calc_frag_len`, so paths with spaces work.
- `majec_calc_thresholds` now counts paired-end data as fragments (`--countReadPairs`), the unit `majec_run_pipeline` uses. This needs Subread ≥ 2.0.2.
- `majec_prepare_deseq2 --confidence_mode variance` wrote an R script that stopped before producing results.
- `majec_prepare_deseq2 --variance_base` failed when given a value.
- `majec_prepare_deseq2` now rounds fractional counts to the nearest integer for DESeq2; it previously truncated them.
- `majec_build_db` rejected metadata files that list `sample_id` under `#id_variables:`.
- `majec_build_db` skipped the count tables for runs without `--output_tpm`. It now loads the counts and leaves `tpm` empty.
- `--analyze_competition` without `--output_confidence` stopped the run after the counts were written. It is now switched off with a warning.
- `{prefix}_prior_adjustment_summary.tsv` gained rows on every rerun with the same prefix. It is now rewritten each run.
- `pip install "majec[viz]"` and `pip install "majec[full_analysis]"` failed to resolve: the extras required `python-kaleido`, the conda-forge name. They now require `kaleido`, its name on PyPI.

### Compatibility and migration

- **Re-run `majec_precompute_annotations` with 0.2.0.** Annotation files from 0.1.x lack the precomputed exon index the builders use. They still load, but the pipeline then rebuilds the index by re-reading the GTFs from the paths recorded in the file. That is slow and needs those GTFs to still exist.
- **Annotation files are portable.** Files built by 0.2.0 contain everything quantification needs, so the GTFs can be moved or deleted afterwards.
- **Caches from 0.1.x are not reused.** The cache format is now 2.0, and 0.1.x caches do not match it. The first 0.2.0 run with `--use_cache` rebuilds them; run it without `--read_only_cache` so the new caches are saved. Old cache directories can be deleted.
- **Removed flags.**
  - On the command line, removed flags make `majec_run_pipeline` or `majec_precompute_annotations` exit with an "unrecognized arguments" error.
  - In a JSON `--config` file, the same keys are silently ignored. Remove `use_subset_coverage_data`, `use_paired_rescue`, `no_paired_rescue` and `paired_rescue_decay` from saved configs so that a config does not appear to enable something it no longer does.
- **Output and table changes.**
  - `{prefix}_{sample}_prior_adjustments.tsv.gz` no longer has the `territory_evidence_ratio` and `territory_confidence` columns.
  - `majec_build_db` no longer creates the `territory_regions`, `transcript_territories` and `territory_coverage` tables.
  - `majec_visualize` no longer draws territory coverage panels.
  - `_subset_coverage_features.bed` is no longer produced.
- **Sample names** may change for BAMs named only `Aligned.sortedByCoord.out.bam` (see [Changed behaviour](#changed-behaviour)). Update any metadata file passed to `majec_build_db --metadata_file` to match.
- **Unchanged since 0.1.5:**
  - TE subfamily IDs (`name:family:class`);
  - `--te_coordinates` and `{prefix}_TE_locus_coordinates.tsv`;
  - output file names;
  - the defaults of every `majec_run_pipeline` and `majec_precompute_annotations` option present in both versions.

### Dependencies

- **Python** ≥ 3.11 (unchanged).
- **Core:** numpy, pandas, scipy, pysam. scipy was previously in the `tools` extra; pysam is new and provides the pure-Python builder. intervaltree is no longer used.
- **Extras:**
  - `rust`: majec-core (new). Prebuilt wheels; on platforms without one, pip builds from source, which needs a Rust toolchain.
    majec-core is built against PyO3 0.29, which includes fixes for two PyO3 security advisories (GHSA-36hh-v3qg-5jq4, GHSA-chgr-c6px-7xpp); majec-core does not use the affected APIs.
  - `database`: polars.
  - `viz`: plotly, python-kaleido, pypdf2, xlsxwriter.
  - `full_analysis`: everything in `database` and `viz`.
  - `tools`: now empty; kept so existing install commands still resolve.
- **External tools:**
  - bedtools is no longer needed.
  - samtools is needed only by `majec_calc_frag_len`, and featureCounts (Subread ≥ 2.0.2) only by `majec_calc_thresholds`.
  - `majec_build_db` calls the `sqlite3` command-line program.
- **Conda:**
  - `majec.yml` drops bedtools and intervaltree and adds pysam.
  - The bioconda package is `noarch` and uses the pure-Python builder; add `pip install majec-core` in the same environment for the compiled builder.

See [docs/installation.md](docs/installation.md).

## 0.1.5

The release described in Lim & Firestone (2026); the citation is in the [README](README.md#citation). For details, see the [0.1.5 release on PyPI](https://pypi.org/project/majec/0.1.5/) and the repository history.
