//! Single-pass BAM processing to build junction-aware equivalence classes.
//!
//! This is the hot loop — the main performance target for the Rust rewrite.
//! Replaces eqc_builder.process_bam() in Python.
//! Uses noodles (pure-Rust BAM parser) — no C library dependencies.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use std::fs::File;
use std::time::Instant;

use noodles::bam;
use noodles::sam;
use noodles::sam::alignment::record::cigar::op::Kind;
use noodles::sam::alignment::record::Cigar as CigarTrait;
use noodles::sam::alignment::record::data::field::Tag;

use crate::exon_index::ExonIntervalIndex;

// Coordinate offsets calibrated on LongBench data
const DONOR_OFFSET: i64 = 0;
const ACCEPTOR_OFFSET: i64 = 1;

/// Upper length for the *capped* fragment-length summary. Salmon's own fld tops
/// out here, so the capped numbers are the like-for-like comparison against it.
/// The histogram itself is never truncated.
const FRAG_CAP: u32 = 1000;

/// Extract aligned blocks and splice junctions simultaneously (single CIGAR pass).
///
/// Returns (blocks, junctions) where:
///   blocks: Vec<(start_1based, end_1based)>
///   junctions: Vec<(donor, acceptor)> as numeric tuples (no String allocation)
fn parse_cigar(
    cigar: &sam::alignment::record_buf::Cigar,
    ref_start_0based: i64,
) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let mut blocks = Vec::new();
    let mut junctions = Vec::new();
    let mut pos = ref_start_0based;

    for op in cigar.as_ref().iter() {
        let len = op.len();
        match op.kind() {
            Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                // M, =, X — aligned to reference
                blocks.push(((pos + 1) as u32, (pos + len as i64) as u32));
                pos += len as i64;
            }
            Kind::Deletion => {
                // D — consumes reference
                pos += len as i64;
            }
            Kind::Skip => {
                // N — splice junction
                let donor = (pos + DONOR_OFFSET) as u32;
                let acceptor = (pos + len as i64 + ACCEPTOR_OFFSET) as u32;
                junctions.push((donor, acceptor));
                pos += len as i64;
            }
            // Insertion, SoftClip, HardClip, Pad — don't consume reference
            _ => {}
        }
    }

    (blocks, junctions)
}

/// Infer transcript strand from read flags and library strandedness.
/// Returns: b'+', b'-', or 0 (unstranded)
fn infer_read_strand(is_read2: bool, is_reverse: bool, strandedness: u8) -> u8 {
    if strandedness == 0 {
        return 0;
    }
    let is_read1 = !is_read2;
    match strandedness {
        1 => {
            if is_read1 {
                if is_reverse { b'-' } else { b'+' }
            } else {
                if is_reverse { b'+' } else { b'-' }
            }
        }
        2 => {
            if is_read1 {
                if is_reverse { b'+' } else { b'-' }
            } else {
                if is_reverse { b'-' } else { b'+' }
            }
        }
        _ => 0,
    }
}

/// Does this BAM carry NH tags?
///
/// Every multiplicity decision in the scan below hangs on NH, and the old code
/// read it as `.unwrap_or(1)` — so an aligner that simply does not write the tag
/// was indistinguishable from one reporting "this read mapped once". STAR and
/// HISAT2 always write NH; minimap2 never does.
///
/// Probes the first `n_records` mapped records and reports whether *any* of them
/// carries the tag. Erring towards "present" is deliberate: only a BAM with no NH
/// anywhere in the probe switches modes, so short-read behaviour is untouched.
/// Costs one extra file handle and ~10K records — a few milliseconds.
fn bam_has_nh_tag(bam_path: &str, n_records: usize) -> PyResult<(bool, usize)> {
    let file = File::open(bam_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to open BAM: {}", e)))?;
    let mut reader = bam::io::Reader::new(file);
    let header = reader.read_header()
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(
            format!("Failed to read BAM header: {}", e)))?;

    let nh_tag = Tag::new(b'N', b'H');
    let mut seen = 0usize;
    for result in reader.record_bufs(&header) {
        let record = result
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("BAM read error: {}", e)))?;
        if record.flags().is_unmapped() {
            continue;
        }
        if record.data().get(&nh_tag).is_some() {
            return Ok((true, seen + 1));
        }
        seen += 1;
        if seen >= n_records {
            break;
        }
    }
    Ok((false, seen))
}

/// One alignment of one read, held until end-of-scan when multiplicity has to be
/// derived from the read name.
///
/// Features are a sorted `Vec<u32>`, not an `FxHashSet`, because in this mode
/// *every* read is buffered — including the ones that turn out to be unique — so
/// the per-alignment constant is what sets peak memory. An empty hash set costs
/// ~48 bytes before it allocates; a Vec costs 24.
struct LrAlignment {
    features: Vec<u32>,
    score: i64,
    mapq: u8,
}

/// Stats counters for the BAM scan.
struct BamStats {
    total_reads: u64,
    unique_mapped: u64,
    multimapped: u64,
    unmapped: u64,
    low_mapq_unique: u64,
    no_feature: u64,
    spliced_reads: u64,
    junction_refined: u64,
    junction_novel: u64,
    junction_incompatible: u64,
    exon_refined: u64,
    exon_incompatible: u64,
    unique_eqc_size_1: u64,
    paired_reads_processed: u64,
    paired_intersection_empty: u64,
    orphaned_mates: u64,
    // --- derive-multiplicity (no NH tag) mode ---
    /// True when the BAM carried no NH tag and multiplicity was derived by name.
    multiplicity_derived: bool,
    /// Distinct read names buffered.
    lr_reads_buffered: u64,
    /// Reads that resolved to exactly one surviving alignment.
    lr_reads_unique: u64,
    /// Reads that resolved to more than one surviving alignment.
    lr_reads_multi: u64,
    /// Alignments discarded by the AS score-ratio filter before classification.
    lr_alignments_dropped: u64,
    /// Secondary alignments landing outside annotation. Kept apart from
    /// `no_feature` so that counter stays per-read and comparable to a run that
    /// read NH.
    lr_secondary_no_feature: u64,
    /// Multireads whose every surviving alignment is below `min_mapq`. These are
    /// exactly the reads the NH-assuming path deleted as `low_mapq_unique`; here
    /// they reach the EM instead, which is the larger of the two effects.
    lr_multi_rescued_low_mapq: u64,
    // Read length tracking (sampled from first 10K reads)
    read_length_samples: Vec<u32>,
    read_length_sampling_done: bool,
    /// Empirical fragment length distribution: cDNA length -> weight, where a
    /// pair compatible with k transcripts contributes 1/k to each.
    ///
    /// A histogram rather than a running (mean, sd): the real distribution is
    /// right-skewed, so summarising it as a Gaussian discards the tail that
    /// matters for effective length on short transcripts. Every pair in the
    /// BAM is seen, so this is a census, not a sample, and no truncation
    /// window is applied here — trimming is the consumer's decision.
    fragment_length_hist: FxHashMap<u32, f64>,
    /// Diagnostic: restricted to pairs compatible with exactly one transcript.
    fragment_length_hist_unique: FxHashMap<u32, u64>,
    /// Diagnostic: unspliced genomic span of those same unique pairs (= TLEN).
    genomic_span_hist: FxHashMap<u32, u64>,
    /// Unique pairs whose span could not be placed in the transcript's exons
    /// (an endpoint in an intron or past the annotated ends). These are exactly
    /// the pairs whose TLEN would be intron-inflated on a genome BAM.
    fragment_span_not_in_exons: u64,
}

impl BamStats {
    fn new() -> Self {
        BamStats {
            total_reads: 0, unique_mapped: 0, multimapped: 0, unmapped: 0,
            low_mapq_unique: 0, no_feature: 0, spliced_reads: 0,
            junction_refined: 0, junction_novel: 0, junction_incompatible: 0,
            exon_refined: 0, exon_incompatible: 0,
            unique_eqc_size_1: 0,
            paired_reads_processed: 0, paired_intersection_empty: 0,
            orphaned_mates: 0,
            multiplicity_derived: false,
            lr_reads_buffered: 0, lr_reads_unique: 0, lr_reads_multi: 0,
            lr_alignments_dropped: 0, lr_multi_rescued_low_mapq: 0,
            lr_secondary_no_feature: 0,
            read_length_samples: Vec::with_capacity(10_000),
            read_length_sampling_done: false,
            fragment_length_hist: FxHashMap::default(),
            fragment_length_hist_unique: FxHashMap::default(),
            genomic_span_hist: FxHashMap::default(),
            fragment_span_not_in_exons: 0,
        }
    }

    fn to_pydict<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        d.set_item("total_reads", self.total_reads).unwrap();
        d.set_item("unique_mapped", self.unique_mapped).unwrap();
        d.set_item("multimapped", self.multimapped).unwrap();
        d.set_item("unmapped", self.unmapped).unwrap();
        d.set_item("low_mapq_unique", self.low_mapq_unique).unwrap();
        d.set_item("no_feature", self.no_feature).unwrap();
        d.set_item("spliced_reads", self.spliced_reads).unwrap();
        d.set_item("junction_refined", self.junction_refined).unwrap();
        d.set_item("junction_novel", self.junction_novel).unwrap();
        d.set_item("junction_incompatible", self.junction_incompatible).unwrap();
        d.set_item("exon_refined", self.exon_refined).unwrap();
        d.set_item("exon_incompatible", self.exon_incompatible).unwrap();
        d.set_item("unique_eqc_size_1", self.unique_eqc_size_1).unwrap();
        d.set_item("paired_reads_processed", self.paired_reads_processed).unwrap();
        d.set_item("paired_intersection_empty", self.paired_intersection_empty).unwrap();
        d.set_item("orphaned_mates", self.orphaned_mates).unwrap();
        d.set_item("multiplicity_derived", self.multiplicity_derived).unwrap();
        if self.multiplicity_derived {
            d.set_item("lr_reads_buffered", self.lr_reads_buffered).unwrap();
            d.set_item("lr_reads_unique", self.lr_reads_unique).unwrap();
            d.set_item("lr_reads_multi", self.lr_reads_multi).unwrap();
            d.set_item("lr_alignments_dropped", self.lr_alignments_dropped).unwrap();
            d.set_item("lr_multi_rescued_low_mapq", self.lr_multi_rescued_low_mapq).unwrap();
            d.set_item("lr_secondary_no_feature", self.lr_secondary_no_feature).unwrap();
        }
        // Median read length from sampled reads
        if !self.read_length_samples.is_empty() {
            let mut sorted = self.read_length_samples.clone();
            sorted.sort_unstable();
            let median = sorted[sorted.len() / 2];
            d.set_item("median_read_length", median).unwrap();
        }
        // Fragment length distribution.
        //
        // `fragment_length_hist` is the ambiguity-weighted estimator, validated
        // against Salmon and RSEM (see docs/algorithm_details.md). The
        // histogram is handed back untruncated; summary stats are computed both
        // over all of it and over [0, FRAG_CAP], because Salmon's own fld tops
        // out at 1000 and the capped numbers are the like-for-like comparison.
        // Mean/sd are computed here so nothing downstream needs the full map.
        if !self.fragment_length_hist.is_empty() {
            let h = PyDict::new(py);
            for (&len, &w) in &self.fragment_length_hist {
                h.set_item(len, w).unwrap();
            }
            d.set_item("fragment_length_hist", h).unwrap();
            summarize_hist(py, &d, "fragment_length", &self.fragment_length_hist, None);
            summarize_hist(py, &d, "fragment_length_capped", &self.fragment_length_hist,
                           Some(FRAG_CAP));
        }

        // Diagnostics. Neither is meant to be consumed downstream; they exist so
        // the estimator can be re-audited from a normal run without a special
        // build. `_unique` is the same measurement restricted to pairs that pin
        // down exactly one transcript -- a length-dependent filter, since longer
        // fragments span more structure and so become uniquely assignable more
        // often, which biases the mean up ~8%. `genomic_span` is those same
        // pairs measured without removing introns, i.e. what TLEN reports.
        if !self.fragment_length_hist_unique.is_empty() {
            let u: FxHashMap<u32, f64> = self.fragment_length_hist_unique.iter()
                .map(|(&k, &v)| (k, v as f64)).collect();
            summarize_hist(py, &d, "fragment_length_unique", &u, None);
        }
        if !self.genomic_span_hist.is_empty() {
            let g: FxHashMap<u32, f64> = self.genomic_span_hist.iter()
                .map(|(&k, &v)| (k, v as f64)).collect();
            summarize_hist(py, &d, "genomic_span", &g, None);
        }
        d.set_item("fragment_span_not_in_exons", self.fragment_span_not_in_exons).unwrap();
        d
    }
}

/// Write `{prefix}_n/_mean/_sd/_median/_p10/_p90/_p99` for a weighted histogram,
/// optionally restricted to lengths <= `cap`.
fn summarize_hist(
    py: Python<'_>,
    d: &Bound<'_, PyDict>,
    prefix: &str,
    hist: &FxHashMap<u32, f64>,
    cap: Option<u32>,
) {
    let keep = |len: u32| cap.map_or(true, |c| len <= c);

    let mut n = 0.0f64;
    let mut sum = 0.0f64;
    for (&len, &w) in hist {
        if keep(len) {
            n += w;
            sum += len as f64 * w;
        }
    }
    if n <= 0.0 {
        return;
    }
    let mean = sum / n;
    let var: f64 = hist.iter()
        .filter(|(&len, _)| keep(len))
        .map(|(&len, &w)| { let e = len as f64 - mean; e * e * w })
        .sum::<f64>() / n;

    let mut lengths: Vec<u32> = hist.keys().copied().filter(|&l| keep(l)).collect();
    lengths.sort_unstable();
    let quantile = |q: f64| -> u32 {
        let target = q * n;
        let mut cum = 0.0f64;
        for &len in &lengths {
            cum += hist[&len];
            if cum >= target {
                return len;
            }
        }
        *lengths.last().unwrap()
    };

    let _ = py;
    d.set_item(format!("{prefix}_n"), n).unwrap();
    d.set_item(format!("{prefix}_mean"), mean).unwrap();
    d.set_item(format!("{prefix}_sd"), var.sqrt()).unwrap();
    d.set_item(format!("{prefix}_median"), quantile(0.50)).unwrap();
    d.set_item(format!("{prefix}_p10"), quantile(0.10)).unwrap();
    d.set_item(format!("{prefix}_p90"), quantile(0.90)).unwrap();
    d.set_item(format!("{prefix}_p99"), quantile(0.99)).unwrap();
}

/// Accumulate a feature set into unique EqC counters.
fn accumulate_unique(
    features: &FxHashSet<u32>,
    unique_eqc: &mut FxHashMap<Vec<u32>, u32>,
    unique_counts: &mut FxHashMap<u32, f64>,
    stats: &mut BamStats,
) {
    if features.len() == 1 {
        // Fast path: skip collect+sort for the common single-transcript case
        let f = *features.iter().next().unwrap();
        *unique_eqc.entry(vec![f]).or_insert(0) += 1;
        *unique_counts.entry(f).or_insert(0.0) += 1.0;
        stats.unique_eqc_size_1 += 1;
    } else {
        let mut key: Vec<u32> = features.iter().copied().collect();
        key.sort_unstable();
        *unique_eqc.entry(key).or_insert(0) += 1;

        let fraction = 1.0 / features.len() as f64;
        for &f in features {
            *unique_counts.entry(f).or_insert(0.0) += fraction;
        }
    }
    stats.unique_mapped += 1;
}

/// Single-pass BAM processing to build junction-aware equivalence classes.
///
/// Args:
///     bam_path: Path to coordinate-sorted, indexed BAM file
///     exon_index: ExonIntervalIndex built from annotation
///     junction_unstranded: dict of {"chr:donor-acceptor": [tx_int, ...]}
///     strandedness: 0=unstranded, 1=stranded, 2=reverse-stranded (default)
///     paired_end: Whether data is paired-end
///     min_mapq: Minimum MAPQ for unique mappers
///     max_reads: Optional cap on reads to process (for testing)
///     derive_multiplicity: None = auto-detect from the NH tag (default),
///         True/False = force. When on, a read's multiplicity is derived from
///         how many alignments carry its name rather than from NH, which is what
///         a long-read (minimap2) BAM requires.
///     lr_score_ratio: In derive mode, keep only alignments scoring at least this
///         fraction of their read's best AS. <= 0 disables the filter, which is
///         the default, and that default is measured rather than inherited.
///
///         This knob controls DECISIVENESS, not stringency, and the intuition
///         runs backwards: a tight cutoff keeps only the top-scoring alignment,
///         so a read is handed to whichever locus wins by a point and is then
///         recorded as UNIQUE to it. On ONT that point is noise. FTL (chr19,
///         13,763 TPM in matched A549 Illumina) and its processed pseudogene
///         FTLP3 (chr20, 527bp, 570 TPM) differ by less than a 1.2kb read's ~60
///         sequencing errors, so at 0.95 the filter moved 18,466 reads onto
///         FTLP3 as uniques: FTLP3 went to 21x its Illumina value while FTL LOST
///         3.7x of its own expression. At ratio 0 the pair lands at 13,358 and
///         2,495 CPM. Across the 200 transcripts the filter moves most, median
///         |log2(LR/Illumina)| is 3.89 at 0.95 and 2.52 at 0.
///
///         0.95 is the NanoCount/oarfish rule, and it is right for them: they
///         quantify against a TRANSCRIPTOME, where the surviving alignments are
///         isoforms of one gene and the filter prunes a within-gene tail. Here
///         the competitors are separate genomic loci, and dropping one does not
///         prune a tail, it fabricates certainty about which locus emitted the
///         read. Keep every locus minimap2 reported and let the EM arbitrate --
///         that is what the EM is for.
///     lr_min_mapq: MAPQ floor for reads that resolve unique in derive mode.
///         Defaults to 0 — i.e. off. Measured, not assumed. With the AS filter off
///         (the default) a read only resolves unique when minimap2 reported exactly
///         one alignment for it, so a MAPQ floor here deletes reads without
///         reclassifying anything. With a filter on it is worse than useless:
///         minimap2 lowers MAPQ *because* of the alternatives the filter has just
///         judged, so the two test the same evidence and the threshold controls
///         read retention and ambiguity at once. On A549 directcDNA, sweeping the
///         ratio over [0, 1] moved the assigned-read count by 295,660 at floor 30
///         and still 220,778 at floor 1 (near-ties get MAPQ *0*, so no non-zero
///         floor helps)
///         — but by exactly 0 at floor 0, where the count pins to the number of
///         reads with at least one annotated alignment.
///
///         What is given up: reads whose alternatives minimap2 declined to report
///         below `-N`/`-p` keep a low MAPQ that nothing here can see, and are taken
///         as unique. The fix for that is at alignment time, not here — it is why
///         NanoCount and oarfish ask for `-N 10 -p 0` / `-N 100` and filter in the
///         quantifier rather than trusting the aligner's own reporting cutoff.
///
/// Returns:
///     dict with keys: unique_eqc, unique_counts, multi_eqc, multi_counts,
///                     junction_counts, stats
#[pyfunction]
#[pyo3(signature = (bam_path, exon_index, junction_unstranded, strandedness=2, paired_end=true, min_mapq=30, max_reads=None, use_alignment_scoring=false, as_temperature=10.0, measure_fragment_length=true, derive_multiplicity=None, lr_score_ratio=0.0, lr_min_mapq=None))]
pub fn process_bam(
    py: Python<'_>,
    bam_path: &str,
    exon_index: &ExonIntervalIndex,
    junction_unstranded: HashMap<String, Vec<u32>>,
    strandedness: u8,
    paired_end: bool,
    min_mapq: u8,
    max_reads: Option<u64>,
    use_alignment_scoring: bool,
    as_temperature: f64,
    measure_fragment_length: bool,
    derive_multiplicity: Option<bool>,
    lr_score_ratio: f64,
    lr_min_mapq: Option<u8>,
) -> PyResult<Py<PyAny>> {
    // Open BAM file & read header to get chromosome names
    let file = File::open(bam_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to open BAM: {}", e)))?;
    let mut reader = bam::io::Reader::new(file);
    let header = reader.read_header()
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read BAM header: {}", e)))?;

    // Build BAM ref_seq_id -> exon_index chrom_id mapping,
    // and reverse mapping chrom_id -> chrom name for final export
    let ref_seqs = header.reference_sequences();
    let chrom_names: Vec<String> = ref_seqs.keys().map(|name| name.to_string()).collect();
    let ref_to_chrom_id: Vec<Option<usize>> = chrom_names.iter()
        .map(|name| exon_index.get_chrom_id(name))
        .collect();
    let mut chrom_id_to_name: FxHashMap<usize, &str> = FxHashMap::default();
    for (ref_id, name) in chrom_names.iter().enumerate() {
        if let Some(chrom_id) = ref_to_chrom_id[ref_id] {
            chrom_id_to_name.entry(chrom_id).or_insert(name.as_str());
        }
    }

    // Pre-parse junction keys "chr:donor-acceptor" into (chrom_id, donor, acceptor) tuples.
    // Avoids per-read String allocation in the hot loop.
    let mut junc_lookup: FxHashMap<(usize, u32, u32), FxHashSet<u32>> = FxHashMap::default();
    for (k, v) in junction_unstranded {
        let parts: Vec<&str> = k.split(|c| c == ':' || c == '-').collect();
        if parts.len() == 3 {
            if let Some(&chrom_id) = exon_index.get_chrom_id(parts[0]).as_ref() {
                let donor: u32 = parts[1].parse().unwrap_or(0);
                let acceptor: u32 = parts[2].parse().unwrap_or(0);
                junc_lookup.insert((chrom_id, donor, acceptor), v.into_iter().collect());
            }
        }
    }

    // Accumulators
    let mut unique_eqc: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
    let mut unique_counts: FxHashMap<u32, f64> = FxHashMap::default();
    // When AS-weighting is off: union features per read (current behavior)
    let mut multi_read_features: FxHashMap<Vec<u8>, FxHashSet<u32>> = FxHashMap::default();
    // When AS-weighting is on: keep per-alignment (features, AS) pairs per read
    let mut multi_read_alignments: FxHashMap<Vec<u8>, Vec<(FxHashSet<u32>, i64)>> = FxHashMap::default();
    // When multiplicity is derived from the read name: every read, unique or not
    let mut lr_reads: FxHashMap<Vec<u8>, Vec<LrAlignment>> = FxHashMap::default();
    let mut junction_agg_counts: FxHashMap<(usize, u32, u32), u32> = FxHashMap::default();
    let mut stats = BamStats::new();

    // Paired-end mate buffer keyed by (read_name, HI_tag) to pair mates at correct locus.
    // Carries (features, is_unique, AS, chrom_id, span_start, span_end); the last three
    // exist so the fragment length can be measured when the mate arrives.
    let mut mate_buffer: FxHashMap<(Vec<u8>, u32), (FxHashSet<u32>, bool, i64, usize, u32, u32)> =
        FxHashMap::default();

    let mut reads_since_check: u64 = 0;
    let mut total_records: u64 = 0;
    let mut next_progress: u64 = 10_000_000;
    let scan_start = Instant::now();

    // Get Python logging.info for progress reports
    let logging = py.import("logging")?;

    // --- where does multiplicity come from? -------------------------------
    // NH, if the aligner wrote one. minimap2 does not, and reading a missing tag
    // as NH=1 made every long read a unique mapper: the `is_secondary &&
    // is_unique` guard below then discarded every secondary alignment, the
    // multimapper buffer stayed empty, and the cross-locus EM had nothing to
    // resolve. Probe instead of assuming.
    let derive_mode = match derive_multiplicity {
        Some(v) => v,
        None => {
            let (has_nh, probed) = bam_has_nh_tag(bam_path, 10_000)?;
            !has_nh && probed > 0
        }
    };
    stats.multiplicity_derived = derive_mode;

    // Long reads are single-end. Force the SE route even if the caller asked for
    // paired: the mate buffer keys on (name, HI), and an aligner that omits NH
    // omits HI too, so every secondary would collide on HI=0 and be mis-paired
    // with an alignment at an unrelated locus.
    let paired_end = paired_end && !derive_mode;

    // MAPQ floor for reads that resolve unique in derive mode. See the parameter
    // docs: with no AS filter a unique read is one minimap2 reported once, so a
    // floor only deletes; with a filter on, the two test the same evidence twice,
    // and any non-zero floor still deletes the near-ties, which carry MAPQ 0.
    let lr_mapq_floor: u8 = lr_min_mapq.unwrap_or(0);

    if derive_mode {
        let filter_desc = if lr_score_ratio > 0.0 {
            format!("Alignments scoring below {:.0}% of their read's best AS are dropped",
                    lr_score_ratio * 100.0)
        } else {
            "No AS filter (lr_score_ratio <= 0): every alignment the aligner \
             reported is taken as a real locus".to_string()
        };
        logging.call_method1("warning", (format!(
            "No NH tag found in {} — deriving read multiplicity from read name. {}; \
             surviving multi-locus reads go to the multimapper EM{}.",
            bam_path,
            filter_desc,
            if use_alignment_scoring { " with AS-weighted priors" } else { " at equal weight" },
        ),))?;
        logging.call_method1("info", (format!(
            "  MAPQ floor for reads resolving unique: {}{} (not min_mapq={}, a STAR \
             calibration that would re-test the alternative the AS filter already \
             rejected, and would delete rather than reclassify near-ties)",
            lr_mapq_floor,
            if lr_mapq_floor == 0 { " (off)" } else { "" }, min_mapq,
        ),))?;
    }

    // Tag keys — constructed once, not per record
    let nh_tag = Tag::new(b'N', b'H');
    let as_tag = Tag::new(b'A', b'S');
    let hi_tag = Tag::new(b'H', b'I');

    // Iterate over records
    for result in reader.record_bufs(&header) {
        let record = result
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("BAM read error: {}", e)))?;

        // Get flags
        let flags = record.flags();

        // Skip unmapped
        if flags.is_unmapped() {
            stats.unmapped += 1;
            continue;
        }

        // Skip supplementary alignments only — secondary alignments carry
        // multimapper loci and must be processed
        if flags.is_supplementary() {
            continue;
        }

        let is_secondary = flags.is_secondary();

        // Only count stats/sample read lengths/check max_reads for primary alignments
        if !is_secondary {
            stats.total_reads += 1;

            // Sample read lengths from first 10K reads
            if !stats.read_length_sampling_done {
                let seq_len = record.sequence().len() as u32;
                if seq_len > 0 {
                    stats.read_length_samples.push(seq_len);
                }
                if stats.read_length_samples.len() >= 10_000 {
                    stats.read_length_sampling_done = true;
                }
            }

            if let Some(max) = max_reads {
                if stats.total_reads > max {
                    break;
                }
            }
        }

        // Periodic interrupt check + progress logging every 10M records
        total_records += 1;
        reads_since_check += 1;
        if reads_since_check >= 500_000 {
            reads_since_check = 0;
            py.check_signals()?;

            if total_records >= next_progress {
                next_progress += 10_000_000;
                let elapsed = scan_start.elapsed().as_secs_f64();
                let rate = total_records as f64 / elapsed / 1000.0;
                logging.call_method1("info", (format!(
                    "  {:.0}M records ({:.0}M primary), {:.1}s ({:.0}K/s), unique EqC: {}, multi buffer: {}",
                    total_records as f64 / 1e6,
                    stats.total_reads as f64 / 1e6,
                    elapsed,
                    rate,
                    unique_eqc.len(),
                    if derive_mode { lr_reads.len() } else { multi_read_features.len() },
                ),))?;
            }
        }

        // NH tag
        let nh: u8 = record.data().get(&nh_tag)
            .and_then(|v| {
                use sam::alignment::record_buf::data::field::Value;
                match v {
                    Value::UInt8(n) => Some(*n),
                    Value::UInt16(n) => Some(*n as u8),
                    Value::UInt32(n) => Some(*n as u8),
                    Value::Int8(n) => Some(*n as u8),
                    Value::Int16(n) => Some(*n as u8),
                    Value::Int32(n) => Some(*n as u8),
                    _ => None,
                }
            })
            .unwrap_or(1);

        let is_unique = nh == 1;

        // Secondary alignment flagged as unique — skip (shouldn't happen).
        // In derive mode `nh` is a fabrication, so `is_unique` carries no
        // information and this guard, written to catch a malformed STAR BAM,
        // would instead discard every secondary alignment in the file — 28.1% of
        // records on an ONT library.
        if !derive_mode && is_secondary && is_unique {
            continue;
        }

        // HI tag (hit index) — used to pair PE mates at the correct locus
        let hi_value: u32 = record.data().get(&hi_tag)
            .and_then(|v| {
                use sam::alignment::record_buf::data::field::Value;
                match v {
                    Value::UInt8(n) => Some(*n as u32),
                    Value::UInt16(n) => Some(*n as u32),
                    Value::UInt32(n) => Some(*n as u32),
                    Value::Int8(n) => Some(*n as u32),
                    Value::Int16(n) => Some(*n as u32),
                    Value::Int32(n) => Some(*n as u32),
                    _ => None,
                }
            })
            .unwrap_or(0);

        // Reference sequence ID. The chrom_id lookup that used to sit here has
        // moved below the CIGAR parse: bailing out on an unannotated contig
        // before counting the splice made spliced_reads a count of "spliced
        // reads on chromosomes we have annotation for", which undercounted the
        // real figure by ~1k on a 7.5M-pair library and disagreed with the
        // pysam builder, which has no such early exit.
        let ref_seq_id = match record.reference_sequence_id() {
            Some(id) => id,
            None => continue,
        };

        // Position (0-based in noodles)
        let ref_start = match record.alignment_start() {
            Some(pos) => pos.get() as i64 - 1, // noodles is 1-based, convert to 0-based
            None => continue,
        };

        // MAPQ
        let mapq = record.mapping_quality().map(|q| q.get()).unwrap_or(255);

        // AS tag (alignment score) — extracted when alignment scoring is enabled,
        // and always in derive mode, where it is the only evidence available for
        // separating a real second locus from one minimap2 merely reported.
        let as_score: i64 = if use_alignment_scoring || derive_mode {
            record.data().get(&as_tag)
                .and_then(|v| {
                    use sam::alignment::record_buf::data::field::Value;
                    match v {
                        Value::Int8(n) => Some(*n as i64),
                        Value::Int16(n) => Some(*n as i64),
                        Value::Int32(n) => Some(*n as i64),
                        Value::UInt8(n) => Some(*n as i64),
                        Value::UInt16(n) => Some(*n as i64),
                        Value::UInt32(n) => Some(*n as i64),
                        _ => None,
                    }
                })
                .unwrap_or(0)
        } else {
            0
        };

        // Strand inference
        let is_read2 = flags.is_last_segment();
        let is_reverse = flags.is_reverse_complemented();
        let tx_strand = infer_read_strand(is_read2, is_reverse, strandedness);

        // Parse CIGAR -> blocks + junctions in one pass (no String allocation)
        let cigar = record.cigar();
        if CigarTrait::is_empty(&cigar) {
            continue;
        }
        let (aligned_blocks, splice_junctions) = parse_cigar(&cigar, ref_start);

        // Outer genomic bounds of this alignment, carried through the mate buffer so a
        // fragment length can be measured once both mates are in hand. parse_cigar walks
        // the CIGAR left to right, so blocks are already ascending. The (0, 0) fallback
        // is unreachable in practice: a CIGAR with no reference-consuming M/=/X op yields
        // no exon overlap and is dropped as no_feature a few lines below.
        let read_lo = aligned_blocks.first().map(|b| b.0).unwrap_or(0);
        let read_hi = aligned_blocks.last().map(|b| b.1).unwrap_or(0);

        if !splice_junctions.is_empty() {
            stats.spliced_reads += 1;
        }

        // Contigs absent from the annotation (scaffolds, decoys) can produce no
        // features by definition, so they are no_feature without a query.
        // `no_feature` is a per-READ counter — the identity
        // `unique_mapped + low_mapq_unique + no_feature == total_reads` depends on
        // it. Derive mode is the first time secondary alignments reach this query
        // en masse (the NH guard used to eat them), and counting them here inflated
        // it 2.7x on an ONT library and broke that identity. Secondaries land in
        // their own counter instead.
        let chrom_id = match ref_to_chrom_id.get(ref_seq_id).and_then(|id| *id) {
            Some(id) => id,
            None => {
                if derive_mode && is_secondary {
                    stats.lr_secondary_no_feature += 1;
                } else {
                    stats.no_feature += 1;
                }
                continue;
            }
        };

        // Query exon overlap (inclusion — broad candidate set)
        let exon_overlap_txs = exon_index.query_blocks(chrom_id, &aligned_blocks, tx_strand);

        if exon_overlap_txs.is_empty() {
            if derive_mode && is_secondary {
                stats.lr_secondary_no_feature += 1;
            } else {
                stats.no_feature += 1;
            }
            continue;
        }

        // Exon containment filtering (exclusion — stricter than overlap)
        let contained_txs = exon_index.query_blocks_contained(chrom_id, &aligned_blocks, tx_strand);
        let has_containment = !contained_txs.is_empty();

        // Query junction compatibility using tuple keys
        let mut junction_compatible_txs: Option<FxHashSet<u32>> = None;
        if !splice_junctions.is_empty() {
            let mut junc_sets: Vec<&FxHashSet<u32>> = Vec::new();
            for &(donor, acceptor) in &splice_junctions {
                let junc_key = (chrom_id, donor, acceptor);
                if let Some(tx_set) = junc_lookup.get(&junc_key) {
                    junc_sets.push(tx_set);
                    *junction_agg_counts.entry(junc_key).or_insert(0) += 1;
                }
            }
            if !junc_sets.is_empty() {
                let mut intersection = junc_sets[0].clone();
                for s in &junc_sets[1..] {
                    intersection.retain(|x| s.contains(x));
                }
                junction_compatible_txs = Some(intersection);
            } else {
                stats.junction_novel += 1;
            }
        }

        // Refined assignment: try containment ∩ junctions first, then fallbacks
        let features: FxHashSet<u32>;
        if let Some(ref junc_txs) = junction_compatible_txs {
            if has_containment {
                // Best case: containment ∩ junctions ∩ overlap
                let triple: FxHashSet<u32> = contained_txs.iter()
                    .filter(|tx| junc_txs.contains(tx) && exon_overlap_txs.contains(tx))
                    .copied().collect();
                if !triple.is_empty() {
                    if triple.len() < exon_overlap_txs.len() {
                        // Count which filter contributed
                        let junc_only: FxHashSet<u32> = exon_overlap_txs.intersection(junc_txs).copied().collect();
                        if junc_only.len() < exon_overlap_txs.len() {
                            stats.junction_refined += 1;
                        }
                        if triple.len() < junc_only.len() {
                            stats.exon_refined += 1;
                        }
                    }
                    features = triple;
                } else {
                    // Triple empty — try overlap ∩ junctions (current behavior)
                    let junc_refined: FxHashSet<u32> = exon_overlap_txs.intersection(junc_txs).copied().collect();
                    if !junc_refined.is_empty() {
                        if junc_refined.len() < exon_overlap_txs.len() {
                            stats.junction_refined += 1;
                        }
                        stats.exon_incompatible += 1;
                        features = junc_refined;
                    } else {
                        // Try overlap ∩ containment
                        let cont_refined: FxHashSet<u32> = exon_overlap_txs.intersection(&contained_txs).copied().collect();
                        if !cont_refined.is_empty() {
                            stats.junction_incompatible += 1;
                            if cont_refined.len() < exon_overlap_txs.len() {
                                stats.exon_refined += 1;
                            }
                            features = cont_refined;
                        } else {
                            stats.junction_incompatible += 1;
                            stats.exon_incompatible += 1;
                            features = exon_overlap_txs;
                        }
                    }
                }
            } else {
                // No containment candidates — junction-only refinement (previous behavior)
                let junc_refined: FxHashSet<u32> = exon_overlap_txs.intersection(junc_txs).copied().collect();
                if !junc_refined.is_empty() {
                    if junc_refined.len() < exon_overlap_txs.len() {
                        stats.junction_refined += 1;
                    }
                    features = junc_refined;
                } else {
                    stats.junction_incompatible += 1;
                    features = exon_overlap_txs;
                }
            }
        } else if has_containment {
            // No junctions — containment-only refinement
            let cont_refined: FxHashSet<u32> = exon_overlap_txs.intersection(&contained_txs).copied().collect();
            if !cont_refined.is_empty() {
                if cont_refined.len() < exon_overlap_txs.len() {
                    stats.exon_refined += 1;
                }
                features = cont_refined;
            } else {
                stats.exon_incompatible += 1;
                features = exon_overlap_txs;
            }
        } else {
            // No containment, no junctions — overlap only (fallback)
            features = exon_overlap_txs;
        }

        // Derive-multiplicity route. Multiplicity is unknowable here: this BAM is
        // coordinate-sorted, so a read's secondaries sit nowhere near its primary
        // and nothing local says whether more alignments of this read exist. So
        // buffer every alignment under its read name and classify the whole file
        // at end-of-scan. The cost is holding the uniques too — fine for a 10M-read
        // ONT library, which is why the mode is gated on the missing tag rather
        // than being the default.
        if derive_mode {
            let read_name = record.name()
                .map(|n| <_ as AsRef<[u8]>>::as_ref(&n).to_vec())
                .unwrap_or_default();
            let mut feats: Vec<u32> = features.into_iter().collect();
            feats.sort_unstable();
            lr_reads.entry(read_name).or_default().push(LrAlignment {
                features: feats,
                score: as_score,
                mapq,
            });
            continue;
        }

        // Paired-end mate handling — key by (name, HI) to pair mates at correct locus
        if paired_end && flags.is_segmented() {
            let name_opt = record.name();
            let name_bytes: &[u8] = match &name_opt {
                Some(n) => <_ as AsRef<[u8]>>::as_ref(n),
                None => b"",
            };

            let mate_key = (name_bytes.to_vec(), hi_value);
            if let Some((mate_features, mate_is_unique, mate_as, mate_chrom, mate_lo, mate_hi)) =
                mate_buffer.remove(&mate_key)
            {
                let combined: FxHashSet<u32> = features.intersection(&mate_features).copied().collect();
                let final_features;
                if !combined.is_empty() {
                    final_features = combined;
                } else {
                    final_features = features.union(&mate_features).copied().collect();
                    if !is_secondary {
                        stats.paired_intersection_empty += 1;
                    }
                }
                let final_is_unique = is_unique && mate_is_unique;
                let pair_as = as_score + mate_as;
                if !is_secondary {
                    stats.paired_reads_processed += 1;
                }

                if final_is_unique {
                    if mapq < min_mapq {
                        stats.low_mapq_unique += 1;
                        continue;
                    }
                    // Fragment length, measured in cDNA space on the one transcript
                    // this pair is compatible with. Restricting to NH==1 pairs with a
                    // single compatible transcript is what Salmon and RSEM do to seed
                    // their fragment length distribution, and it is what makes the
                    // length unambiguous: with two candidate transcripts the same
                    // pair implies two different lengths.
                    if measure_fragment_length && mate_chrom == chrom_id {
                        let lo = mate_lo.min(read_lo);
                        let hi = mate_hi.max(read_hi);

                        // Primary estimator: spread the pair over every transcript it
                        // is compatible with, 1/k each.
                        //
                        // The obvious alternative -- keep only pairs compatible with
                        // exactly one transcript -- is a *length-dependent* filter: a
                        // longer fragment spans more structure, so it is likelier to
                        // become uniquely assignable, and conditioning on uniqueness
                        // over-samples long fragments (measured: +8% on the mean, and
                        // it keeps only ~18% of pairs). Salmon and RSEM do not
                        // condition that way -- they spread a fragment over its
                        // compatible transcripts under a uniform prior -- and matching
                        // them is what puts this estimator on their answer.
                        //
                        // Transcripts that cannot produce the span (an endpoint in an
                        // intron) drop out and the weight renormalises over the rest,
                        // which is what makes the estimate immune to the intron
                        // contamination that inflates TLEN on a genome BAM.
                        let mut placed: Vec<u32> = Vec::with_capacity(final_features.len());
                        for &tx in final_features.iter() {
                            if let Some(flen) = exon_index.spliced_distance(tx, lo, hi) {
                                placed.push(flen);
                            }
                        }
                        if !placed.is_empty() {
                            let w = 1.0 / placed.len() as f64;
                            for flen in placed {
                                *stats.fragment_length_hist.entry(flen).or_insert(0.0) += w;
                            }
                        }

                        // Diagnostics on the unique-transcript subset: the biased
                        // estimator above, and the same pairs measured as an unspliced
                        // genomic span (= what TLEN reports, and what v1 used). Both
                        // are accumulated on identical pairs, so the three-way
                        // comparison isolates the estimator from the read set.
                        if final_features.len() == 1 {
                            let tx = *final_features.iter().next().unwrap();
                            if let Some(flen) = exon_index.spliced_distance(tx, lo, hi) {
                                *stats.fragment_length_hist_unique.entry(flen).or_insert(0) += 1;
                                *stats.genomic_span_hist.entry(hi - lo + 1).or_insert(0) += 1;
                            } else {
                                stats.fragment_span_not_in_exons += 1;
                            }
                        }
                    }
                    accumulate_unique(&final_features, &mut unique_eqc, &mut unique_counts, &mut stats);
                } else {
                    if !is_secondary {
                        stats.multimapped += 1;
                    }
                    if use_alignment_scoring {
                        multi_read_alignments.entry(name_bytes.to_vec()).or_default()
                            .push((final_features, pair_as));
                    } else {
                        multi_read_features.entry(name_bytes.to_vec()).or_default().extend(&final_features);
                    }
                }
            } else {
                mate_buffer.insert(
                    mate_key,
                    (features, is_unique, as_score, chrom_id, read_lo, read_hi),
                );
            }
            continue;
        }

        // Single-end route
        if is_unique {
            if mapq < min_mapq {
                stats.low_mapq_unique += 1;
                continue;
            }
            accumulate_unique(&features, &mut unique_eqc, &mut unique_counts, &mut stats);
        } else {
            if !is_secondary {
                stats.multimapped += 1;
            }
            let read_name = record.name().map(|n| <_ as AsRef<[u8]>>::as_ref(&n).to_vec()).unwrap_or_default();
            if use_alignment_scoring {
                multi_read_alignments.entry(read_name).or_default()
                    .push((features, as_score));
            } else {
                multi_read_features.entry(read_name).or_default().extend(&features);
            }
        }
    }

    // Process orphaned mates (mate never found — count them solo)
    if paired_end && !mate_buffer.is_empty() {
        stats.orphaned_mates = mate_buffer.len() as u64;
        for ((name, _hi), (features, orphan_is_unique, orphan_as, _chrom, _lo, _hi_pos)) in
            mate_buffer.drain()
        {
            if orphan_is_unique {
                accumulate_unique(&features, &mut unique_eqc, &mut unique_counts, &mut stats);
            } else {
                if use_alignment_scoring {
                    multi_read_alignments.entry(name).or_default()
                        .push((features, orphan_as));
                } else {
                    multi_read_features.entry(name).or_default().extend(&features);
                }
            }
        }
    }

    // Build multimapper equivalence classes
    let mut multi_eqc: FxHashMap<Vec<u32>, f64> = FxHashMap::default();
    let mut multi_counts: FxHashMap<u32, f64> = FxHashMap::default();

    if derive_mode {
        stats.lr_reads_buffered = lr_reads.len() as u64;
        for (_name, alns) in &lr_reads {
            if alns.is_empty() {
                continue;
            }

            // Score filter. minimap2 reports up to `-N` chains scoring >= `-p` of
            // the primary's chaining score, which is far looser than STAR's
            // `--outFilterMultimapScoreRange 1`: on A549 directcDNA, 32% of reads
            // with a reported secondary carry a primary MAPQ of 60, i.e. minimap2
            // has already decided the alternative is not competitive. Keeping only
            // alignments within `lr_score_ratio` of the best is the NanoCount /
            // oarfish rule, and it takes the multiread rate on that library from
            // 25.3% of reads to ~10.9%.
            //
            // Applied as a gap in AS units rather than as a ratio, because
            // minimap2 AS goes negative (min -608 on that library) and a ratio
            // inverts once the denominator is <= 0.
            let best = alns.iter().map(|a| a.score).max().unwrap();
            let keep: Vec<&LrAlignment> = if lr_score_ratio <= 0.0 {
                // Filter off. Has to be an explicit branch: a cutoff computed as
                // `best - 1.0 * best` is 0, not -infinity, so the arithmetic path
                // would still silently drop every negative-AS alignment and the
                // "no filter" arm would not be one.
                alns.iter().collect()
            } else {
                let cutoff = if best > 0 {
                    best - ((1.0 - lr_score_ratio) * best as f64).round() as i64
                } else {
                    best
                };
                alns.iter().filter(|a| a.score >= cutoff).collect()
            };
            stats.lr_alignments_dropped += (alns.len() - keep.len()) as u64;

            if keep.len() == 1 {
                // Resolved unique — and only now is the unique-mapper MAPQ filter
                // a meaningful test.
                let a = keep[0];
                if a.mapq < lr_mapq_floor {
                    stats.low_mapq_unique += 1;
                    continue;
                }
                stats.lr_reads_unique += 1;
                let fs: FxHashSet<u32> = a.features.iter().copied().collect();
                accumulate_unique(&fs, &mut unique_eqc, &mut unique_counts, &mut stats);
                continue;
            }

            // A genuine multiread, and deliberately NOT subject to min_mapq: a
            // read with two near-equal loci is *supposed* to score low, and
            // filtering it there is what deleted 12.3% of an ONT library's primary
            // alignments under the NH-assuming path. Those reads reach the EM now.
            stats.lr_reads_multi += 1;
            stats.multimapped += 1;
            // Deliberately measured against `min_mapq`, not `lr_mapq_floor`: this
            // counter answers "how many reads would the NH-assuming path have
            // deleted?", and that path used min_mapq.
            if keep.iter().all(|a| a.mapq < min_mapq) {
                stats.lr_multi_rescued_low_mapq += 1;
            }

            if use_alignment_scoring {
                // Relative temperature. `as_temperature` is in raw AS units and
                // those are not comparable across aligners: STAR's score range 1
                // bounds competitors to within a single point, while minimap2 AS
                // on this library runs a median of 410. The same tau=10 is
                // therefore near-uniform on STAR and near-argmax on minimap2 — one
                // number meaning opposite things. Here it is read as a percentage
                // of the read's own best score, so it means one thing at either
                // scale. This is why MAJEC's short-read "AS-softmax is noise"
                // result does not transfer: it was measured where there was no
                // score spread to weight by.
                let tau = (as_temperature * best.max(1) as f64 / 100.0).max(1e-9);
                let weights: Vec<f64> = keep.iter()
                    .map(|a| ((a.score - best) as f64 / tau).exp())
                    .collect();
                let total_weight: f64 = weights.iter().sum();
                for (i, a) in keep.iter().enumerate() {
                    let w = weights[i] / total_weight;
                    *multi_eqc.entry(a.features.clone()).or_insert(0.0) += w;
                    let fraction = w / a.features.len() as f64;
                    for &f in &a.features {
                        *multi_counts.entry(f).or_insert(0.0) += fraction;
                    }
                }
            } else {
                // Equal weight. After filtering to within `lr_score_ratio` the
                // survivors are near-ties by construction, which is precisely the
                // regime where the short-read work concluded AS weighting adds
                // noise. Union the loci into a single EqC — the same object the
                // short-read multimapper path emits — so nothing downstream of
                // here has to know the reads were long.
                let mut key: Vec<u32> = Vec::new();
                for a in &keep {
                    key.extend_from_slice(&a.features);
                }
                key.sort_unstable();
                key.dedup();
                *multi_eqc.entry(key.clone()).or_insert(0.0) += 1.0;
                let fraction = 1.0 / key.len() as f64;
                for &f in &key {
                    *multi_counts.entry(f).or_insert(0.0) += fraction;
                }
            }
        }
        drop(lr_reads);
    } else if use_alignment_scoring {
        // AS-weighted: softmax over per-alignment AS scores, fractional EqC counts
        for (_name, alignments) in &multi_read_alignments {
            if alignments.is_empty() {
                continue;
            }
            if alignments.len() == 1 {
                // Single alignment — weight is 1.0
                let (ref feats, _score) = alignments[0];
                let mut key: Vec<u32> = feats.iter().copied().collect();
                key.sort_unstable();
                *multi_eqc.entry(key).or_insert(0.0) += 1.0;
                let fraction = 1.0 / feats.len() as f64;
                for &f in feats {
                    *multi_counts.entry(f).or_insert(0.0) += fraction;
                }
            } else {
                // Softmax: w_i = exp(AS_i / tau), normalized
                let max_as = alignments.iter().map(|(_, s)| *s).max().unwrap_or(0);
                let weights: Vec<f64> = alignments.iter()
                    .map(|(_, s)| ((*s - max_as) as f64 / as_temperature).exp())
                    .collect();
                let total_weight: f64 = weights.iter().sum();

                for (i, (ref feats, _)) in alignments.iter().enumerate() {
                    let w = weights[i] / total_weight;
                    let mut key: Vec<u32> = feats.iter().copied().collect();
                    key.sort_unstable();
                    *multi_eqc.entry(key).or_insert(0.0) += w;
                    let fraction = w / feats.len() as f64;
                    for &f in feats {
                        *multi_counts.entry(f).or_insert(0.0) += fraction;
                    }
                }
            }
        }
        drop(multi_read_alignments);
    } else {
        // Original behavior: union features, integer-like counts
        for (_name, feature_set) in &multi_read_features {
            let mut key: Vec<u32> = feature_set.iter().copied().collect();
            key.sort_unstable();
            *multi_eqc.entry(key).or_insert(0.0) += 1.0;
            let fraction = 1.0 / feature_set.len() as f64;
            for &f in feature_set {
                *multi_counts.entry(f).or_insert(0.0) += fraction;
            }
        }
        drop(multi_read_features);
    }

    // ---- Convert to Python ----
    let result = PyDict::new(py);

    // unique_eqc: {tuple(ints): count}
    let py_unique_eqc = PyDict::new(py);
    for (key, count) in &unique_eqc {
        let py_key = PyTuple::new(py, key.iter().copied())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        py_unique_eqc.set_item(py_key, *count)?;
    }
    result.set_item("unique_eqc", py_unique_eqc)?;

    // unique_counts: {int: float}
    let py_unique_counts = PyDict::new(py);
    for (&feat, &count) in &unique_counts {
        py_unique_counts.set_item(feat, count)?;
    }
    result.set_item("unique_counts", py_unique_counts)?;

    // multi_eqc: {tuple(ints): count} (f64 — integer-valued when AS-weighting is off)
    let py_multi_eqc = PyDict::new(py);
    for (key, count) in &multi_eqc {
        let py_key = PyTuple::new(py, key.iter().copied())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        py_multi_eqc.set_item(py_key, *count)?;
    }
    result.set_item("multi_eqc", py_multi_eqc)?;

    // multi_counts: {int: float}
    let py_multi_counts = PyDict::new(py);
    for (&feat, &count) in &multi_counts {
        py_multi_counts.set_item(feat, count)?;
    }
    result.set_item("multi_counts", py_multi_counts)?;

    // junction_counts: rebuild string keys only for final Python export
    let py_junc_counts = PyDict::new(py);
    for ((chrom_id, donor, acceptor), count) in &junction_agg_counts {
        if let Some(chrom) = chrom_id_to_name.get(chrom_id) {
            py_junc_counts.set_item(format!("{}:{}-{}", chrom, donor, acceptor), *count)?;
        }
    }
    result.set_item("junction_counts", py_junc_counts)?;

    // stats
    let stats_dict = stats.to_pydict(py);
    stats_dict.set_item("unique_eqc_count", unique_eqc.len())?;
    stats_dict.set_item("multi_eqc_count", multi_eqc.len())?;
    stats_dict.set_item("total_unique_reads_in_eqc", unique_eqc.values().sum::<u32>())?;
    // Round, not truncate. Under alignment-score weighting the per-class
    // weights are fractional and their sum lands a few ulps under the true
    // read count, which a bare `as u64` floors away as an off-by-one.
    stats_dict.set_item("total_multi_reads_in_eqc",
                        multi_eqc.values().sum::<f64>().round() as u64)?;
    result.set_item("stats", stats_dict)?;

    Ok(result.into())
}
