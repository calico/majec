//! Binned exon interval index for fast genomic overlap queries.
//!
//! Mirrors the Python ExonIntervalIndex but uses Rust HashMaps for speed.
//! Constructed from Python annotation data via `from_annotation()`.

use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

const BIN_SIZE: u32 = 1000;

/// An exon interval: (start, end, transcript_int).
/// Coordinates are 1-based inclusive (GTF convention).
type ExonInterval = (u32, u32, u32);

/// Key for chromosome + strand lookup.
/// Strand is encoded as: b'+' = 43, b'-' = 45, 0 = unstranded.
/// chrom_id is an internal numeric ID assigned during index construction.
type ChromStrandKey = (usize, u8);

/// Fast genomic interval lookup using a binned index.
///
/// For each chromosome+strand, genomic space is divided into 1kb bins.
/// Each bin stores the exon intervals that overlap it.
/// Chromosomes are mapped to internal numeric IDs to avoid string allocations
/// during per-read queries.
#[pyclass]
pub struct ExonIntervalIndex {
    /// (chrom_id, strand_byte) -> bin_idx -> Vec<(start, end, tx_int)>
    bins: FxHashMap<ChromStrandKey, FxHashMap<u32, Vec<ExonInterval>>>,
    /// chromosome name -> internal numeric ID
    chrom_to_id: FxHashMap<String, usize>,
    n_exons: usize,
    /// tx_int -> exons in genomic order as (start, end, cum_offset), where
    /// cum_offset is the spliced (cDNA) coordinate of the exon's first base.
    ///
    /// The binned index above answers *membership* ("which transcripts touch
    /// this span"). It cannot answer *distance* ("how far apart are two
    /// genomic points once introns are removed"), which is what a fragment
    /// length is. Offsets accumulate in genomic order regardless of strand:
    /// a distance is symmetric, so 5'->3' direction is irrelevant here.
    tx_exons: FxHashMap<u32, Vec<ExonInterval>>,
}

#[pymethods]
impl ExonIntervalIndex {
    /// Build from annotation data passed in from Python.
    ///
    /// Args:
    ///     transcript_exons: list of (transcript_int, [(chrom, start, end, strand), ...])
    ///         where strand is "+" or "-"
    #[new]
    fn new() -> Self {
        ExonIntervalIndex {
            bins: FxHashMap::default(),
            chrom_to_id: FxHashMap::default(),
            n_exons: 0,
            tx_exons: FxHashMap::default(),
        }
    }

    /// Add exons from Python annotation data.
    ///
    /// transcript_exons: list of (tx_int, chrom, start, end, strand_str) tuples
    fn add_exons(&mut self, exon_records: Vec<(u32, String, u32, u32, String)>) {
        for (tx_int, chrom, start, end, strand_str) in exon_records {
            let strand_byte = strand_str.as_bytes().first().copied().unwrap_or(0);
            self.add_exon(&chrom, start, end, strand_byte, tx_int);
        }
    }

    /// Finalize the index by sorting intervals within each bin, and build the
    /// per-transcript cumulative-offset exon vectors used for spliced distance.
    fn finalize(&mut self) {
        for bin_map in self.bins.values_mut() {
            for intervals in bin_map.values_mut() {
                intervals.sort_unstable();
            }
        }
        // Exons arrive in arbitrary order and an annotation may repeat one, so
        // sort and dedup before accumulating offsets — a duplicated exon would
        // otherwise inflate every downstream cDNA coordinate in that transcript.
        for exons in self.tx_exons.values_mut() {
            exons.sort_unstable_by_key(|&(start, end, _)| (start, end));
            exons.dedup_by_key(|&mut (start, end, _)| (start, end));
            let mut cum: u32 = 0;
            for e in exons.iter_mut() {
                e.2 = cum;
                cum += e.1 - e.0 + 1;
            }
        }
    }

    /// Number of exon intervals in the index.
    #[getter]
    fn n_exons(&self) -> usize {
        self.n_exons
    }

    /// Build the index from Python annotation data in one call.
    ///
    /// Args:
    ///     exon_records: list of (tx_int, chrom, start, end, strand) tuples
    ///
    /// Returns:
    ///     A finalized ExonIntervalIndex ready for querying.
    #[staticmethod]
    fn from_annotation(exon_records: Vec<(u32, String, u32, u32, String)>) -> Self {
        let mut index = ExonIntervalIndex::new();
        index.add_exons(exon_records);
        index.finalize();
        index
    }
}

impl ExonIntervalIndex {
    /// Get or assign a numeric ID for a chromosome name.
    fn get_or_assign_chrom_id(&mut self, chrom: &str) -> usize {
        let next_id = self.chrom_to_id.len();
        *self.chrom_to_id.entry(chrom.to_string()).or_insert(next_id)
    }

    /// Look up the internal chrom_id for a chromosome name.
    /// Returns None if the chromosome was never added to the index.
    pub fn get_chrom_id(&self, chrom: &str) -> Option<usize> {
        self.chrom_to_id.get(chrom).copied()
    }

    fn add_exon(&mut self, chrom: &str, start: u32, end: u32, strand: u8, tx_int: u32) {
        let chrom_id = self.get_or_assign_chrom_id(chrom);
        let key = (chrom_id, strand);
        let start_bin = start / BIN_SIZE;
        let end_bin = end / BIN_SIZE;
        let bin_map = self.bins.entry(key).or_default();
        for b in start_bin..=end_bin {
            bin_map.entry(b).or_default().push((start, end, tx_int));
        }
        // Once per exon (not once per bin) — offsets are filled in finalize().
        self.tx_exons.entry(tx_int).or_default().push((start, end, 0));
        self.n_exons += 1;
    }

    /// Spliced (cDNA) distance between two genomic points on one transcript.
    ///
    /// Returns the inclusive base count from `g_start` to `g_end` with introns
    /// removed, or `None` if either endpoint falls outside this transcript's
    /// exons — which is the correct answer for a pair whose implied fragment
    /// is not producible from this transcript, and the reason the estimate is
    /// immune to the intron contamination that inflates TLEN on a genome BAM.
    ///
    /// Requires `finalize()` to have run.
    pub fn spliced_distance(&self, tx_int: u32, g_start: u32, g_end: u32) -> Option<u32> {
        if g_end < g_start {
            return None;
        }
        let exons = self.tx_exons.get(&tx_int)?;
        let cdna_of = |pos: u32| -> Option<u32> {
            // Rightmost exon whose start is <= pos; exons are disjoint and sorted.
            let i = exons.partition_point(|&(start, _, _)| start <= pos);
            if i == 0 {
                return None;
            }
            let (start, end, cum) = exons[i - 1];
            if pos <= end {
                Some(cum + (pos - start))
            } else {
                None // falls in an intron (or past the last exon)
            }
        };
        Some(cdna_of(g_end)? - cdna_of(g_start)? + 1)
    }

    /// Find all transcript_ints with exons overlapping [start, end] on the given strand.
    /// If strand == 0, queries both '+' and '-'.
    pub fn query_interval(&self, chrom_id: usize, start: u32, end: u32, strand: u8) -> FxHashSet<u32> {
        let mut result = FxHashSet::default();
        let strands: &[u8] = if strand == 0 { &[b'+', b'-'] } else { &[strand] };

        for &s in strands {
            let key = (chrom_id, s);
            if let Some(bin_map) = self.bins.get(&key) {
                let start_bin = start / BIN_SIZE;
                let end_bin = end / BIN_SIZE;
                for b in start_bin..=end_bin {
                    if let Some(intervals) = bin_map.get(&b) {
                        for &(ex_start, ex_end, tx_int) in intervals {
                            if start <= ex_end && end >= ex_start {
                                result.insert(tx_int);
                            }
                        }
                    }
                }
            }
        }
        result
    }

    /// Query multiple aligned blocks, returning transcripts overlapping ANY block.
    pub fn query_blocks(&self, chrom_id: usize, blocks: &[(u32, u32)], strand: u8) -> FxHashSet<u32> {
        let mut result = FxHashSet::default();
        let strands: &[u8] = if strand == 0 { &[b'+', b'-'] } else { &[strand] };
        for &(start, end) in blocks {
            for &s in strands {
                let key = (chrom_id, s);
                if let Some(bin_map) = self.bins.get(&key) {
                    let start_bin = start / BIN_SIZE;
                    let end_bin = end / BIN_SIZE;
                    for b in start_bin..=end_bin {
                        if let Some(intervals) = bin_map.get(&b) {
                            for &(ex_start, ex_end, tx_int) in intervals {
                                if start <= ex_end && end >= ex_start {
                                    result.insert(tx_int);
                                }
                            }
                        }
                    }
                }
            }
        }
        result
    }

    /// Exon containment query: returns transcripts where EVERY aligned block
    /// is fully contained by at least one exon of that transcript.
    ///
    /// "Contained" means `ex_start <= block_start && block_end <= ex_end`.
    ///
    /// For multi-block (spliced) reads, a transcript must pass containment
    /// for each block independently (different exons can contain different blocks).
    pub fn query_blocks_contained(&self, chrom_id: usize, blocks: &[(u32, u32)], strand: u8) -> FxHashSet<u32> {
        let strands: &[u8] = if strand == 0 { &[b'+', b'-'] } else { &[strand] };

        // For each block, find transcripts with a containing exon
        let mut per_block_sets: Vec<FxHashSet<u32>> = Vec::with_capacity(blocks.len());

        for &(start, end) in blocks {
            let mut contained_txs = FxHashSet::default();
            for &s in strands {
                let key = (chrom_id, s);
                if let Some(bin_map) = self.bins.get(&key) {
                    let start_bin = start / BIN_SIZE;
                    let end_bin = end / BIN_SIZE;
                    for b in start_bin..=end_bin {
                        if let Some(intervals) = bin_map.get(&b) {
                            for &(ex_start, ex_end, tx_int) in intervals {
                                if ex_start <= start && end <= ex_end {
                                    contained_txs.insert(tx_int);
                                }
                            }
                        }
                    }
                }
            }
            per_block_sets.push(contained_txs);
        }

        // Intersect across all blocks: transcript must pass containment for every block
        if per_block_sets.is_empty() {
            return FxHashSet::default();
        }

        // Start with smallest set for efficiency
        let min_idx = per_block_sets.iter().enumerate()
            .min_by_key(|(_, s)| s.len())
            .map(|(i, _)| i)
            .unwrap();
        let mut result = per_block_sets.swap_remove(min_idx);
        for other in &per_block_sets {
            result.retain(|tx| other.contains(tx));
        }
        result
    }
}
