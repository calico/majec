"""
Junction-aware equivalence class builder for MAJEC.

Single-pass BAM scanner that builds refined equivalence classes by combining
exon overlap AND splice junction compatibility, replacing the featureCounts
FIFO-based approach.

This is a library module imported by the pipeline. It is the pure-Python
fallback for the compiled majec_core builder (eqc_builder_rust.py).
"""

import logging
import math
import time
from bisect import bisect_right
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import pysam
except ImportError:
    pysam = None


# ---------------------------------------------------------------------------
# Exon interval index — binned approach, no external dependencies
# ---------------------------------------------------------------------------


class ExonIntervalIndex:
    """
    Fast genomic interval lookup using a binned index.

    For each chromosome+strand, genomic space is divided into fixed-size bins.
    Each bin stores the exon intervals that overlap it. Querying a position
    checks only the relevant bin.
    """

    BIN_SIZE = 1000  # 1kb bins — most exons fit in 1-2 bins

    def __init__(self):
        # {(chrom, strand): {bin_idx: [(start, end, tx_int), ...]}}
        self.bins = defaultdict(lambda: defaultdict(list))
        self.n_exons = 0

    def add_exon(self, chrom, start, end, strand, tx_int):
        """Add an exon interval. Coordinates are 1-based, inclusive (GTF style)."""
        key = (chrom, strand)
        start_bin = start // self.BIN_SIZE
        end_bin = end // self.BIN_SIZE
        for b in range(start_bin, end_bin + 1):
            self.bins[key][b].append((start, end, tx_int))
        self.n_exons += 1

    def finalize(self):
        """Sort intervals within each bin and convert to regular dicts for pickling."""
        regular_bins = {}
        for key in self.bins:
            regular_bins[key] = {}
            for b in self.bins[key]:
                self.bins[key][b].sort()
                regular_bins[key][b] = self.bins[key][b]
        self.bins = regular_bins

    def query_interval(self, chrom, start, end, strand):
        """
        Find all transcript_ints with exons overlapping [start, end].
        Returns a set of transcript_ints.

        If strand is None, queries both strands.
        """
        result = set()

        strands = [strand] if strand else ["+", "-"]
        for s in strands:
            key = (chrom, s)
            if key not in self.bins:
                continue
            start_bin = start // self.BIN_SIZE
            end_bin = end // self.BIN_SIZE
            for b in range(start_bin, end_bin + 1):
                if b not in self.bins[key]:
                    continue
                for ex_start, ex_end, tx_int in self.bins[key][b]:
                    if start <= ex_end and end >= ex_start:
                        result.add(tx_int)
        return result

    def query_blocks(self, chrom, blocks, strand):
        """
        Query multiple aligned blocks and return transcripts overlapping ANY block.
        blocks: list of (start, end) tuples in 1-based coordinates.
        """
        result = set()
        for start, end in blocks:
            result.update(self.query_interval(chrom, start, end, strand))
        return result

    def query_blocks_contained(self, chrom, blocks, strand):
        """
        Exon containment query: returns transcripts where EVERY aligned block
        is fully contained by at least one exon of that transcript.

        "Contained" means ex_start <= block_start and block_end <= ex_end.

        For multi-block (spliced) reads, a transcript must pass containment
        for each block independently (different exons can contain different blocks).
        """
        strands = [strand] if strand else ["+", "-"]

        per_block_sets = []
        for start, end in blocks:
            contained_txs = set()
            for s in strands:
                key = (chrom, s)
                if key not in self.bins:
                    continue
                start_bin = start // self.BIN_SIZE
                end_bin = end // self.BIN_SIZE
                for b in range(start_bin, end_bin + 1):
                    if b not in self.bins[key]:
                        continue
                    for ex_start, ex_end, tx_int in self.bins[key][b]:
                        if ex_start <= start and end <= ex_end:
                            contained_txs.add(tx_int)
            per_block_sets.append(contained_txs)

        if not per_block_sets:
            return set()

        # Intersect across all blocks
        result = min(per_block_sets, key=len).copy()
        for other in per_block_sets:
            result &= other
        return result

    # -- cDNA-space distance ------------------------------------------------
    #
    # The binned index above answers *membership* ("which transcripts touch
    # this span"). It cannot answer *distance* ("how far apart are two genomic
    # points once introns are removed"), which is what a fragment length is.
    # Offsets accumulate in genomic order regardless of strand: a distance is
    # symmetric, so 5'->3' direction is irrelevant here.
    #
    # Mirrors ExonIntervalIndex::spliced_distance in the Rust core.

    # Class-level default so indexes unpickled from annotations written before
    # this existed still work - their __dict__ has no _tx_exons.
    _tx_exons = None

    def __getstate__(self):
        """Keep the derived vectors out of the pickle.

        They are reconstructible from self.bins in a second or two and would
        otherwise roughly double the size of every annotation file on disk.
        """
        state = self.__dict__.copy()
        state.pop("_tx_exons", None)
        return state

    def _build_tx_exons(self):
        """Per-transcript exon vectors with cumulative cDNA offsets.

        Derived from self.bins rather than stored, so existing annotation
        pickles keep working with no rebuild. Bins duplicate an exon across
        every bucket it spans, hence the dedup - a repeated exon would inflate
        every downstream cDNA coordinate in that transcript.
        """
        t0 = time.time()
        per_tx = defaultdict(set)
        for bin_map in self.bins.values():
            for intervals in bin_map.values():
                for start, end, tx_int in intervals:
                    per_tx[tx_int].add((start, end))

        built = {}
        for tx_int, exons in per_tx.items():
            starts, ends, offsets = [], [], []
            cum = 0
            for s, e in sorted(exons):
                starts.append(s)
                ends.append(e)
                offsets.append(cum)
                cum += e - s + 1
            built[tx_int] = (starts, ends, offsets)

        self._tx_exons = built
        logging.info(
            f"  Built cDNA offset vectors for {len(built):,} transcripts "
            f"({time.time() - t0:.1f}s)"
        )

    def spliced_distance(self, tx_int, g_start, g_end):
        """Distance from g_start to g_end along tx_int with introns removed.

        Returns None if either endpoint falls in an intron or outside the
        transcript - the correct answer for a pair whose implied fragment this
        transcript cannot produce, and the reason the estimate is immune to the
        intron contamination that inflates TLEN on a genome BAM.
        """
        if g_end < g_start:
            return None
        if self._tx_exons is None:
            self._build_tx_exons()
        entry = self._tx_exons.get(tx_int)
        if entry is None:
            return None
        starts, ends, offsets = entry

        # Rightmost exon whose start is <= pos; exons are disjoint and sorted.
        i = bisect_right(starts, g_start)
        if i == 0 or g_start > ends[i - 1]:
            return None
        lo_cdna = offsets[i - 1] + (g_start - starts[i - 1])

        j = bisect_right(starts, g_end)
        if j == 0 or g_end > ends[j - 1]:
            return None
        hi_cdna = offsets[j - 1] + (g_end - starts[j - 1])

        return hi_cdna - lo_cdna + 1


# ---------------------------------------------------------------------------
# Fragment length histogram summaries
# ---------------------------------------------------------------------------

# Salmon's own fld tops out at 1000, so the capped summary is the like-for-like
# comparison against it. See docs/algorithm_details.md (Effective length and TPM).
FRAG_CAP = 1000


def _summarize_hist(stats, prefix, hist, cap=None):
    """Write {prefix}_{n,mean,sd,median,p10,p90,p99} into stats.

    Mirrors summarize_hist in the Rust core, including the quantile
    convention: the first length at which the cumulative weight reaches q*n.
    """
    items = [(length, w) for length, w in hist.items() if cap is None or length <= cap]
    n = math.fsum(w for _, w in items)
    if n <= 0:
        return

    mean = math.fsum(length * w for length, w in items) / n
    var = math.fsum((length - mean) ** 2 * w for length, w in items) / n

    lengths = sorted(length for length, _ in items)
    weights = dict(items)

    def quantile(q):
        target = q * n
        cum = 0.0
        for length in lengths:
            cum += weights[length]
            if cum >= target:
                return length
        return lengths[-1]

    stats[f"{prefix}_n"] = n
    stats[f"{prefix}_mean"] = mean
    stats[f"{prefix}_sd"] = math.sqrt(var)
    stats[f"{prefix}_median"] = quantile(0.50)
    stats[f"{prefix}_p10"] = quantile(0.10)
    stats[f"{prefix}_p90"] = quantile(0.90)
    stats[f"{prefix}_p99"] = quantile(0.99)


# ---------------------------------------------------------------------------
# Index builders — from annotation data, no GTF re-parsing needed
# ---------------------------------------------------------------------------


def build_exon_index_from_annotation(transcript_to_exons, string_to_int):
    """
    Build an ExonIntervalIndex from the transcript_to_exons dict produced
    during annotation precompute.

    Args:
        transcript_to_exons: {transcript_id_str: [{'chr': c, 'start': s, 'end': e, 'strand': str}, ...]}
        string_to_int: {transcript_id_str: int}

    Returns:
        ExonIntervalIndex ready for querying
    """
    logging.info("Building exon interval index from annotation data...")
    index = ExonIntervalIndex()

    for transcript_id, exons in transcript_to_exons.items():
        tx_int = string_to_int.get(transcript_id)
        if tx_int is None:
            continue
        for exon in exons:
            index.add_exon(
                exon["chr"], exon["start"], exon["end"], exon["strand"], tx_int
            )

    index.finalize()
    logging.info(f"  Exon index built: {index.n_exons:,} exons")
    return index


def build_junction_unstranded_lookup(junction_map, junction_int_to_string):
    """
    Build unstranded junction key -> transcript set lookup.

    The EqC builder uses unstranded junction keys (chr:donor-acceptor) because
    read strand is inferred from library prep, not from the junction itself.

    Args:
        junction_map: {junction_int: {'transcripts': [int], ...}}
        junction_int_to_string: list or dict mapping junction int -> string key

    Returns:
        dict: {unstranded_key_str: set(transcript_ints)}
    """
    logging.info("Building unstranded junction lookup...")
    junction_unstranded = defaultdict(set)

    for junc_int, info in junction_map.items():
        if isinstance(junction_int_to_string, list):
            junc_str = junction_int_to_string[junc_int]
        else:
            junc_str = junction_int_to_string[junc_int]
        # Strip strand suffix: "chr1:100-200:+" -> "chr1:100-200"
        unstranded = junc_str.rsplit(":", 1)[0]
        junction_unstranded[unstranded].update(info["transcripts"])

    result = dict(junction_unstranded)
    logging.info(f"  {len(result):,} unstranded junctions")
    return result


# ---------------------------------------------------------------------------
# BAM processing
# ---------------------------------------------------------------------------

# Coordinate offset calibration: pysam 0-based -> annotation key format
# Calibrated on LongBench data: donor+0, acceptor+1 gives 90.3% match rate
DONOR_OFFSET = 0
ACCEPTOR_OFFSET = 1


def _get_aligned_blocks_1based(read):
    """
    Extract aligned blocks from CIGAR as 1-based inclusive intervals.
    Considers M (0), = (7), X (8) as aligned operations.
    """
    blocks = []
    pos = read.reference_start  # 0-based
    for op, length in read.cigartuples:
        if op in (0, 7, 8):  # M, =, X
            blocks.append((pos + 1, pos + length))
            pos += length
        elif op == 2:  # D
            pos += length
        elif op == 3:  # N (splice)
            pos += length
        # I (1), S (4), H (5) don't consume reference
    return blocks


def _get_splice_junctions(read):
    """
    Extract splice junction keys from CIGAR N operations.
    Returns list of unstranded junction keys: "chr:donor-acceptor"
    """
    junctions = []
    chrom = read.reference_name
    pos = read.reference_start  # 0-based
    for op, length in read.cigartuples:
        if op in (0, 7, 8):
            pos += length
        elif op == 2:  # D
            pos += length
        elif op == 3:  # N - splice junction
            donor = pos + DONOR_OFFSET
            acceptor = pos + length + ACCEPTOR_OFFSET
            junctions.append(f"{chrom}:{donor}-{acceptor}")
            pos += length
    return junctions


def _infer_read_strand(read, strandedness):
    """
    Infer the transcript strand from read alignment and library strandedness.

    strandedness:
        0 = unstranded
        1 = stranded (read1 = transcript strand)
        2 = reversely stranded (read1 = reverse of transcript strand, most common)

    Returns '+', '-', or None (unstranded).
    """
    if strandedness == 0:
        return None

    is_read1 = not read.is_read2  # single-end reads count as read1
    is_reverse = read.is_reverse

    if strandedness == 1:
        if is_read1:
            return "-" if is_reverse else "+"
        else:
            return "+" if is_reverse else "-"
    elif strandedness == 2:
        if is_read1:
            return "+" if is_reverse else "-"
        else:
            return "-" if is_reverse else "+"
    return None


def process_bam(
    bam_path,
    exon_index,
    junction_unstranded,
    string_to_int,
    strandedness=2,
    paired_end=True,
    min_mapq=30,
    max_reads=None,
    use_alignment_scoring=False,
    as_temperature=10.0,
):
    """
    Single-pass BAM processing to build junction-aware equivalence classes.

    Returns:
        unique_eqc: dict of {tuple(sorted_feature_ints): read_count}
        unique_counts: dict of {feature_int: fractional_count}
        multi_eqc: dict of {tuple(sorted_feature_ints): read_count}
        multi_counts: dict of {feature_int: fractional_count}
        junction_counts: dict of {junction_key: count} (aggregate junction usage)
        stats: dict of processing statistics
    """
    if pysam is None:
        raise ImportError(
            "pysam is required for the junction-aware EqC builder. "
            "Install with: mamba install -c bioconda pysam"
        )

    bamfile = pysam.AlignmentFile(bam_path, "rb")

    # Unique mapper accumulators (streaming)
    unique_eqc = defaultdict(int)
    unique_counts = defaultdict(float)

    # Multimapper accumulators (need to buffer by read name)
    multi_read_features = defaultdict(
        set
    )  # read_name -> union of features (when AS off)
    multi_read_alignments = defaultdict(
        list
    )  # read_name -> [(features, AS)] (when AS on)

    # Junction count accumulator
    junction_agg_counts = defaultdict(int)

    # Stats
    stats = {
        "total_reads": 0,
        "unique_mapped": 0,
        "multimapped": 0,
        "unmapped": 0,
        "low_mapq_unique": 0,
        "no_feature": 0,
        "spliced_reads": 0,
        "junction_refined": 0,
        "junction_novel": 0,
        "junction_incompatible": 0,
        "exon_refined": 0,
        "exon_incompatible": 0,
        "unique_eqc_size_1": 0,
        "paired_reads_processed": 0,
        "paired_intersection_empty": 0,
        "orphaned_mates": 0,
        # Primaries whose multiplicity could not be read. Rust-side this is the
        # trigger for derive-by-name; here there is no such path, so it exists
        # only to make the degradation visible after the scan.
        "reads_without_nh": 0,
    }

    # For paired-end: buffer mate1 features until mate2 arrives. The genomic
    # bounds ride along so a fragment length can be measured once both mates
    # are in hand.
    if paired_end:
        mate_buffer = {}  # (name, HI) -> (features, is_unique, as, chrom, lo, hi)

    # Fragment length, measured in cDNA space (docs/algorithm_details.md); the Rust
    # core does this under measure_fragment_length=True, which is its default.
    frag_hist = defaultdict(float)
    frag_hist_unique = defaultdict(float)
    genomic_span_hist = defaultdict(float)
    frag_span_not_in_exons = 0

    # Read length sampling (first 10K reads, matches Rust behavior)
    read_length_samples = []
    read_length_sampling_done = False

    start_time = time.time()

    # until_eof=True streams the file linearly, the way the Rust parser does.
    # A bare fetch() walks the index instead, which stops before the trailing
    # unmapped block and left stats['unmapped'] reading 0 on BAMs holding a
    # million unmapped records. It also drops the .bai requirement, which the
    # Rust path never had.
    for read in bamfile.fetch(until_eof=True):
        if read.is_unmapped:
            stats["unmapped"] += 1
            continue

        if read.is_supplementary:
            continue

        is_secondary = read.is_secondary

        if not is_secondary:
            stats["total_reads"] += 1

            if not read_length_sampling_done:
                seq_len = read.query_length
                if seq_len and seq_len > 0:
                    read_length_samples.append(seq_len)
                if len(read_length_samples) >= 10_000:
                    read_length_sampling_done = True

            if max_reads and stats["total_reads"] > max_reads:
                break

            if stats["total_reads"] % 2_000_000 == 0:
                elapsed = time.time() - start_time
                rate = stats["total_reads"] / elapsed
                logging.info(
                    f"  {stats['total_reads']/1e6:.0f}M reads, {elapsed:.0f}s "
                    f"({rate/1000:.0f}K/s), "
                    f"unique EqC: {len(unique_eqc):,}, "
                    f"multi buffer: {len(multi_read_features):,}"
                )

        # Determine if unique or multimapper via NH tag
        try:
            nh = read.get_tag("NH")
        except KeyError:
            nh = 1  # assume unique if no NH tag
            # Counted on primaries only, so the total is comparable to
            # stats['total_reads'] without secondaries inflating it.
            if not is_secondary:
                stats["reads_without_nh"] += 1

        is_unique = nh == 1

        if is_secondary and is_unique:
            continue

        if use_alignment_scoring or not is_unique:
            try:
                as_score = read.get_tag("AS")
            except KeyError:
                as_score = 0
        else:
            as_score = 0

        try:
            hi_tag = read.get_tag("HI")
        except KeyError:
            hi_tag = 0

        if not read.cigartuples:
            continue

        chrom = read.reference_name

        # Infer strand
        tx_strand = _infer_read_strand(read, strandedness)

        # Get aligned blocks and splice junctions
        aligned_blocks = _get_aligned_blocks_1based(read)
        splice_junctions = _get_splice_junctions(read)

        if splice_junctions:
            stats["spliced_reads"] += 1

        # Query exon overlap (inclusion — broad candidate set)
        exon_overlap_txs = exon_index.query_blocks(chrom, aligned_blocks, tx_strand)

        if not exon_overlap_txs:
            stats["no_feature"] += 1
            continue

        # Exon containment filtering (exclusion — stricter than overlap)
        contained_txs = exon_index.query_blocks_contained(
            chrom, aligned_blocks, tx_strand
        )
        has_containment = bool(contained_txs)

        # Query junction compatibility
        junction_compatible_txs = None
        if splice_junctions:
            junc_sets = []
            for jk in splice_junctions:
                if jk in junction_unstranded:
                    junc_sets.append(junction_unstranded[jk])
                    junction_agg_counts[jk] += 1
            if junc_sets:
                junction_compatible_txs = set.intersection(*junc_sets)
            else:
                stats["junction_novel"] += 1

        # Refined assignment: try containment ∩ junctions first, then fallbacks
        if junction_compatible_txs is not None:
            if has_containment:
                # Best case: containment ∩ junctions ∩ overlap
                triple = contained_txs & junction_compatible_txs & exon_overlap_txs
                if triple:
                    if len(triple) < len(exon_overlap_txs):
                        junc_only = exon_overlap_txs & junction_compatible_txs
                        if len(junc_only) < len(exon_overlap_txs):
                            stats["junction_refined"] += 1
                        if len(triple) < len(junc_only):
                            stats["exon_refined"] += 1
                    features = triple
                else:
                    # Triple empty — try overlap ∩ junctions
                    junc_refined = exon_overlap_txs & junction_compatible_txs
                    if junc_refined:
                        if len(junc_refined) < len(exon_overlap_txs):
                            stats["junction_refined"] += 1
                        stats["exon_incompatible"] += 1
                        features = junc_refined
                    else:
                        # Try overlap ∩ containment
                        cont_refined = exon_overlap_txs & contained_txs
                        if cont_refined:
                            stats["junction_incompatible"] += 1
                            if len(cont_refined) < len(exon_overlap_txs):
                                stats["exon_refined"] += 1
                            features = cont_refined
                        else:
                            stats["junction_incompatible"] += 1
                            stats["exon_incompatible"] += 1
                            features = exon_overlap_txs
            else:
                # No containment candidates — junction-only refinement
                junc_refined = exon_overlap_txs & junction_compatible_txs
                if junc_refined:
                    if len(junc_refined) < len(exon_overlap_txs):
                        stats["junction_refined"] += 1
                    features = junc_refined
                else:
                    stats["junction_incompatible"] += 1
                    features = exon_overlap_txs
        elif has_containment:
            # No junctions — containment-only refinement
            cont_refined = exon_overlap_txs & contained_txs
            if cont_refined:
                if len(cont_refined) < len(exon_overlap_txs):
                    stats["exon_refined"] += 1
                features = cont_refined
            else:
                stats["exon_incompatible"] += 1
                features = exon_overlap_txs
        else:
            # No containment, no junctions — overlap only (fallback)
            features = exon_overlap_txs

        # Handle paired-end: combine features from both mates
        # Use (read_name, HI) as key so mates at different loci pair separately
        # Outer genomic bounds of this alignment. aligned_blocks is ascending.
        read_lo = aligned_blocks[0][0]
        read_hi = aligned_blocks[-1][1]

        # Set when this record completes a pair on one chromosome, which is the
        # only situation in which a fragment length is measurable.
        pair_span = None

        if paired_end and read.is_paired:
            read_name = read.query_name
            mate_key = (read_name, hi_tag)
            if mate_key in mate_buffer:
                (
                    mate_features,
                    mate_is_unique,
                    mate_as,
                    mate_chrom,
                    mate_lo,
                    mate_hi,
                ) = mate_buffer.pop(mate_key)
                combined = features & mate_features
                if combined:
                    features = combined
                else:
                    features = features | mate_features
                    # Count pairs, not alignment records — same convention as
                    # paired_reads_processed below, and as the Rust builder.
                    if not is_secondary:
                        stats["paired_intersection_empty"] += 1
                is_unique = is_unique and mate_is_unique
                as_score = as_score + mate_as
                if not is_secondary:
                    stats["paired_reads_processed"] += 1
                if mate_chrom == chrom:
                    pair_span = (min(mate_lo, read_lo), max(mate_hi, read_hi))
            else:
                mate_buffer[mate_key] = (
                    features,
                    is_unique,
                    as_score,
                    chrom,
                    read_lo,
                    read_hi,
                )
                continue

        # Route to unique or multi accumulator
        if is_unique:
            if read.mapping_quality < min_mapq:
                stats["low_mapq_unique"] += 1
                continue

            if pair_span is not None:
                lo, hi = pair_span
                # Spread the pair over every transcript it is compatible with,
                # 1/k each. The obvious alternative — keep only pairs
                # compatible with exactly one transcript — is a
                # *length-dependent* filter: a longer fragment spans more
                # structure, so it is likelier to become uniquely assignable,
                # and conditioning on uniqueness over-samples long fragments
                # (+8% on the mean, keeps ~18% of pairs). Salmon and RSEM
                # spread under a uniform prior instead, and matching them is
                # what puts this estimator on their answer. Transcripts that
                # cannot produce the span drop out and the weight renormalises
                # over the rest.
                placed = []
                for tx in features:
                    flen = exon_index.spliced_distance(tx, lo, hi)
                    if flen is not None:
                        placed.append(flen)
                if placed:
                    w = 1.0 / len(placed)
                    for flen in placed:
                        frag_hist[flen] += w

                # Diagnostics on the unique-transcript subset: the biased
                # estimator above, and the same pairs measured as an unspliced
                # genomic span (= what TLEN reports, and what v1 used).
                # Accumulated on identical pairs, so the three-way comparison
                # isolates the estimator from the read set.
                if len(features) == 1:
                    tx = next(iter(features))
                    flen = exon_index.spliced_distance(tx, lo, hi)
                    if flen is not None:
                        frag_hist_unique[flen] += 1
                        genomic_span_hist[hi - lo + 1] += 1
                    else:
                        frag_span_not_in_exons += 1

            stats["unique_mapped"] += 1
            features_tuple = tuple(sorted(features))

            unique_eqc[features_tuple] += 1

            fraction = 1.0 / len(features)
            for f in features:
                unique_counts[f] += fraction

            if len(features) == 1:
                stats["unique_eqc_size_1"] += 1
        else:
            if not is_secondary:
                stats["multimapped"] += 1
            read_name = read.query_name
            if use_alignment_scoring:
                multi_read_alignments[read_name].append((features, as_score))
            else:
                multi_read_features[read_name].update(features)

    bamfile.close()

    # Process orphaned mates (mate never found — count them solo)
    if paired_end and mate_buffer:
        stats["orphaned_mates"] = len(mate_buffer)
        logging.info(f"  Processing {len(mate_buffer):,} orphaned mates...")
        for (read_name, _hi), (
            features,
            orphan_is_unique,
            orphan_as,
            _chrom,
            _lo,
            _hi_pos,
        ) in mate_buffer.items():
            features_tuple = tuple(sorted(features))
            if orphan_is_unique:
                unique_eqc[features_tuple] += 1
                fraction = 1.0 / len(features)
                for f in features:
                    unique_counts[f] += fraction
                stats["unique_mapped"] += 1
                if len(features) == 1:
                    stats["unique_eqc_size_1"] += 1
            else:
                if use_alignment_scoring:
                    multi_read_alignments[read_name].append((features, orphan_as))
                else:
                    multi_read_features[read_name].update(features)
        del mate_buffer

    elapsed = time.time() - start_time
    logging.info(
        f"  BAM scan complete: {stats['total_reads']:,} reads in {elapsed:.0f}s "
        f"({stats['total_reads']/elapsed/1000:.0f}K/s)"
    )

    # NH is the only thing this builder has to decide multiplicity with, and
    # minimap2 never writes it. Absent NH every read reads as unique and every
    # secondary is dropped at the `is_secondary and is_unique` test above, which
    # is quietly wrong rather than loudly wrong. The Rust core detects this and
    # derives multiplicity from the read name instead; there is no such path
    # here, so all this can do is say so.
    #
    # Thresholded rather than fired on the first missing tag: a stray record
    # without NH is not the same condition as an aligner that never emits one.
    if stats["reads_without_nh"] > 0.01 * max(stats["total_reads"], 1):
        frac = stats["reads_without_nh"] / max(stats["total_reads"], 1)
        logging.warning(
            f"  {stats['reads_without_nh']:,} primary alignments ({frac:.0%}) carry "
            "no NH tag. Every one was counted as a unique mapper and their secondary "
            "alignments were discarded, so the multimapper EM has nothing to resolve. "
            "The usual cause is a long-read aligner such as minimap2. Install "
            "majec_core and pass --derive_multiplicity on to recover multiplicity "
            "from the read name instead."
        )

    # Compute median read length from sampled reads
    if read_length_samples:
        read_length_samples.sort()
        stats["median_read_length"] = read_length_samples[len(read_length_samples) // 2]

    # Build multimapper equivalence classes
    multi_eqc = defaultdict(float)
    multi_counts = defaultdict(float)

    if use_alignment_scoring:
        logging.info(
            f"  Building AS-weighted multimapper EqCs from {len(multi_read_alignments):,} reads..."
        )
        for read_name, alignments in multi_read_alignments.items():
            if not alignments:
                continue
            if len(alignments) == 1:
                feats, _ = alignments[0]
                features_tuple = tuple(sorted(feats))
                multi_eqc[features_tuple] += 1.0
                fraction = 1.0 / len(feats)
                for f in feats:
                    multi_counts[f] += fraction
            else:
                max_as = max(s for _, s in alignments)
                weights = [
                    math.exp((s - max_as) / as_temperature) for _, s in alignments
                ]
                total_weight = sum(weights)
                for i, (feats, _) in enumerate(alignments):
                    w = weights[i] / total_weight
                    features_tuple = tuple(sorted(feats))
                    multi_eqc[features_tuple] += w
                    fraction = w / len(feats)
                    for f in feats:
                        multi_counts[f] += fraction
        del multi_read_alignments
    else:
        logging.info(
            f"  Building multimapper EqCs from {len(multi_read_features):,} reads..."
        )
        for read_name, feature_set in multi_read_features.items():
            features_tuple = tuple(sorted(feature_set))
            multi_eqc[features_tuple] += 1.0
            fraction = 1.0 / len(feature_set)
            for f in feature_set:
                multi_counts[f] += fraction
        del multi_read_features

    stats["unique_eqc_count"] = len(unique_eqc)
    stats["multi_eqc_count"] = len(multi_eqc)
    stats["total_unique_reads_in_eqc"] = sum(unique_eqc.values())
    stats["total_multi_reads_in_eqc"] = sum(multi_eqc.values())

    # Fragment length. Same keys, same order of derivation as the Rust core's
    # BamStats::to_dict, so a consumer cannot tell which parser produced them.
    # Absent entirely on a single-end library, where nothing is measurable.
    if frag_hist:
        stats["fragment_length_hist"] = dict(frag_hist)
        _summarize_hist(stats, "fragment_length", frag_hist)
        _summarize_hist(stats, "fragment_length_capped", frag_hist, cap=FRAG_CAP)
    if frag_hist_unique:
        _summarize_hist(stats, "fragment_length_unique", frag_hist_unique)
    if genomic_span_hist:
        _summarize_hist(stats, "genomic_span", genomic_span_hist)
    stats["fragment_span_not_in_exons"] = frag_span_not_in_exons

    return (
        dict(unique_eqc),
        dict(unique_counts),
        dict(multi_eqc),
        dict(multi_counts),
        dict(junction_agg_counts),
        stats,
    )


# ---------------------------------------------------------------------------
# Cache adapter functions — convert raw EqC data to pipeline format
# ---------------------------------------------------------------------------


def build_junction_metrics_from_counts(
    junction_counts, junction_map, junction_int_to_string, junction_string_to_int
):
    """
    Build pipeline-compatible junction_metrics from aggregated junction_counts
    (unstranded key -> count).

    Replicates the logic of apply_junction_evidence() in junctions.py but works
    from aggregated counts instead of per-line .jcounts data.
    """
    junction_metrics = defaultdict(
        lambda: {
            "junction_read_count": 0,
            "unique_junction_reads": 0,
            "shared_junction_reads": 0,
            "observed_junctions": {},
            "n_unique_junctions": 0,
            "n_shared_junctions": 0,
        }
    )

    matched = 0
    unmatched = 0

    for unstranded_key, count in junction_counts.items():
        if count == 0:
            continue

        # Try both stranded versions
        found = False
        for strand in ["+", "-"]:
            stranded_key = f"{unstranded_key}:{strand}"
            junction_id = junction_string_to_int.get(stranded_key, -1)

            if junction_id != -1 and junction_id in junction_map:
                info = junction_map[junction_id]
                n_transcripts = info.get("n_transcripts", 1)

                for transcript in info["transcripts"]:
                    metrics = junction_metrics[transcript]
                    metrics["junction_read_count"] += count

                    if n_transcripts == 1:
                        metrics["unique_junction_reads"] += count
                    else:
                        metrics["shared_junction_reads"] += count

                    if junction_id not in metrics["observed_junctions"]:
                        metrics["observed_junctions"][junction_id] = 0
                        if n_transcripts == 1:
                            metrics["n_unique_junctions"] += 1
                        else:
                            metrics["n_shared_junctions"] += 1

                    metrics["observed_junctions"][junction_id] += count

                found = True
                matched += 1
                break  # Don't double-count on both strands

        if not found:
            unmatched += 1

    logging.info(
        f"  Junction metrics: {matched:,} matched, {unmatched:,} unmatched junctions"
    )
    logging.info(f"  Built metrics for {len(junction_metrics):,} transcripts")

    return dict(junction_metrics)


def build_initial_counts(unique_counts, n_features):
    """
    Build initial_unique_counts pd.Series from fractional counts dict.
    Index is integer transcript IDs, values are fractional counts.
    """
    counts = pd.Series(0.0, index=np.arange(n_features), dtype=np.float64)
    for feat_int, count in unique_counts.items():
        if feat_int < n_features:
            counts.iloc[feat_int] = count
    return counts


def convert_multi_groups_to_frozenset(multi_eqc):
    """
    Convert multimapper groups from tuple keys to frozenset keys
    (pipeline cache format).
    """
    converted = {}
    for key_tuple, count in multi_eqc.items():
        converted[frozenset(key_tuple)] = count
    return converted


def build_multimapper_counts(multi_eqc, n_features):
    """
    Build initial_multimapper_counts pd.Series from multi EqC groups.
    multi_eqc keys can be tuples or frozensets.
    """
    counts = pd.Series(0.0, index=np.arange(n_features), dtype=np.float64)
    for feature_set, read_count in multi_eqc.items():
        fraction = read_count / len(feature_set)
        for feat in feature_set:
            if feat < n_features:
                counts.iloc[feat] += fraction
    return counts


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def run_junction_eqc_builder(
    bam_path,
    exon_index,
    junction_unstranded,
    string_to_int,
    junction_map,
    junction_int_to_string,
    junction_string_to_int,
    strandedness=2,
    paired_end=True,
    min_mapq=30,
    sample_name=None,
    use_alignment_scoring=False,
    as_temperature=10.0,
):
    """
    High-level entry point: BAM → standard pipeline payload.

    Single call replaces the 3-pass featureCounts flow (unique FIFO +
    samtools filter + multi FIFO) and .jcounts parsing.

    Args:
        bam_path: Path to coordinate-sorted, indexed BAM file
        exon_index: ExonIntervalIndex (precomputed from annotation)
        junction_unstranded: dict of {unstranded_key: set(transcript_ints)}
        string_to_int: {transcript_id_str: int}
        junction_map: {junction_int: {'transcripts': [int], ...}}
        junction_int_to_string: list or dict mapping junction int -> string
        junction_string_to_int: {junction_string: int}
        strandedness: 0=unstranded, 1=stranded, 2=reverse-stranded
        paired_end: Whether data is paired-end
        min_mapq: Minimum MAPQ for unique mappers
        sample_name: Optional sample name for logging

    Returns:
        dict with keys:
            'unique_mapper_groups': {tuple: count}
            'initial_unique_counts': pd.Series
            'multimapper_classes': {frozenset: count}
            'initial_multimapper_counts': pd.Series
            'junction_metrics': dict
            'has_multimapper_data': True
            'stats': dict
    """
    log_prefix = f"[{sample_name}] " if sample_name else ""
    logging.info(f"{log_prefix}Starting junction-aware EqC builder on {bam_path}")

    if use_alignment_scoring:
        logging.info(
            f"{log_prefix}AS-weighted multimapper scoring enabled (tau={as_temperature})"
        )

    # 1. Single-pass BAM scan
    unique_eqc, unique_counts, multi_eqc, multi_counts, junction_counts, stats = (
        process_bam(
            bam_path,
            exon_index,
            junction_unstranded,
            string_to_int,
            strandedness=strandedness,
            paired_end=paired_end,
            min_mapq=min_mapq,
            use_alignment_scoring=use_alignment_scoring,
            as_temperature=as_temperature,
        )
    )

    n_features = len(string_to_int)

    # 2. Build junction_metrics from aggregated junction counts
    logging.info(f"{log_prefix}Building junction metrics...")
    junction_metrics = build_junction_metrics_from_counts(
        junction_counts, junction_map, junction_int_to_string, junction_string_to_int
    )

    # 3. Build initial unique counts as Series
    initial_unique_counts = build_initial_counts(unique_counts, n_features)

    # 4. Build multimapper data
    multimapper_classes = convert_multi_groups_to_frozenset(multi_eqc)
    initial_multimapper_counts = build_multimapper_counts(
        multimapper_classes, n_features
    )

    logging.info(
        f"{log_prefix}EqC builder complete: "
        f"{len(unique_eqc):,} unique classes ({stats['total_unique_reads_in_eqc']:,} reads), "
        f"{len(multimapper_classes):,} multi classes ({stats['total_multi_reads_in_eqc']:,} reads), "
        f"{len(junction_metrics):,} transcripts with junction metrics"
    )

    return {
        "unique_mapper_groups": unique_eqc,
        "initial_unique_counts": initial_unique_counts,
        "multimapper_classes": multimapper_classes,
        "initial_multimapper_counts": initial_multimapper_counts,
        "junction_metrics": junction_metrics,
        "has_multimapper_data": True,
        "stats": stats,
    }
