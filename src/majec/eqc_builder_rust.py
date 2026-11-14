"""
Python wrapper for the Rust-based EqC builder (majec_core).

Drop-in replacement for eqc_builder.run_junction_eqc_builder().
Falls back to the pure-Python pysam implementation if the Rust module
is not installed.

Usage in pipeline.py:
    # Replace:
    #   from majec.eqc_builder import run_junction_eqc_builder
    # With:
    #   from majec.eqc_builder_rust import run_junction_eqc_builder
"""

import logging
import time

import numpy as np
import pandas as pd

try:
    import majec_core

    RUST_AVAILABLE = True
except ImportError:
    majec_core = None
    RUST_AVAILABLE = False


def builder_provenance():
    """Describe the EqC builder this process will use, for the log and the run manifest.

    majec_core is a separately installed compiled distribution. It does not
    track the Python source, so the majec version alone cannot say which build
    produced a run. Record the distribution version and a hash of the loaded
    binary, or the pysam version when the pure-Python fallback is in use.
    """
    if not RUST_AVAILABLE:
        try:
            import pysam

            pysam_version = pysam.__version__
        except ImportError:
            pysam_version = None
        return {"builder": "python (pysam fallback)", "pysam_version": pysam_version}

    import glob
    import hashlib
    import os
    from importlib import metadata

    try:
        version = metadata.version("majec-core")
    except metadata.PackageNotFoundError:
        version = getattr(majec_core, "__version__", None)
    path = majec_core.__file__ or ""
    if path.endswith((".so", ".pyd")):
        binary = path
    else:
        pkg_dir = os.path.dirname(path)
        found = sorted(
            glob.glob(os.path.join(pkg_dir, "*.so"))
            + glob.glob(os.path.join(pkg_dir, "*.pyd"))
        )
        binary = found[0] if found else None
    digest = None
    if binary:
        h = hashlib.sha256()
        with open(binary, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
    return {
        "builder": "rust (majec_core)",
        "majec_core_version": version,
        "majec_core_binary": binary,
        "majec_core_sha256": digest,
    }


# Long-read multiplicity handling lives entirely in the Rust core: deriving
# multiplicity by read name needs a whole-file alignment buffer and an AS filter
# that the pysam builder does not implement. Rather than accept the options and
# quietly return the pre-0.2.0 answer — every read unique, every secondary
# discarded — the fallback refuses them.
#
# 'off' and 'auto' pass through. 'off' is exactly what the pysam builder does,
# and 'auto' cannot probe from here, so it degrades to 'off'; the pysam scan
# warns on its own if the BAM turns out to carry no NH.
def _reject_lr_options(derive_multiplicity, lr_score_ratio, lr_min_mapq):
    asked = []
    if derive_multiplicity is True:
        asked.append("--derive_multiplicity on")
    if lr_score_ratio:
        asked.append(f"--lr_score_ratio {lr_score_ratio}")
    if lr_min_mapq:
        asked.append(f"--lr_min_mapq {lr_min_mapq}")
    if asked:
        raise RuntimeError(
            f"{', '.join(asked)} requires the compiled majec_core extension, "
            "which is not installed. The pure-Python (pysam) builder reads "
            "multiplicity off the NH tag only and cannot honour these options. "
            "Install majec_core, or drop them and accept that reads without NH "
            "will all be counted as unique mappers."
        )


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
    derive_multiplicity=None,
    lr_score_ratio=0.0,
    lr_min_mapq=None,
):
    """
    High-level entry point: BAM -> standard pipeline payload.

    Drop-in replacement for eqc_builder.run_junction_eqc_builder().
    If the Rust module is available and exon_index is a Rust ExonIntervalIndex,
    uses the fast Rust path. Otherwise falls back to pure Python.

    Returns:
        dict with keys matching the Python version:
            'unique_mapper_groups': {tuple: count}
            'initial_unique_counts': pd.Series
            'multimapper_classes': {frozenset: count}
            'initial_multimapper_counts': pd.Series
            'junction_metrics': dict
            'has_multimapper_data': True
            'stats': dict
    """
    # Fall back to Python if Rust not available
    if not RUST_AVAILABLE:
        _reject_lr_options(derive_multiplicity, lr_score_ratio, lr_min_mapq)
        logging.warning("majec_core not installed, falling back to Python eqc_builder")
        from majec.eqc_builder import run_junction_eqc_builder as _python_impl

        return _python_impl(
            bam_path,
            exon_index,
            junction_unstranded,
            string_to_int,
            junction_map,
            junction_int_to_string,
            junction_string_to_int,
            strandedness=strandedness,
            paired_end=paired_end,
            min_mapq=min_mapq,
            sample_name=sample_name,
            use_alignment_scoring=use_alignment_scoring,
            as_temperature=as_temperature,
        )

    log_prefix = f"[{sample_name}] " if sample_name else ""
    logging.info(f"{log_prefix}Starting Rust junction-aware EqC builder on {bam_path}")
    if use_alignment_scoring:
        logging.info(
            f"{log_prefix}AS-weighted multimapper scoring enabled (tau={as_temperature})"
        )

    # Convert Python ExonIntervalIndex to Rust if needed
    if not isinstance(exon_index, majec_core.ExonIntervalIndex):
        logging.info(f"{log_prefix}Converting Python ExonIntervalIndex to Rust...")
        exon_records = []
        for (chrom, strand_val), bin_map in exon_index.bins.items():
            strand = strand_val if isinstance(strand_val, str) else chr(strand_val)
            seen = set()
            for bin_idx, intervals in bin_map.items():
                for start, end, tx_int in intervals:
                    key = (tx_int, chrom, start, end, strand)
                    if key not in seen:
                        seen.add(key)
                        exon_records.append(key)
        exon_index = majec_core.ExonIntervalIndex.from_annotation(exon_records)
        logging.info(
            f"{log_prefix}  Rust exon index built: {exon_index.n_exons:,} exons"
        )

    # Ensure junction_unstranded values are lists (Rust expects Vec<u32>)
    junc_for_rust = {}
    for k, v in junction_unstranded.items():
        if isinstance(v, set):
            junc_for_rust[k] = list(v)
        else:
            junc_for_rust[k] = v

    start_time = time.time()

    # Call Rust process_bam
    raw = majec_core.process_bam(
        bam_path,
        exon_index,
        junc_for_rust,
        strandedness=strandedness,
        paired_end=paired_end,
        min_mapq=min_mapq,
        use_alignment_scoring=use_alignment_scoring,
        as_temperature=as_temperature,
        derive_multiplicity=derive_multiplicity,
        lr_score_ratio=lr_score_ratio,
        lr_min_mapq=lr_min_mapq,
    )

    bam_elapsed = time.time() - start_time
    stats = raw["stats"]
    logging.info(
        f"{log_prefix}Rust BAM scan: {stats['total_reads']:,} reads in {bam_elapsed:.1f}s "
        f"({stats['total_reads']/bam_elapsed/1000:.0f}K/s)"
    )

    # No-NH (long-read) mode. Reported unconditionally when it fires, because it
    # changes what "unique" means for the whole run and the pipeline is otherwise
    # silent about having switched.
    if stats.get("multiplicity_derived"):
        buffered = stats.get("lr_reads_buffered", 0)
        n_uniq = stats.get("lr_reads_unique", 0)
        n_multi = stats.get("lr_reads_multi", 0)
        dropped = stats.get("lr_alignments_dropped", 0)
        rescued = stats.get("lr_multi_rescued_low_mapq", 0)
        pct = lambda x: f"{x/buffered*100:.1f}%" if buffered else "n/a"
        # An AS filter is off by default: it decides how DECISIVE the builder is,
        # not how careful, and tightening it converts near-ties into spurious
        # uniques. Say so rather than printing "0 alignments below 0% of best AS".
        as_note = (
            f"{dropped:,} alignments below {lr_score_ratio:.0%} of best AS"
            if lr_score_ratio > 0
            else "AS filter off, every reported alignment kept as a real locus"
        )
        logging.info(
            f"{log_prefix}Multiplicity derived by read name (no NH tag): "
            f"{buffered:,} reads — {n_uniq:,} unique ({pct(n_uniq)}), "
            f"{n_multi:,} multi ({pct(n_multi)}); {as_note}"
        )
        if rescued:
            logging.info(
                f"{log_prefix}  {rescued:,} multireads ({pct(rescued)}) have every "
                f"alignment below MAPQ {min_mapq} — these reach the EM here, "
                f"and would have been deleted as low-MAPQ uniques otherwise"
            )
    spliced = stats.get("spliced_reads", 0)
    if spliced > 0:
        refined = stats.get("junction_refined", 0)
        novel = stats.get("junction_novel", 0)
        incompat = stats.get("junction_incompatible", 0)
        logging.info(
            f"{log_prefix}Junction stats: {spliced:,} spliced reads — "
            f"{refined:,} refined ({refined/spliced*100:.1f}%), "
            f"{novel:,} novel ({novel/spliced*100:.1f}%), "
            f"{incompat:,} incompatible ({incompat/spliced*100:.1f}%)"
        )

    # Exon containment stats (relative to all assigned reads = unique + multi)
    assigned = stats.get("unique_mapped", 0) + stats.get("multimapped", 0)
    exon_ref = stats.get("exon_refined", 0)
    exon_inc = stats.get("exon_incompatible", 0)
    if assigned > 0 and (exon_ref > 0 or exon_inc > 0):
        logging.info(
            f"{log_prefix}Containment stats: {assigned:,} assigned reads — "
            f"{exon_ref:,} refined ({exon_ref/assigned*100:.1f}%), "
            f"{exon_inc:,} incompatible ({exon_inc/assigned*100:.1f}%)"
        )

    # Fragment length distribution, measured in cDNA space during the same scan.
    # Replaces tools/fragment_lengths.py, which read TLEN off a genome BAM and so
    # counted intron length inside any pair whose mate straddled an intron.
    n_frag = stats.get("fragment_length_n", 0)
    if n_frag:
        logging.info(
            f"{log_prefix}Fragment length: n={n_frag:,.0f}, "
            f"mean={stats['fragment_length_mean']:.1f}, "
            f"sd={stats['fragment_length_sd']:.1f}, "
            f"median={stats['fragment_length_median']}, "
            f"p90={stats['fragment_length_p90']}, p99={stats['fragment_length_p99']}"
        )
        # The <=1000 bp view is the one comparable to Salmon's fld and RSEM's
        # .model, both of which stop at 1000.
        logging.info(
            f"{log_prefix}  capped at 1000bp: "
            f"mean={stats['fragment_length_capped_mean']:.1f}, "
            f"sd={stats['fragment_length_capped_sd']:.1f} "
            f"({n_frag - stats['fragment_length_capped_n']:,.0f} longer, "
            f"{stats.get('fragment_span_not_in_exons', 0):,} unique-pair spans "
            f"unplaceable in exons)"
        )

    n_features = len(string_to_int)
    unique_eqc = raw["unique_eqc"]
    unique_counts_dict = raw["unique_counts"]
    multi_eqc = raw["multi_eqc"]
    junction_counts = raw["junction_counts"]

    # Build junction_metrics (still in Python for now — runs once, not per-read)
    from majec.eqc_builder import build_junction_metrics_from_counts

    logging.info(f"{log_prefix}Building junction metrics...")
    junction_metrics = build_junction_metrics_from_counts(
        junction_counts, junction_map, junction_int_to_string, junction_string_to_int
    )

    # Build initial unique counts as Series
    initial_unique_counts = pd.Series(
        0.0, index=np.arange(n_features), dtype=np.float64
    )
    for feat_int, count in unique_counts_dict.items():
        if feat_int < n_features:
            initial_unique_counts.iloc[feat_int] = count

    # Build multimapper data
    multimapper_classes = {frozenset(k): v for k, v in multi_eqc.items()}
    initial_multimapper_counts = pd.Series(
        0.0, index=np.arange(n_features), dtype=np.float64
    )
    for feature_set, read_count in multimapper_classes.items():
        fraction = read_count / len(feature_set)
        for feat in feature_set:
            if feat < n_features:
                initial_multimapper_counts.iloc[feat] += fraction

    logging.info(
        f"{log_prefix}EqC builder complete: "
        f"{len(unique_eqc):,} unique classes ({stats.get('total_unique_reads_in_eqc', 0):,} reads), "
        f"{len(multimapper_classes):,} multi classes ({stats.get('total_multi_reads_in_eqc', 0):,} reads), "
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
