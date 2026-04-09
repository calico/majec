#!/usr/bin/env python
# AF_TE_pipeline_v32_optimized.py - Integer ID mapping version
# Key optimization: Replace string transcript IDs with integers to reduce memory usage

import argparse
import subprocess
import pandas as pd
import os
import sys
import tempfile
from collections import defaultdict
import multiprocessing
import time
import shutil
import json
import numpy as np
import gc
import logging
import shlex
import pickle
import threading
import traceback
import gzip
from pathlib import Path
from datetime import datetime, timezone

# --- Helper Functions ---

from majec.utils.confidence_utils import (
    calculate_confidence_metrics,
    calculate_differential_confidence_parallel,
    track_assignment_entropy_parallel,
    process_and_save_sample_outputs,
    analyze_competition_patterns,
    decode_transcript_ids,
    decode_dataframe_columns,
)
from majec.utils.group_confidence import (
    run_group_confidence_from_pipeline_parallel_optimized,
)

from majec.utils.junctions import (
    parse_junctions_if_requested,
    calculate_junction_confidence_metrics_vectorized,
    calculate_junction_evidence_from_metrics,
    calculate_junction_completeness_position_aware_optimized,
    calculate_subset_penalties_holistic,
    calculate_territory_adjustment_factors,
)

from majec.utils.cache import PipelineCache

from majec.utils.pipeline_utils import (
    setup_logging,
    safe_subprocess_run,
    standard_em_step,
    grouped_momentum_acceleration,
    adaptive_convergence_check,
    load_precomputed_junction_map,
    get_sample_name_from_bam,
    calculate_effective_length_distributional,
)

# Global variables for multiprocessing workers
_int_to_string_map = None
_string_to_int_map = None
_junction_int_to_string_map = None
_junction_string_to_int_map = None
_junction_map = None
_transcript_junction_order = None
_transcript_info = None
_subset_relationships = None
_splice_competitor_map = None
_transcript_expected_junctions = None
_gene_map = None
_te_map = None


def init_worker(
    int_to_string_map,
    string_to_int_map,
    junction_int_to_string_map,
    junction_string_to_int_map,
    junction_map,
    transcript_junction_order,
    transcript_info,
    subset_relationships,
    subset_coverage_territory_mapping,
    splice_competitor_map,
    transcript_expected_junctions,
    gene_map,
    te_map,
):
    """Initialize worker process with ALL read-only mappings."""
    global _int_to_string_map, _string_to_int_map
    global _junction_int_to_string_map, _junction_string_to_int_map
    global _junction_map, _transcript_junction_order, _transcript_info
    global _subset_relationships, _subset_coverage_territory_mapping, _splice_competitor_map
    global _transcript_expected_junctions, _gene_map, _te_map

    _int_to_string_map = int_to_string_map
    _string_to_int_map = string_to_int_map
    _junction_int_to_string_map = junction_int_to_string_map
    _junction_string_to_int_map = junction_string_to_int_map
    _junction_map = junction_map
    _transcript_junction_order = transcript_junction_order
    _transcript_info = transcript_info
    _subset_relationships = subset_relationships
    _subset_coverage_territory_mapping = subset_coverage_territory_mapping
    _splice_competitor_map = splice_competitor_map
    _transcript_expected_junctions = transcript_expected_junctions
    _gene_map = gene_map
    _te_map = te_map


def init_worker_downstream(
    int_to_string_map,
    string_to_int_map,
    gene_map,
    te_map,
    junction_int_to_string_map,
    transcript_info,
):
    """Initialize worker with data for EM, confidence, and final report writing."""
    global _int_to_string_map, _string_to_int_map, _gene_map, _te_map, _junction_int_to_string_map, _transcript_info

    _int_to_string_map = int_to_string_map
    _string_to_int_map = string_to_int_map
    _gene_map = gene_map
    _te_map = te_map
    _junction_int_to_string_map = junction_int_to_string_map
    _transcript_info = transcript_info


#####. string -> int & int -> string mapping functions #####
def encode_transcript_ids(data, string_to_int, copy=True):
    """
    Convert string transcript IDs to integers in various data structures.

    Args:
        data: Can be DataFrame, Series, dict, or dict of dicts
        string_to_int: Mapping from string IDs to integers
        copy: Whether to copy the data structure

    Returns:
        Data structure with integer IDs
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy() if copy else data
        if df.index.dtype == "object":  # String index
            df.index = df.index.map(lambda x: string_to_int.get(x, -1))
        return df

    elif isinstance(data, pd.Series):
        s = data.copy() if copy else data
        if s.index.dtype == "object":  # String index
            s.index = s.index.map(lambda x: string_to_int.get(x, -1))
        s.name = data.name
        return s

    elif isinstance(data, dict):
        # Check if it's a nested dict (like unique_mapper_groups)
        first_key = next(iter(data.keys())) if data else None
        if first_key and isinstance(first_key, tuple):
            # Dict with tuple keys (equivalence classes)
            encoded = {}
            for feature_tuple, value in data.items():
                encoded_tuple = tuple(string_to_int.get(f, -1) for f in feature_tuple)
                encoded[encoded_tuple] = value
            return encoded
        else:
            # Simple dict
            return {string_to_int.get(k, -1): v for k, v in data.items()}

    return data


def _aggregate_by_map(df, id_map):
    """Aggregate a DataFrame's rows using a LocusID -> AggregateID mapping."""
    mask = df.index.isin(id_map["LocusID"].values)
    if mask.sum() == 0:
        return pd.DataFrame()
    subset = df[mask].copy()
    locus_to_agg = id_map.set_index("LocusID")["AggregateID"].to_dict()
    subset.index = subset.index.map(locus_to_agg)
    return subset.groupby(level=0).sum()


def _aggregate_genes_and_tes(df, gene_map, te_map):
    """Aggregate a DataFrame by gene and TE maps, returning combined result."""
    genes_agg = (
        _aggregate_by_map(df, gene_map) if gene_map is not None else pd.DataFrame()
    )
    tes_agg = _aggregate_by_map(df, te_map) if te_map is not None else pd.DataFrame()

    parts = [p for p in [genes_agg, tes_agg] if not p.empty]
    if parts:
        return pd.concat(parts)
    return pd.DataFrame()


def _apply_bulk_updates(target_df, updates):
    """Apply a dict-of-dicts as bulk column updates to a DataFrame."""
    if not updates or not any(updates.values()):
        return
    all_tids = set()
    for values in updates.values():
        all_tids.update(values.keys())
    update_df = pd.DataFrame(index=list(all_tids))
    for col, values in updates.items():
        if values:
            update_df[col] = pd.Series(values)
    target_df.loc[update_df.index, update_df.columns] = update_df


# --- Core Functions ---
def load_preprocessed_annotations(annotation_file):
    """
    Load preprocessed annotation data from compressed pickle.
    Now includes TSL data and expected junctions.
    """
    logging.info(f"Loading preprocessed annotations from {annotation_file}")

    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    with gzip.open(annotation_file, "rb") as f:
        annotation_data = pickle.load(f)

    metadata = annotation_data["metadata"]
    logging.info(f"  Loaded annotations for {metadata['n_transcripts']:,} transcripts")
    logging.info(f"  Gene GTF: {os.path.basename(metadata['gene_gtf'])}")
    if metadata["te_gtf"]:
        logging.info(f"  TE GTF: {os.path.basename(metadata['te_gtf'])}")
    logging.info(f"  Created on: {metadata['creation_date']}")

    # Check for TSL data
    if metadata.get("has_tsl_data", False):
        logging.info(f"  TSL data available: {metadata.get('tsl_distribution', {})}")
    else:
        logging.info(f"  No TSL data in annotation")

    # Convert gene_map and te_map back to DataFrames
    gene_map = pd.DataFrame(
        {
            "LocusID": annotation_data["gene_map"]["LocusID"],
            "AggregateID": annotation_data["gene_map"]["AggregateID"],
        }
    )

    te_map = None
    if annotation_data["te_map"] is not None:
        te_map = pd.DataFrame(
            {
                "LocusID": annotation_data["te_map"]["LocusID"],
                "AggregateID": annotation_data["te_map"]["AggregateID"],
            }
        )

    # Format junction map for compatibility
    junction_map = None
    if "junction_map" in annotation_data:
        junction_map = {
            "junction_map": annotation_data["junction_map"],
            "transcript_unique_junctions": annotation_data.get(
                "transcript_unique_junctions", {}
            ),
            "metadata": {
                "total_junctions": metadata.get("n_junctions", 0),
                "unique_junctions": metadata.get("n_unique_junctions", 0),
            },
        }
        logging.info(
            f"  Junction map loaded: {metadata['n_junctions']:,} junctions "
            f"({metadata['n_unique_junctions']:,} unique)"
        )

    return (
        annotation_data["string_to_int"],
        annotation_data["int_to_string"],
        annotation_data["junction_string_to_int"],
        annotation_data["junction_int_to_string"],
        gene_map,
        te_map,
        junction_map,
        annotation_data["transcript_junction_order"],
        annotation_data.get("transcript_info", {}),
        annotation_data.get("transcript_expected_junctions", {}),
        annotation_data.get("subset_relationships", {}),
        annotation_data.get("subset_coverage_territory_mapping", {}),
        annotation_data.get("junction_competitors", {}),
        metadata,
    )


def parse_cDNA_fragment_stats_json(stats_file_path):
    """
    Parses a JSON stats file created by get_cDNA_fragment_stats.py.
    Args:
        stats_file_path (str or Path): The path to the input JSON file.
    Returns:
        A tuple of (mean, std_dev) if successful, otherwise (None, None).
    """
    try:
        with open(stats_file_path, "r") as f:
            data = json.load(f)

        # Directly access the required keys from the parsed JSON dictionary
        mean = data["mean"]
        std_dev = data["std_dev"]

        return mean, std_dev

    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
        # This robustly handles missing files, corrupted JSON, or missing keys.
        # We log a warning but don't crash the whole pipeline.
        logging.warning(f"Could not read or parse stats file '{stats_file_path}': {e}")
        return None, None


def prepare_length_vector(
    sample_name, bam_path, args, transcript_info, active_transcript_ids
):
    """
    Prepares the length normalization vector for a single sample,
    operating ONLY on the active set of transcripts.
    """
    # Create a Series of annotated lengths ONLY for the active transcripts

    logging.info(f"[{sample_name}] getting transcript lengths.")
    all_transcript_lengths = {
        tid: info.get("transcript_length", 1) for tid, info in transcript_info.items()
    }

    logging.info(f"[{sample_name}] getting active lengths.")
    active_lengths_map = {
        tid: all_transcript_lengths[tid]
        for tid in active_transcript_ids
        if tid in all_transcript_lengths
    }

    annotated_lengths = pd.Series(active_lengths_map)

    # Return variable with the fallback value
    length_vector = annotated_lengths

    # Try to get a more accurate effective length
    frag_mean, frag_sd = None, None
    if args.frag_stats_dir:
        logging.info(f"[{sample_name}] getting accurate effective lengths...")
        bam_basename = os.path.basename(bam_path)
        stats_filename = f"{bam_basename.rsplit('.', 1)[0]}_cDNA_frag_stats.json"
        stats_file_path = os.path.join(args.frag_stats_dir, stats_filename)
        try:
            if not os.path.exists(stats_file_path):
                raise FileNotFoundError(f"Required file not found: {stats_file_path}")

            # Parse the correctly-identified stats file
            frag_mean, frag_sd = parse_cDNA_fragment_stats_json(stats_file_path)
        except FileNotFoundError as e:
            logging.error(f"ERROR: {e}")
            logging.warning(
                f"[{sample_name}] Could not find stats file. Using annotated lengths."
            )

    elif args.mean_fragment_length:
        # This simpler case doesn't need the map, but won't provide an SD
        frag_mean = args.mean_fragment_length
        frag_sd = None

    # 3. If we found data, calculate effective length and overwrite the fallback vector
    if frag_mean is not None:
        if frag_sd is not None:
            logging.info(
                f"[{sample_name}] Using fragment dist (mean={frag_mean:.2f}, sd={frag_sd:.2f}) for effective lengths."
            )
        else:
            logging.info(
                f"[{sample_name}] Using fragment mean ({frag_mean:.2f}) for effective lengths."
            )

        # Call the high-accuracy distributional function
        length_vector = calculate_effective_length_distributional(
            annotated_lengths, frag_mean, frag_sd  # Pass the small, active set
        )
    else:
        logging.info(
            f"[{sample_name}] No fragment length info found. Using annotated transcript lengths for normalization."
        )
        # No action needed, length_vector already holds the correct fallback

    # 4. Final check for safety and return
    length_vector[length_vector <= 1] = 1
    return length_vector


def get_combined_gtf_path(metadata, temp_dir):
    """
    Get or create combined GTF path from metadata.

    Args:
        metadata: Annotation metadata containing GTF paths
        temp_dir: Temporary directory for combined GTF



    Returns:
        str: Path to combined GTF file
    """
    gene_gtf = metadata["gene_gtf"]
    te_gtf = metadata.get("te_gtf")
    gtfs_to_combine = []

    if not os.path.exists(gene_gtf):
        raise FileNotFoundError(f"Gene GTF not found: {gene_gtf}")
    gtfs_to_combine.append(gene_gtf)

    if te_gtf:
        if not os.path.exists(te_gtf):
            raise FileNotFoundError(f"TE GTF not found: {te_gtf}")
        gtfs_to_combine.append(te_gtf)

    # --- Decide if concatenation is needed ---
    if len(gtfs_to_combine) == 1:
        # No need to combine, just return the path to the single GTF
        return gtfs_to_combine[0]
    else:
        # We need to concatenate the files.
        combined_gtf_path = os.path.join(temp_dir, "combined_for_run.gtf")
        logging.info(f"Creating combined GTF with {len(gtfs_to_combine)} files...")

        # Use a robust `cat` command via subprocess
        with open(combined_gtf_path, "wb") as outfile:
            subprocess.run(["cat"] + gtfs_to_combine, stdout=outfile, check=True)

        return combined_gtf_path


def read_unique_report_from_fifo_encoded(
    fifo_path, result_container, sample_name=None, string_to_int=None
):
    """Modified version that encodes feature tuples."""
    unique_mapper_groups = defaultdict(int)
    unique_mapper_counts = defaultdict(float)
    lines_processed = 0
    start_time = time.time()

    log_prefix = f"  [{sample_name}]" if sample_name else "  "
    unknown_transcript_count = 0
    try:
        with open(fifo_path, "r") as fifo:
            logging.info(f"{log_prefix} FIFO opened, processing report stream...")

            for line in fifo:
                if line.startswith("#"):
                    continue

                if "\tAssigned\t" not in line:
                    continue

                lines_processed += 1

                log_interval = 10_000_000 if lines_processed > 10_000_000 else 5_000_000
                if lines_processed % log_interval == 0:
                    elapsed = time.time() - start_time
                    rate = lines_processed / elapsed
                    logging.info(
                        f"{log_prefix}   FIFO processing: {lines_processed/1e6:.1f}M lines, {rate/1000:.0f}K lines/sec"
                    )

                try:
                    parts = line.strip().split("\t", 3)
                    features_int = [
                        string_to_int[feat.strip()] for feat in parts[3].split(",")
                    ]
                except KeyError as e:
                    unknown_transcript_count += 1
                    if unknown_transcript_count < 10:
                        logging.warning(
                            "################################################################################"
                        )
                        logging.warning(
                            f"[{sample_name}] ###~~~!!!!! Unknown transcript: {e} - skipping line !!!!!~~~###"
                        )
                        logging.warning(
                            "################################################################################"
                        )
                    else:
                        logging.warning(
                            f"[{sample_name}] Suppressing further unknown transcript warnings..."
                        )
                    continue

                # 2. Populate the equivalence classes for the EM.
                features_tuple = tuple(sorted(features_int))
                unique_mapper_groups[features_tuple] += 1

                # 3. Do the accounting for the initial counts (simple fractional).
                count_increment = 1.0 / len(features_int)
                for feat_int in features_int:
                    unique_mapper_counts[feat_int] += count_increment

    except Exception as e:
        logging.error(f"{log_prefix} Error in FIFO processing: {e}")
        result_container["error"] = str(e)
        result_container["unique_mapper_groups"] = {}
        result_container["unique_mapper_counts"] = {}

    finally:
        # Store results
        elapsed = time.time() - start_time
        logging.info(
            f"{log_prefix} FIFO processing complete: {lines_processed/1e6:.1f}M lines in {elapsed:.1f}s ({lines_processed/elapsed/1000:.0f}K lines/sec)"
        )
        result_container["unique_mapper_groups"] = dict(unique_mapper_groups)
        result_container["unique_mapper_counts"] = dict(unique_mapper_counts)


def read_multimapper_report_from_fifo_with_counts(
    fifo_path, result_container, sample_name=None, string_to_int=None
):
    """
    Parse FIFO to create BOTH equivalence classes AND proper fractional counts.
    """
    read_to_features = defaultdict(set)
    lines_processed = 0
    start_time = time.time()

    log_prefix = f"  [{sample_name}]" if sample_name else "  "

    try:
        with open(fifo_path, "r") as fifo:
            logging.info(f"{log_prefix} FIFO opened, processing multi-mapper report...")

            for line in fifo:
                if line.startswith("#"):
                    continue

                if "\tAssigned\t" not in line:
                    continue

                lines_processed += 1

                parts = line.strip().split("\t", 3)
                if len(parts) >= 4 and parts[3] != "NA":
                    read_id = parts[0]
                    feature_string = parts[3]

                    # Encode features
                    features = []
                    for feat in feature_string.split(","):
                        feat = feat.strip()
                        features.append(string_to_int[feat])

                    if features:
                        read_to_features[read_id].update(features)

        logging.info(
            f"{log_prefix} FIFO parsing completed: {lines_processed:,} assignment lines, "
            f"{len(read_to_features):,} unique reads"
        )

        # Create equivalence classes
        multimapper_groups = defaultdict(int)

        # Create fractional counts
        feature_counts = defaultdict(float)

        for read_id, feature_set in read_to_features.items():
            if feature_set:
                # Equivalence class: count read once
                features_tuple = tuple(sorted(feature_set))
                multimapper_groups[features_tuple] += 1

                # Fractional counts: each feature gets 1/N of read
                fraction = 1.0 / len(feature_set)
                for feature in feature_set:
                    feature_counts[feature] += fraction

        # Clean up
        del read_to_features

        # Return both results
        result_container["multimapper_groups"] = dict(multimapper_groups)
        result_container["multimapper_counts"] = dict(feature_counts)

        logging.info(
            f"{log_prefix} Created {len(multimapper_groups):,} equivalence classes"
        )
        logging.info(
            f"{log_prefix} Total fractional counts: {sum(feature_counts.values()):.0f}"
        )

    except Exception as e:
        logging.error(f"{log_prefix} Error in FIFO processing: {e}")
        result_container["error"] = str(e)
        result_container["multimapper_groups"] = {}
        result_container["multimapper_counts"] = {}


def get_special_feature_counts_bedtools(
    bam_file_path,
    bed_file_path,
    genomie_file_path,
    strand_specific=True,  # Should almost always be True for RNA-seq
):
    """
    Uses bedtools coverage to get read counts for the special MAJEC features.

    This function is a direct, robust replacement for using featureCounts for this task,
    correctly measuring physical read overlap.

    Args:
        bam_file_path (str): The full path to the coordinate-sorted, indexed BAM file.
        bed_file_path (str): The full path to the special features BED file.
        strand_specific (bool): Whether to perform stranded counting (-s flag).

    Returns:
        dict: A dictionary of {feature_id: count}, e.g., {'REGION_1': 50, 'REGION_2': 120}.
              Returns an empty dictionary if the process fails.
    """
    sample_name = os.path.basename(bam_file_path)
    logging.info(
        f"  [{sample_name}] Getting special feature coverage using bedtools..."
    )

    # --- Step 1: Construct the bedtools command ---

    # Base command
    cmd_parts = [
        "bedtools",
        "coverage",
        "-a",
        bed_file_path,
        "-b",
        bam_file_path,
        "-g",
        genomie_file_path,
        "-split",
        "-mean",  # the -mean flag is not mentioned in the bedtools documentation but clearly works in practice for v2.31.1
        "-sorted",
    ]
    if strand_specific:
        cmd_parts.append("-s")

    command_string = shlex.join(cmd_parts)
    # --- Step 2: Execute the command safely ---

    try:
        # The wrapper handles the execution and the CalledProcessError.
        # It will log any errors automatically.
        result = safe_subprocess_run(
            command_string, description=f"bedtools coverage for {sample_name}"
        )

        # We only proceed if the command was successful.
        # The wrapper returns the result object on success.

    except FileNotFoundError:
        # This can still happen if `bedtools` itself is not installed.
        logging.error(
            "FATAL: `bedtools` command not found. Please ensure it is installed and in your PATH."
        )
        raise

    except Exception as e:
        # Catch any other unexpected errors from the wrapper
        logging.error(
            f"  [{sample_name}] An unexpected error occurred while running bedtools: {e}"
        )
        return {}

    # --- Step 3: Parse the tab-separated output ---

    # The output of `bedtools coverage -counts` is a BED file with the count in the last column.
    # We can use pandas.read_csv for a very fast and robust way to parse this.
    try:
        from io import StringIO

        # The stdout is in the .stdout attribute of the result object
        stdout_text = result.stdout

        col_names = [
            "chrom",
            "start",
            "end",
            "feature_id",
            "score",
            "strand",
            "mean_coverage",
        ]

        df = pd.read_csv(StringIO(stdout_text), sep="\t", header=None, names=col_names)

        counts_dict = df.set_index("feature_id")["mean_coverage"].to_dict()

    except Exception as e:
        logging.error(
            f"  [{sample_name}] Failed to parse the output from bedtools: {e}"
        )
        return {}

    logging.info(
        f"  [{sample_name}] Successfully quantified mean coverage for {len(counts_dict)} special features."
    )
    return counts_dict


def _build_pipeline_context(annotation_metadata):
    """Build pipeline context dict from global worker state."""
    return {
        "junction_map": _junction_map,
        "junction_string_to_int": _junction_string_to_int_map,
        "junction_int_to_string": _junction_int_to_string_map,
        "transcript_junction_order": _transcript_junction_order,
        "transcript_info": _transcript_info,
        "subset_relationships": _subset_relationships,
        "subset_coverage_territory_mapping": _subset_coverage_territory_mapping,
        "splice_competitor_map": _splice_competitor_map,
        "transcript_expected_junctions": _transcript_expected_junctions,
        "gene_map": _gene_map,
        "te_map": _te_map,
        "annotation_metadata": annotation_metadata,
    }


def _apply_priors_if_needed(
    sample_name,
    initial_counts,
    junction_evidence,
    junction_metrics,
    pipeline_context,
    special_feature_counts_series,
    args,
):
    """Apply junction-informed priors if any evidence/penalty flags are active."""
    if junction_evidence or args.use_tsl_penalty or args.use_junction_completeness:
        return create_junction_informed_priors_encoded_with_penalties(
            sample_name,
            initial_counts,
            junction_evidence,
            junction_metrics,
            pipeline_context,
            special_feature_counts_series,
            args,
        )
    return initial_counts


def _process_sample_cached(
    sample_name,
    bam_path,
    cache_manager,
    cache_path,
    pipeline_context,
    shared_cache,
    args,
):
    """Process a sample using cached featureCounts results."""
    cache_data = cache_manager.load_sample_cache(cache_path)
    logging.info(f"[{sample_name}] Using cached featureCounts results")

    initial_unique_counts = cache_data["initial_unique_counts"]
    unique_mapper_groups = cache_data["unique_mapper_groups"]
    cached_junction_metrics = cache_data.get("junction_metrics", {})
    if args.use_subset_coverage_data:
        special_feature_counts_series = pd.Series(
            cache_data.get("special_feature_counts", {})
        )
    else:
        special_feature_counts_series = pd.Series({})
    special_feature_counts_series.name = sample_name

    junction_evidence = {}
    if cached_junction_metrics:
        logging.info(
            f"[{sample_name}] Calculating junction evidence from {len(cached_junction_metrics)} cached metrics"
        )
        junction_evidence = calculate_junction_evidence_from_metrics(
            cached_junction_metrics,
            _junction_map["junction_map"],
            junction_weight=args.junction_weight,
            decay_exponent=args.junction_decay_exponent,
        )
        logging.info(
            f"[{sample_name}] Calculated evidence for {len(junction_evidence)} transcripts"
        )

    modified_unique_counts = _apply_priors_if_needed(
        sample_name,
        initial_unique_counts,
        junction_evidence,
        cached_junction_metrics,
        pipeline_context,
        special_feature_counts_series,
        args,
    )

    shared_cache[sample_name] = {
        "initial_unique_counts": initial_unique_counts,
        "modified_unique_counts": modified_unique_counts,
        "unique_mapper_groups": unique_mapper_groups,
        "junction_evidence": junction_evidence,
        "bam_path": bam_path,
        "junction_metrics": cache_data.get("junction_metrics", {}),
        "has_multimapper_data": cache_data.get("has_multimapper_data", False),
        "initial_multimapper_counts": cache_data.get("initial_multimapper_counts"),
        "multimapper_classes": cache_data.get("multimapper_classes"),
    }

    logging.info(
        f"[{sample_name}] Storing junction_metrics with "
        f"{len(shared_cache[sample_name]['junction_metrics'])} entries in shared cache"
    )
    return (
        modified_unique_counts[modified_unique_counts > 0].values,
        sample_name,
        cache_path,
    )


def _process_sample_fresh(
    sample_name,
    bam_path,
    combined_gtf_path,
    pipeline_context,
    shared_cache,
    cache_manager,
    annotation_metadata,
    threads_per_worker,
    temp_dir,
    args,
):
    """Process a sample from scratch via featureCounts."""
    sample_start_time = time.time()
    cache_path = None
    include_junctions = _junction_map is not None

    logging.info(
        f"  [{sample_name}] Starting processing with {threads_per_worker} threads..."
    )
    fc_start = time.time()

    result = run_featurecounts_with_junctions_encoded(
        [bam_path],
        combined_gtf_path,
        args.strandedness,
        threads_per_worker,
        args.paired_end,
        extra_args="-Q 30 -O --fraction",
        report_format="CORE",
        temp_dir=temp_dir,
        include_junctions=include_junctions,
        string_to_int=_string_to_int_map,
        sample_name=sample_name,
    )

    (
        initial_unique_counts,
        um_report_path,
        um_fc_dir,
        junction_file,
        unique_mapper_groups,
    ) = result
    fc_elapsed = time.time() - fc_start
    logging.info(
        f"  [{sample_name}] FeatureCounts completed in {fc_elapsed:.1f}s (used FIFO)"
    )
    logging.info(
        f"  [{sample_name}] Found {len(unique_mapper_groups):,} equivalence groups"
    )

    # Special feature coverage
    if args.use_subset_coverage_data:
        logging.info(
            f"[{sample_name}] starting subset feature coverage quantification..."
        )
        special_feature_counts = get_special_feature_counts_bedtools(
            bam_path,
            annotation_metadata.get("subset_coverage_features_bed_path"),
            annotation_metadata.get("genome_file_path"),
            strand_specific=(args.strandedness != 0),
        )
        logging.info(
            f"[{sample_name}] obtained {len(special_feature_counts)} special feature counts"
        )
    else:
        special_feature_counts = {}

    special_feature_counts_series = pd.Series(special_feature_counts)
    special_feature_counts_series.name = sample_name

    # Junction processing
    junction_evidence = {}
    junction_metrics = {}
    modified_unique_counts = initial_unique_counts

    if include_junctions and junction_file and os.path.exists(junction_file):
        args.threads_per_worker = threads_per_worker
        junction_evidence, junction_metrics = parse_junctions_if_requested(
            args,
            junction_file,
            [bam_path],
            pipeline_context,
            junction_string_to_int=_junction_string_to_int_map,
        )

        modified_unique_counts = _apply_priors_if_needed(
            sample_name,
            initial_unique_counts,
            junction_evidence,
            junction_metrics,
            pipeline_context,
            special_feature_counts_series,
            args,
        )

        # Save to disk cache
        if not args.read_only_cache and cache_manager:
            cache_key, key_factors = cache_manager.get_cache_key(
                bam_path, annotation_metadata, args
            )
            cache_path = cache_manager.get_sample_cache_path(sample_name, cache_key)
            cache_manager.save_sample_cache(
                sample_name,
                cache_key,
                {
                    "initial_unique_counts": initial_unique_counts,
                    "unique_mapper_groups": unique_mapper_groups,
                    "special_feature_counts": special_feature_counts,
                    "junction_metrics": junction_metrics,
                    "has_multimapper_data": False,
                },
                key_factors,
            )

    shutil.rmtree(um_fc_dir)

    shared_cache[sample_name] = {
        "initial_unique_counts": initial_unique_counts,
        "modified_unique_counts": modified_unique_counts,
        "unique_mapper_groups": unique_mapper_groups,
        "junction_evidence": junction_evidence,
        "bam_path": bam_path,
        "junction_metrics": junction_metrics,
    }

    total_elapsed = time.time() - sample_start_time
    logging.info(f"  [{sample_name}] Total processing: {total_elapsed:.1f}s")
    return (
        modified_unique_counts[modified_unique_counts > 0].values,
        sample_name,
        cache_path,
    )


def process_sample_for_caching_encoded(args_tuple):
    """Process a single sample: use cache if valid, otherwise run featureCounts fresh."""
    (
        bam_path,
        combined_gtf_path,
        args,
        temp_dir,
        threads_per_worker,
        shared_cache,
        cache_manager,
        annotation_metadata,
    ) = args_tuple
    sample_name = get_sample_name_from_bam(bam_path)

    global _string_to_int_map, _junction_string_to_int_map
    global _junction_map, _transcript_junction_order, _transcript_info
    global _subset_relationships, _splice_competitor_map
    global _subset_coverage_territory_mapping

    pipeline_context = _build_pipeline_context(annotation_metadata)

    # Try cache first
    if cache_manager and not args.rebuild_cache:
        valid, cache_path, cache_key = cache_manager.is_cache_valid(
            sample_name, bam_path, annotation_metadata, args
        )
        if valid:
            try:
                return _process_sample_cached(
                    sample_name,
                    bam_path,
                    cache_manager,
                    cache_path,
                    pipeline_context,
                    shared_cache,
                    args,
                )
            except Exception as e:
                logging.warning(f"[{sample_name}] Failed to load cache: {e}")
                traceback.print_exc()

    # Fresh processing
    try:
        return _process_sample_fresh(
            sample_name,
            bam_path,
            combined_gtf_path,
            pipeline_context,
            shared_cache,
            cache_manager,
            annotation_metadata,
            threads_per_worker,
            temp_dir,
            args,
        )
    except Exception as e:
        logging.warning(f"    Failed to process {sample_name}: {e}")
        traceback.print_exc()
        return np.array([])


def run_featurecounts_with_junctions_encoded(
    bam_files,
    combined_gtf,
    strandedness,
    threads,
    is_paired_end,
    extra_args="",
    report_format=None,
    temp_dir=".",
    include_junctions=False,
    string_to_int=None,
    sample_name=None,
    is_multimapper=False,
):
    """
    Modified version of run_featurecounts_with_junctions_encoded that uses FIFO processing with tempfile backup.

    Args:
        bam_files: List of BAM files
        combined_gtf: GTF annotation file
        strandedness: Strand specificity setting
        threads: Number of threads
        is_paired_end: Whether data is paired-end
        extra_args: Additional featureCounts arguments
        report_format: Report format (e.g., "CORE")
        temp_dir: Temporary directory
        include_junctions: Whether to include junctions
        string_to_int: ID mapping
        sample_name: Sample name for logging
        is_multimapper: Whether processing multi-mappers (determines FIFO processor)

    Returns:
        Tuple of (counts_df, report_path, fc_temp_dir, junction_file_path, equivalence_groups)
    """
    fc_temp_dir = tempfile.mkdtemp(dir=temp_dir)
    output_path = os.path.join(fc_temp_dir, "fc_output.tsv")

    # Use the first BAM file's name for the report
    bam_basename = os.path.basename(bam_files[0])
    report_fifo = os.path.join(fc_temp_dir, f"{bam_basename}.featureCounts")

    paired_end_flag = "-p" if is_paired_end else ""
    junction_flag = "-J" if include_junctions else ""
    report_flag = f"-R {report_format}" if report_format else ""

    quoted_bams = " ".join(f'"{b}"' for b in bam_files)
    cmd = (
        f'featureCounts -a "{combined_gtf}" -o "{output_path}" -s {strandedness} '
        f"-T {threads} -t exon -g transcript_id {paired_end_flag} {extra_args} "
        f"{report_flag} {junction_flag} {quoted_bams}"
    )

    logging.info(f"  ... featureCounts command: {cmd}")

    # Try FIFO approach first
    try:
        # Create FIFO
        os.mkfifo(report_fifo)

        # Initialize container and choose processor based on mode
        if is_multimapper:
            result_container = {
                "multimapper_groups": None,
                "multimapper_counts": None,
                "error": None,
            }
            fifo_processor = read_multimapper_report_from_fifo_with_counts
            logging.info(f"  ... Using multi-mapper FIFO processor")
        else:
            result_container = {
                "unique_mapper_groups": None,
                "unique_mapper_counts": None,
                "error": None,
            }
            fifo_processor = read_unique_report_from_fifo_encoded
            logging.info(f"  ... Using unique mapper FIFO processor")

        # Start reader thread with correct processor
        reader_thread = threading.Thread(
            target=fifo_processor,
            args=(report_fifo, result_container, sample_name, string_to_int),
        )
        reader_thread.start()

        time.sleep(0.5)  # Give reader time to start

        # Run featureCounts
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=fc_temp_dir,
        )

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"FeatureCounts failed: {stderr}")

        reader_thread.join(timeout=120)

        if reader_thread.is_alive():
            logging.error("Report reader thread timed out!")
            raise RuntimeError("Reader timeout")

        if result_container.get("error"):
            raise RuntimeError(f"FIFO processing error: {result_container['error']}")

    except Exception as e:
        logging.warning(f"  [{sample_name}] FIFO approach failed: {e}")
        logging.warning(f"  [{sample_name}] Falling back to file-based approach...")

        # Clean up FIFO
        try:
            os.unlink(report_fifo)
        except:
            pass

        # Re-run featureCounts to disk
        process = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=fc_temp_dir
        )

        if process.returncode != 0:
            raise RuntimeError(
                f"FeatureCounts failed even without FIFO: {process.stderr}"
            )

        # Read from file using correct processor based on mode
        if is_multimapper:
            result_container = {
                "multimapper_groups": None,
                "multimapper_counts": None,
                "error": None,
            }
            read_multimapper_report_from_fifo_with_counts(
                report_fifo, result_container, sample_name, string_to_int
            )
        else:
            result_container = {
                "unique_mapper_groups": None,
                "unique_mapper_counts": None,
                "error": None,
            }
            read_unique_report_from_fifo_encoded(
                report_fifo, result_container, sample_name, string_to_int
            )

    finally:
        # Clean up FIFO if it still exists
        try:
            os.unlink(report_fifo)
        except:
            pass

    # Return based on mode
    junction_file_path = f"{output_path}.jcounts" if include_junctions else None

    if is_multimapper:
        logging.info(f"  ... Multi-mapper mode: returning FIFO-derived data")
        return report_fifo, fc_temp_dir, junction_file_path, result_container
    else:
        logging.info(f"  ... Unique mapper mode: returning counts DataFrame")
        sparse_counts_series = pd.Series(
            result_container.get("unique_mapper_counts", {})
        )
        all_transcript_ids = range(len(string_to_int))
        counts_df = sparse_counts_series.reindex(all_transcript_ids, fill_value=0.0)
        counts_df.name = sample_name
        return (
            counts_df,
            report_fifo,
            fc_temp_dir,
            junction_file_path,
            result_container.get("unique_mapper_groups", {}),
        )


def _apply_junction_boosts(
    sample_name, modified_counts, prior_tracking, weighted_junction_evidence, args
):
    """Step 1: Apply weighted junction evidence as additive boosts to counts."""
    logging.info(f"    [{sample_name}] Calculating junction boosts....")

    evidence_series = pd.Series(weighted_junction_evidence)
    final_boost_series = evidence_series * args.junction_weight
    modified_counts += final_boost_series.reindex(modified_counts.index, fill_value=0)

    if prior_tracking is not None:
        common_indices = prior_tracking.index.intersection(evidence_series.index)
        if not common_indices.empty:
            prior_tracking.loc[common_indices, "junction_boost"] = (
                final_boost_series.loc[common_indices]
            )
            prior_tracking.loc[common_indices, "raw_junction_evidence"] = (
                evidence_series.loc[common_indices]
            )
            prior_tracking.loc[common_indices, "junction_weight"] = args.junction_weight
        prior_tracking["post_junction_count"] = modified_counts.loc[
            prior_tracking.index
        ]

    logging.info(
        f"    [{sample_name}] Transcripts with junction evidence: {sum(final_boost_series > 0)} transcripts"
    )


def _apply_tsl_penalties(
    sample_name, modified_counts, prior_tracking, pipeline_context, args
):
    """Step 2: Apply Transcript Support Level penalties."""
    transcript_info = pipeline_context.get("transcript_info")
    if not transcript_info:
        return

    tsl_penalty_dict = {"1": 1.0, "2": 0.9, "3": 0.7, "4": 0.5, "5": 0.3, "NA": 0.8}
    if args.tsl_penalty_values:
        tsl_penalty_dict.update(args.tsl_penalty_values)

    active_index = (
        prior_tracking.index
        if prior_tracking is not None
        else modified_counts[modified_counts > 0].index
    )

    if prior_tracking is not None:
        prior_tracking["tsl"] = "NA"
        prior_tracking["tsl_penalty"] = 1.0

    penalized_count = 0
    for transcript_int in active_index:
        if transcript_int in transcript_info:
            tsl = transcript_info[transcript_int].get("tsl", "NA")
            penalty = tsl_penalty_dict.get(tsl, tsl_penalty_dict.get("NA", 1.0))

            if prior_tracking is not None:
                prior_tracking.loc[transcript_int, "tsl"] = tsl
                prior_tracking.loc[transcript_int, "tsl_penalty"] = penalty

            if penalty < 1.0 and modified_counts[transcript_int] > 0:
                modified_counts[transcript_int] *= penalty
                penalized_count += 1

    if prior_tracking is not None:
        logging.info(
            f"    TSL penalties by level: {dict(prior_tracking['tsl'].value_counts().sort_index())}"
        )
    else:
        logging.info(
            f"    [{sample_name}] TSL penalties applied to {penalized_count} transcripts"
        )


def _build_completeness_updates(
    filtered_model_details, junction_metrics, pipeline_context
):
    """Build the bulk update dict from completeness model details."""
    updates = {
        k: {}
        for k in [
            "n_expected_junctions",
            "n_observed_junctions",
            "junction_observation_rate",
            "completeness_penalty",
            "completeness_model",
            "terminal_recovery_used",
            "RMS_used",
            "worst_junction_position",
            "z_score",
            "n_worst_junctions",
            "worst_junction_positions",
            "worst_junction_counts",
            "worst_junction_z_scores",
            "recovery_point",
            "worst_expected_vs_actual",
            "median_coverage",
            "pure_median_1pct",
            "used_splice_competition",
            "n_competing_junctions",
        ]
    }

    for transcript_int, details in filtered_model_details.items():
        if transcript_int in pipeline_context["transcript_junction_order"]:
            n_expected = len(
                pipeline_context["transcript_junction_order"][transcript_int]
            )
            updates["n_expected_junctions"][transcript_int] = n_expected
            n_observed = len(
                junction_metrics.get(transcript_int, {}).get("observed_junctions", {})
            )
            updates["n_observed_junctions"][transcript_int] = n_observed
            if n_expected > 0:
                updates["junction_observation_rate"][transcript_int] = (
                    n_observed / n_expected
                )

        updates["completeness_penalty"][transcript_int] = details.get("penalty", 1.0)
        updates["completeness_model"][transcript_int] = details.get("model", "none")
        updates["terminal_recovery_used"][transcript_int] = details.get(
            "terminal_recovery_used", False
        )
        updates["RMS_used"][transcript_int] = details.get("model", "none") == "RMS"
        updates["z_score"][transcript_int] = details.get("z_score", np.nan)

        if "worst_positions" in details:
            updates["worst_junction_position"][transcript_int] = details[
                "worst_positions"
            ][0]
            updates["n_worst_junctions"][transcript_int] = details.get(
                "n_worst_considered", 1
            )
            updates["worst_junction_positions"][transcript_int] = ",".join(
                map(str, details["worst_positions"])
            )
            updates["worst_junction_counts"][transcript_int] = ",".join(
                map(str, details["worst_counts"])
            )

        if "recovery_point" in details and details["recovery_point"] is not None:
            updates["recovery_point"][transcript_int] = details["recovery_point"]

        if "median_coverage" in details:
            updates["median_coverage"][transcript_int] = details["median_coverage"]

        updates["pure_median_1pct"][transcript_int] = details.get(
            "pure_median_1pct", False
        )
        updates["used_splice_competition"][transcript_int] = details.get(
            "used_splice_competition", False
        )
        updates["n_competing_junctions"][transcript_int] = details.get(
            "n_junctions_with_competitors", 0
        )

        if "worst_z_scores" in details:
            updates["worst_junction_z_scores"][transcript_int] = ",".join(
                f"{z:.2f}" for z in details["worst_z_scores"]
            )
            if "worst_expected_values" in details and "worst_actual_counts" in details:
                updates["worst_expected_vs_actual"][transcript_int] = ",".join(
                    f"{act:.0f}/{exp:.0f}"
                    for exp, act in zip(
                        details["worst_expected_values"], details["worst_actual_counts"]
                    )
                )

    return updates


def _apply_completeness_penalties(
    sample_name,
    modified_counts,
    prior_tracking,
    junction_metrics,
    pipeline_context,
    args,
):
    """Step 3: Calculate and apply junction completeness penalties. Returns model_details."""
    logging.info(f"[{sample_name}] Starting junction completeness calculations")

    if prior_tracking is not None:
        # Initialize completeness columns
        completeness_defaults = {
            "n_expected_junctions": 0,
            "n_observed_junctions": 0,
            "junction_observation_rate": 1.0,
            "completeness_penalty": 1.0,
            "completeness_model": "none",
            "terminal_recovery_used": False,
            "RMS_used": False,
            "worst_junction_position": -1,
            "z_score": np.nan,
            "n_worst_junctions": 1,
            "worst_junction_positions": "",
            "worst_junction_counts": "",
            "worst_junction_z_scores": "",
            "recovery_point": -1,
            "worst_expected_vs_actual": "",
            "median_coverage": np.nan,
            "pure_median_1pct": False,
            "used_splice_competition": False,
            "n_competing_junctions": 0,
        }
        for col, default in completeness_defaults.items():
            prior_tracking[col] = default

    completeness_scores, stats, model_details = (
        calculate_junction_completeness_position_aware_optimized(
            junction_metrics,
            pipeline_context["transcript_junction_order"],
            min_penalty=args.junction_completeness_min_score,
            overdispersion=args.junction_completeness_overdispersion,
            pseudocount=0.1,
            transcript_info=pipeline_context["transcript_info"],
            terminal_relax=args.terminal_relax,
            terminal_recovery_rate=args.terminal_recovery_rate,
            library_type=args.library_type,
            splice_competitor_map=pipeline_context.get("splice_competitor_map", {}),
            junction_map=pipeline_context["junction_map"]["junction_map"],
            use_paired_rescue=args.use_paired_rescue,
            paired_rescue_decay=args.paired_rescue_decay,
        )
    )

    # Log terminal recovery usage
    if stats["terminal_recovery_used"] > 0:
        pct_recovery = (
            100 * stats["terminal_recovery_used"] / stats["complex_analyzed"]
            if stats["complex_analyzed"] > 0
            else 0
        )
        logging.info(
            f"  Terminal recovery applied to {stats['terminal_recovery_used']} transcripts ({pct_recovery:.1f}% of complex)"
        )
        if args.library_type in ["dT", "polyA"]:
            logging.info(
                f"  Library type '{args.library_type}': 5' dropout recovery enabled"
            )
        elif args.library_type == "random":
            logging.info(f"  Library type 'random': 3' dropout recovery enabled")

    # Apply penalties to counts
    penalized_count = 0
    for transcript_int, details in model_details.items():
        penalty = details.get("penalty", 1.0)
        if penalty < 1.0:
            modified_counts[transcript_int] *= penalty
            penalized_count += 1

    if prior_tracking is not None:
        # Save stats file
        stats_file = f"{args.prefix}_{sample_name}_junction_completeness_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        # Build and apply bulk tracking updates
        logging.info(
            f"[{sample_name}] Updating tracking data for {len(model_details)} transcripts..."
        )
        filtered_model_details = {
            k: model_details[k] for k in prior_tracking.index if k in model_details
        }
        updates = _build_completeness_updates(
            filtered_model_details, junction_metrics, pipeline_context
        )
        _apply_bulk_updates(prior_tracking, updates)
        logging.info(f"[{sample_name}] Tracking updates complete")

    logging.info(
        f"    [{sample_name}] Junction completeness penalties: "
        f"{sum(score < 1.0 for score in completeness_scores.values())} transcripts"
    )
    logging.info(f"    [{sample_name}]   Model: median-based with terminal recovery")
    logging.info(
        f"    [{sample_name}]   Penalty severity - Mild: {stats['mild_penalty']}, "
        f"Moderate: {stats['moderate_penalty']}, Severe: {stats['severe_penalty']}"
    )

    if args.terminal_relax and stats["terminal_recovery_used"] > 0:
        logging.info(
            f"    [{sample_name}]   Terminal recovery: {stats['terminal_recovery_used']} transcripts "
            f"(rate={args.terminal_recovery_rate})"
        )

    if stats["pure_median_1pct"] > 0:
        logging.info(
            f"    [{sample_name}]   <1% rule triggered: {stats['pure_median_1pct']} transcripts"
        )

    if prior_tracking is not None:
        n_multi = sum(prior_tracking["n_worst_junctions"] > 1)
        if n_multi > 0:
            logging.info(
                f"    [{sample_name}]   Multi-junction penalties: {n_multi} transcripts "
                f"(considered up to {prior_tracking['n_worst_junctions'].max()} worst junctions)"
            )
        prior_tracking["post_completeness_count"] = modified_counts

    return model_details


def _combine_penalties(penalty_a, penalty_b):
    """Combine two penalty factors using min/multiplicative average."""
    return (min(penalty_a, penalty_b) + penalty_a * penalty_b) / 2


def _apply_subset_penalties(
    sample_name,
    modified_counts,
    prior_tracking,
    initial_counts,
    junction_metrics,
    model_details,
    special_feature_counts,
    pipeline_context,
    args,
):
    """Step 4: Calculate and apply subset isoform penalties with territory evidence."""
    logging.info(f"    [{sample_name}] calculating subset penalties...")

    nonzero_counts = initial_counts[initial_counts > 0]
    expression_scale = (
        np.percentile(nonzero_counts, 25) if len(nonzero_counts) > 0 else 20
    )

    if prior_tracking is not None:
        # Initialize subset columns
        for col, default in [
            ("is_subset", False),
            ("original_subset_penalty", 1.0),
            ("subset_z_score", np.nan),
            ("adjusted_subset_penalty", np.nan),
            ("territory_confidence", np.nan),
            ("territory_evidence_ratio", np.nan),
        ]:
            prior_tracking[col] = default

    subset_details = calculate_subset_penalties_holistic(
        modified_counts,
        pipeline_context["subset_relationships"],
        junction_metrics,
        pipeline_context["junction_map"]["junction_map"],
        pipeline_context["transcript_junction_order"],
        min_penalty=args.subset_penalty_min,
        subset_evidence_threshold=args.subset_evidence_threshold,
        overdispersion=args.subset_penalty_overdispersion,
        expression_scale_factor=expression_scale,
    )

    territory_mapping = pipeline_context.get("subset_coverage_territory_mapping", {})
    if territory_mapping and len(special_feature_counts) > 0:
        logging.info(
            f"    [{sample_name}] calculating coverage-based adjustments to subset penalties..."
        )
        territory_evidence = calculate_territory_adjustment_factors(
            special_feature_counts, territory_mapping
        )
    else:
        logging.info(
            f"    [{sample_name}] no territory data, skipping coverage-based adjustments"
        )
        territory_evidence = {}

    logging.info(
        f"    [{sample_name}] Applying combined subset penalties with penalty combination..."
    )

    # In light mode, track strategies for logging only (lightweight dict)
    strategy_counts = {} if prior_tracking is None else None
    subset_updates = None
    if prior_tracking is not None:
        subset_updates = {
            k: {}
            for k in [
                "is_subset",
                "original_subset_penalty",
                "subset_z_score",
                "adjusted_subset_penalty",
                "territory_confidence",
                "territory_evidence_ratio",
                "penalty_combination_strategy",
            ]
        }

    # Filter to expressed transcripts when tracking
    if prior_tracking is not None:
        active_subset_details = {
            k: subset_details[k] for k in prior_tracking.index if k in subset_details
        }
    else:
        active_subset_details = subset_details

    for transcript_int, details in active_subset_details.items():
        evidence = territory_evidence.get(transcript_int)
        initial_subset_penalty_value = details.get("penalty", 1.0)
        completeness_penalty = model_details.get(transcript_int, {}).get("penalty", 1.0)
        has_completeness = completeness_penalty < 1.0

        if evidence:
            evidence_ratio = evidence["evidence_ratio"]
            confidence = evidence["confidence"]
            if evidence_ratio < initial_subset_penalty_value:
                territory_penalty = evidence_ratio
            else:
                territory_penalty = (
                    (1 - confidence) * initial_subset_penalty_value
                ) + (confidence * evidence_ratio)
            final_penalty = _combine_penalties(territory_penalty, completeness_penalty)
            penalty_strategy = "comb_territory_completeness"
        elif has_completeness:
            final_penalty = _combine_penalties(
                completeness_penalty, initial_subset_penalty_value
            )
            penalty_strategy = "comb_completeness_subset"
        else:
            final_penalty = initial_subset_penalty_value
            penalty_strategy = "subset_only"

        # Adjust from current state (initial * completeness) to target (initial * final)
        adjustment_factor = (
            final_penalty / completeness_penalty
            if completeness_penalty > 0
            else final_penalty
        )
        modified_counts[transcript_int] *= adjustment_factor

        if subset_updates is not None:
            if evidence:
                subset_updates["territory_confidence"][transcript_int] = confidence
                subset_updates["territory_evidence_ratio"][
                    transcript_int
                ] = evidence_ratio
            subset_updates["is_subset"][transcript_int] = bool(details["is_subset"])
            subset_updates["original_subset_penalty"][
                transcript_int
            ] = initial_subset_penalty_value
            if "z_score" in details:
                subset_updates["subset_z_score"][transcript_int] = details["z_score"]
            subset_updates["adjusted_subset_penalty"][transcript_int] = final_penalty
            subset_updates["penalty_combination_strategy"][
                transcript_int
            ] = penalty_strategy
        else:
            strategy_counts[penalty_strategy] = (
                strategy_counts.get(penalty_strategy, 0) + 1
            )

    if subset_updates is not None:
        logging.info(f"    [{sample_name}] Applying subset penalty updates...")
        _apply_bulk_updates(prior_tracking, subset_updates)
        if subset_updates["penalty_combination_strategy"]:
            combo_counts = pd.Series(
                subset_updates["penalty_combination_strategy"]
            ).value_counts()
            logging.info(f"    [{sample_name}] Penalty combination strategies:")
            for strategy, count in combo_counts.items():
                logging.info(f"      {strategy}: {count} transcripts")
    else:
        logging.info(
            f"    [{sample_name}] Subset penalties applied to {len(subset_details)} transcripts"
        )
        for strategy, count in strategy_counts.items():
            logging.info(f"      {strategy}: {count} transcripts")


def _write_prior_tracking_report(sample_name, prior_tracking, args):
    """Write verbose prior adjustment details and summary to disk."""
    logging.info(f"    [{sample_name}] Calculating final statistics...")

    prior_tracking["total_penalty"] = 1.0
    for col in ["tsl_penalty", "completeness_penalty", "subset_penalty"]:
        if col in prior_tracking.columns:
            prior_tracking["total_penalty"] *= prior_tracking[col]
        else:
            prior_tracking[col] = 1.0

    prior_tracking["total_prior_adjustment"] = (
        prior_tracking["final_count"] / prior_tracking["initial_count"]
    )
    prior_tracking.loc[
        prior_tracking["initial_count"] == 0, "total_prior_adjustment"
    ] = np.inf

    prior_tracking["had_junction_boost"] = prior_tracking["junction_boost"] > 0
    prior_tracking["had_penalty"] = prior_tracking["total_penalty"] < 1.0

    # Build penalty types
    has_tsl = prior_tracking["tsl_penalty"] < 1.0
    has_comp = prior_tracking["completeness_penalty"] < 1.0
    has_subset = prior_tracking["subset_penalty"] < 1.0

    penalty_types = []
    for i in range(len(prior_tracking)):
        parts = []
        if has_tsl.iloc[i]:
            parts.append("tsl")
        if has_comp.iloc[i]:
            parts.append("completeness")
        if has_subset.iloc[i]:
            parts.append("subset")
        penalty_types.append("_".join(parts) if parts else "none")
    prior_tracking["penalty_types"] = penalty_types

    # Write detailed file
    prior_tracking_decoded = decode_transcript_ids(prior_tracking, _int_to_string_map)
    output_file = f"{args.prefix}_{sample_name}_prior_adjustments.tsv.gz"
    prior_tracking_decoded.to_csv(output_file, sep="\t", index=True, compression="gzip")
    logging.info(f"    [{sample_name}] Wrote prior adjustment details to {output_file}")

    # Write summary
    summary_stats = {
        "sample": sample_name,
        "transcripts_total": len(prior_tracking),
        "transcripts_with_reads": sum(prior_tracking["initial_count"] > 0),
        "transcripts_boosted": sum(prior_tracking["had_junction_boost"]),
        "transcripts_penalized": sum(prior_tracking["had_penalty"]),
        "tsl_penalties": sum(prior_tracking["tsl_penalty"] < 1.0),
        "completeness_penalties": sum(prior_tracking["completeness_penalty"] < 1.0),
        "subset_penalties": sum(prior_tracking["subset_penalty"] < 1.0),
        "total_initial_reads": prior_tracking["initial_count"].sum(),
        "total_final_reads": prior_tracking["final_count"].sum(),
        "median_boost": (
            prior_tracking.loc[
                prior_tracking["junction_boost"] > 0, "junction_boost"
            ].median()
            if any(prior_tracking["junction_boost"] > 0)
            else 0
        ),
        "median_penalty": (
            prior_tracking.loc[
                prior_tracking["total_penalty"] < 1.0, "total_penalty"
            ].median()
            if any(prior_tracking["total_penalty"] < 1.0)
            else 1.0
        ),
    }

    summary_file = f"{args.prefix}_prior_adjustment_summary.tsv"
    if not os.path.exists(summary_file):
        with open(summary_file, "w") as f:
            f.write("\t".join(summary_stats.keys()) + "\n")
    with open(summary_file, "a") as f:
        f.write("\t".join(str(v) for v in summary_stats.values()) + "\n")


def create_junction_informed_priors_encoded_with_penalties(
    sample_name,
    initial_counts,
    weighted_junction_evidence,
    junction_metrics,
    pipeline_context,
    special_feature_counts,
    args,
):
    """Apply junction evidence, TSL penalties, completeness, and subset penalties to initial counts."""
    modified_counts = initial_counts.copy()

    # In light mode, skip prior tracking entirely for memory/speed savings
    if getattr(args, "light", False):
        prior_tracking = None
    else:
        prior_tracking = pd.DataFrame(index=initial_counts.index[initial_counts > 0])
        prior_tracking["initial_count"] = initial_counts

    # Step 1: Junction evidence boosts
    _apply_junction_boosts(
        sample_name, modified_counts, prior_tracking, weighted_junction_evidence, args
    )

    # Step 2: TSL penalties
    if args.use_tsl_penalty:
        _apply_tsl_penalties(
            sample_name, modified_counts, prior_tracking, pipeline_context, args
        )

    # Step 3: Junction completeness penalties
    model_details = {}
    if args.use_junction_completeness and junction_metrics:
        model_details = _apply_completeness_penalties(
            sample_name,
            modified_counts,
            prior_tracking,
            junction_metrics,
            pipeline_context,
            args,
        )

    # Step 4: Subset penalties
    if args.use_subset_penalty and pipeline_context.get("subset_relationships"):
        _apply_subset_penalties(
            sample_name,
            modified_counts,
            prior_tracking,
            initial_counts,
            junction_metrics,
            model_details,
            special_feature_counts,
            pipeline_context,
            args,
        )

    # Final tracking and optional report
    if prior_tracking is not None:
        prior_tracking["final_count"] = modified_counts
        if args.verbose_output:
            _write_prior_tracking_report(sample_name, prior_tracking, args)

    return modified_counts


def calculate_global_thresholds_and_cache_results_encoded(
    pipeline_context,
    all_bam_paths,
    combined_gtf_path,
    args,
    temp_dir,
    string_to_int,
    int_to_string,
    percentile_thresholds=[25, 75],
):
    """Modified to use encoded processing with proper worker initialization."""

    logging.info(
        f".  ->  Processing {len(all_bam_paths)} samples to calculate thresholds AND cache unique mapper results..."
    )

    logging.info("Junction evidence will be integrated into initial counts")

    n_samples = len(all_bam_paths)
    total_threads = args.threads

    overhead_threads = n_samples
    available_for_featurecounts = max(n_samples, total_threads - overhead_threads)
    threads_per_worker = max(1, available_for_featurecounts // n_samples)

    with multiprocessing.Manager() as manager:
        managed_cache = manager.dict()

        # Get cache_manager if it exists
        cache_manager = pipeline_context.get("cache_manager")
        annotation_metadata = pipeline_context["annotation_metadata"]

        pool_args = [
            (
                bam_path,
                combined_gtf_path,
                args,
                temp_dir,
                threads_per_worker,
                managed_cache,
                cache_manager,
                annotation_metadata,
            )
            for bam_path in all_bam_paths
        ]

        # Process with initialized workers
        with multiprocessing.Pool(
            processes=min(args.threads, len(all_bam_paths)),
            initializer=init_worker,
            initargs=(
                int_to_string,
                string_to_int,
                pipeline_context["junction_int_to_string"],
                pipeline_context["junction_string_to_int"],
                pipeline_context["junction_map"],
                pipeline_context["transcript_junction_order"],
                pipeline_context["transcript_info"],
                pipeline_context["subset_relationships"],
                pipeline_context["subset_coverage_territory_mapping"],
                pipeline_context["splice_competitor_map"],
                pipeline_context["transcript_expected_junctions"],
                pipeline_context["gene_map"],
                pipeline_context["te_map"],
            ),
        ) as pool:
            pool_results = pool.map(process_sample_for_caching_encoded, pool_args)

        sample_expression_arrays = []
        cache_path_map = {}
        for expr_array, sample_name, cache_path in pool_results:
            sample_expression_arrays.append(expr_array)
            if cache_path:
                cache_path_map[sample_name] = cache_path

        pipeline_context["cache_path_manifest"] = cache_path_map
        pipeline_context["cached_results"].update(dict(managed_cache))

    # Calculate global thresholds
    all_expression_values = []
    for expr_array in sample_expression_arrays:
        all_expression_values.extend(expr_array)

    if not all_expression_values:
        logging.error("Could not collect expression values for global thresholds!")
        return None

    all_expression_array = np.array(all_expression_values)
    low_threshold = np.percentile(all_expression_array, percentile_thresholds[0])
    high_threshold = np.percentile(all_expression_array, percentile_thresholds[1])

    pipeline_context["global_thresholds"] = {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "total_values": len(all_expression_values),
        "percentile_thresholds": percentile_thresholds,
        "samples_processed": len(all_bam_paths),
        "optimization": "cached_results_with_integer_ids",
        "junction_aware": True,
    }

    logging.info(f". -> THRESHOLDS CALCULATED + RESULTS CACHED ===")
    logging.info(
        f"  Low threshold: {low_threshold:.2f}, High threshold: {high_threshold:.2f}"
    )

    return


def run_joint_em_for_sample_with_cache_encoded(args_tuple):
    """Modified to work with integer IDs throughout."""
    (
        sample_cache_data,
        global_data,
        bam_path,
        args,
        combined_gtf_path,
        temp_dir,
        threads_per_worker,
    ) = args_tuple
    sample_name = get_sample_name_from_bam(bam_path)

    # Use global mappings
    global _string_to_int_map, _int_to_string_map, _transcript_info

    logging.info(
        f"[{sample_name}] Starting Joint EM process (1/4) Processing multi-mappers..."
    )

    # --- 1. MULTIMAPPER PROCESSING ---
    # Check if we have cached multimapper data
    if sample_cache_data and sample_cache_data.get("has_multimapper_data", False):
        # Load multimapper data from cache
        logging.info(f"[{sample_name}]  (1/4) Using cached multi-mapper results ✨")
        initial_multimapper_counts = sample_cache_data.get(
            "initial_multimapper_counts", pd.Series(dtype=np.float64)
        )
        multimapper_classes = sample_cache_data.get("multimapper_classes", {})

        # Convert back to defaultdict
        multimapper_classes = defaultdict(int, multimapper_classes)
    else:
        # Normal multimapper processing
        logging.info(f"[{sample_name}]  (1/4) Processing multi-mappers...")
        if not sample_cache_data:
            logging.info(f"[{sample_name}]   - No sample_cache_data")
        else:
            logging.info(
                f"[{sample_name}]   - has_multimapper_data = {sample_cache_data.get('has_multimapper_data', 'NOT FOUND')}"
            )
        multi_bam = os.path.join(temp_dir, f"{sample_name}_multi_mappers.bam")
        filter_expr = "mapq < 30"
        safe_subprocess_run(
            f'samtools view -h -@ {threads_per_worker} -e '
            f"'{filter_expr}'"
            f' -o "{multi_bam}" "{bam_path}"',
            "filtering for multi-mappers",
        )

        # Check if multi-mapper BAM has any reads
        read_count = int(
            subprocess.check_output(f'samtools view -c "{multi_bam}"', shell=True).strip()
        )

        if read_count == 0:
            logging.info(f"[{sample_name}] No multi-mappers found, using zero counts")
            initial_multimapper_counts = pd.Series(dtype=np.float64)
            multimapper_classes = defaultdict(int)
        else:
            mm_result = run_featurecounts_with_junctions_encoded(
                [multi_bam],
                combined_gtf_path,
                args.strandedness,
                threads_per_worker,
                args.paired_end,
                extra_args="-M -O --fraction",
                report_format="CORE",
                temp_dir=temp_dir,
                include_junctions=False,
                string_to_int=_string_to_int_map,
                sample_name=sample_name,
                is_multimapper=True,  # Tell it this is multi-mapper processing
            )

            mm_report_path, mm_fc_dir, _, mm_groups_from_fifo = mm_result
            multimapper_classes = defaultdict(int)
            for feature_set, count in mm_groups_from_fifo["multimapper_groups"].items():
                multimapper_classes[frozenset(feature_set)] += count

            mm_fractional_counts = mm_groups_from_fifo["multimapper_counts"]
            initial_multimapper_counts = pd.Series(
                mm_fractional_counts, dtype=np.float64
            )

            logging.info(f"#######[{sample_name}] Cache update check:########")
            logging.info(f"  sample_cache_data exists: {sample_cache_data is not None}")
            logging.info(f"  sample_cache_data truthy: {bool(sample_cache_data)}")
            logging.info(f"  args.use_cache: {args.use_cache}")
            logging.info(
                f"  cache_manager in global_data: {'cache_manager' in global_data}"
            )

            if not args.read_only_cache:
                if sample_cache_data and args.use_cache:
                    cache_manager = global_data["cache_manager"]
                    multimapper_cache_data = {
                        "initial_multimapper_counts": initial_multimapper_counts,
                        "multimapper_classes": dict(
                            multimapper_classes
                        ),  # Convert from defaultdict
                    }
                    success = cache_manager.update_sample_cache_with_multimapper_data(
                        sample_name, multimapper_cache_data
                    )

                    if success:
                        logging.info(f"[{sample_name}] Cache updated successfully")
                    else:
                        logging.warning(
                            f"[{sample_name}] Failed to update cache with multimapper data"
                        )

            shutil.rmtree(mm_fc_dir)
            os.remove(multi_bam)

    # --- 2. UNIQUE MAPPER PROCESSING (use cache) ---
    junction_evidence = {}
    if sample_cache_data:
        logging.info(f"[{sample_name}]  (2/4) Using unique mapper results ✨")

        if "modified_unique_counts" in sample_cache_data:
            initial_unique_counts = sample_cache_data["modified_unique_counts"]
            junction_evidence = sample_cache_data.get("junction_evidence", {})
        else:
            initial_unique_counts = sample_cache_data["initial_unique_counts"]

        unique_mapper_groups = sample_cache_data["unique_mapper_groups"]

        if junction_evidence:
            logging.info(
                f"[{sample_name}]    -> Using junction-informed initial counts {len(junction_evidence)} transcripts with unique junction evidence"
            )
    else:
        logging.error(f"[{sample_name}] Cache miss - this should not happen!")
        raise RuntimeError(f"No cached data found for {sample_name}")

    # --- 3. DUMP EM CLASSES (if requested) ---
    if args.dump_em_classes:
        # Decode for debugging output
        class_data = {
            "sample_name": sample_name,
            "initial_unique_counts": decode_transcript_ids(
                sample_cache_data["initial_unique_counts"], _int_to_string_map
            ).to_dict(),
            "unique_mapper_groups": decode_transcript_ids(
                unique_mapper_groups, _int_to_string_map
            ),
            "multimapper_classes": decode_transcript_ids(
                multimapper_classes, _int_to_string_map
            ),
            "boosted_unique_counts": decode_transcript_ids(
                initial_unique_counts, _int_to_string_map
            ).to_dict(),
            "initial_multimapper_counts": decode_transcript_ids(
                initial_multimapper_counts, _int_to_string_map
            ).to_dict(),
            "junction_evidence": decode_transcript_ids(
                junction_evidence, _int_to_string_map
            ),
            "junction_boosted_transcripts": [
                _int_to_string_map[i]
                for i in junction_evidence.keys()
                if 0 <= i < len(_int_to_string_map)
            ],
            "global_expression_thresholds": global_data["global_thresholds"],
            "used_cached_results": True,
            "junction_aware": True,
        }

        dump_path = f"{args.prefix}_{sample_name}_em_classes.pkl"
        with open(dump_path, "wb") as f:
            pickle.dump(class_data, f)

        logging.info(f"[{sample_name}] Saved EM classes to {dump_path}")

    # --- 4. EM ALGORITHM ---
    logging.info(
        f"[{sample_name}]  (3/4) Initializing active features and starting EM iterations..."
    )
    all_feature_ids = np.arange(len(_string_to_int_map))  # Use integer IDs directly
    # Just track active features, no complex filtering

    # intialize reports incase not doing report calculations
    transcript_assignments = {}
    shared_fractions = {}
    distinguishability_metrics = {}
    junction_confidence_metrics = {}

    logging.info(f"[{sample_name}]  (3/4) Preparing length normalization vector...")

    # The bam_path_map would need to be passed into the worker, perhaps in the global_data dict

    active_features = set()

    # Anything with non-zero initial counts
    active_features.update(initial_unique_counts[initial_unique_counts > 0].index)

    # Anything in any equivalence class
    for group in unique_mapper_groups.keys():
        active_features.update(group)
    for feature_set in multimapper_classes.keys():
        active_features.update(feature_set)

    active_features = np.array(sorted(active_features))
    n_active = len(active_features)
    n_total = len(all_feature_ids)

    logging.info(
        f"[{sample_name}] Processing {n_active:,} active features "
        f"({n_active/n_total*100:.1f}% of {n_total:,} total)"
    )

    if args.output_tpm:
        length_vector = prepare_length_vector(
            sample_name,
            bam_path,
            args,
            _transcript_info,  # Use the global transcript_info map
            list(active_features),
        )
    else:
        length_vector = None

    # Initialize counts for active features only
    active_unique_counts = pd.Series(0.0, index=active_features, dtype=np.float64)
    if len(initial_unique_counts) > 0:
        overlap = active_features[np.isin(active_features, initial_unique_counts.index)]
        active_unique_counts[overlap] = initial_unique_counts[overlap].values

    active_initial_multi = pd.Series(0.0, index=active_features, dtype=np.float64)
    if len(initial_multimapper_counts) > 0:
        overlap = active_features[
            np.isin(active_features, initial_multimapper_counts.index)
        ]
        active_initial_multi[overlap] = initial_multimapper_counts[overlap].values

    # Keep full unique_mapper_groups for confidence calculations later
    # unique_mapper_groups_full = unique_mapper_groups.copy() probably delete

    # Clean up full-size arrays we don't need during EM
    del initial_unique_counts
    del initial_multimapper_counts
    gc.collect()

    logging.info(
        f"[{sample_name}] Initializing priors & starting joint momentum-accelerated EM iterations..."
    )

    # Initialize active counts
    active_total_counts = active_unique_counts.add(active_initial_multi, fill_value=0)
    active_final_unique = active_unique_counts.copy()
    active_final_multi = active_initial_multi.copy()

    # Create full-sized pre-EM counts for discord calculation later (skip in light mode)
    if getattr(args, "light", False):
        pre_em_total_counts = None
    else:
        pre_em_total_counts = pd.Series(0.0, index=all_feature_ids)
        pre_em_total_counts[active_features] = active_total_counts.copy()

    theta_history = [active_total_counts.copy()]
    momentum_start = getattr(args, "momentum_start", 5)
    momentum_failures = 0
    momentum_successes = 0
    transcript_assignments = None

    def _finalize_em_results():
        """Expand active arrays to full size, compute confidence & distinguishability."""
        nonlocal active_total_counts, active_final_unique, active_final_multi

        current_total = pd.Series(0.0, index=all_feature_ids)
        current_total[active_features] = active_total_counts
        final_unique = pd.Series(0.0, index=all_feature_ids)
        final_unique[active_features] = active_final_unique
        final_multi = pd.Series(0.0, index=all_feature_ids)
        final_multi[active_features] = active_final_multi

        del active_total_counts, active_final_unique, active_final_multi
        gc.collect()

        t_assignments = {}
        d_metrics = {}

        if args.output_confidence:
            logging.info(
                f"[{sample_name}] Tracking assignment entropy for confidence scores..."
            )
            t_assignments = track_assignment_entropy_parallel(
                unique_mapper_groups,
                multimapper_classes,
                current_total,
                min_expression=args.confidence_min_expression,
                threads_per_sample=threads_per_worker,
                sample_name=sample_name,
            )

            logging.info(f"[{sample_name}] Calculating distinguishability metrics...")
            expressed = set(
                current_total[current_total >= args.confidence_min_expression].index
            )
            d_metrics = calculate_differential_confidence_parallel(
                unique_mapper_groups,
                multimapper_classes,
                current_total,
                expressed,
                gene_map=global_data.get("gene_map"),
                te_map=global_data.get("te_map"),
                threads_per_sample=threads_per_worker,
                sample_name=sample_name,
            )

        return current_total, final_unique, final_multi, t_assignments, d_metrics

    # Create pipeline context for EM iterations
    em_pipeline_context = {
        "global_thresholds": global_data["global_thresholds"],
        "gene_map": global_data.get("gene_map"),
        "cached_results": {},
    }
    # Run EM iterations on active features
    for i in range(args.em_iterations):
        previous_total = active_total_counts.copy()
        if i % 5 == 0:
            total_reads_in_groups = sum(
                count for group, count in unique_mapper_groups.items()
            )
            logging.info(
                f"[{sample_name}] Total reads in equivalence classes: {total_reads_in_groups}"
            )
        if i < momentum_start:
            active_total_counts, active_final_unique, active_final_multi = (
                standard_em_step(
                    active_total_counts,
                    unique_mapper_groups,
                    multimapper_classes,
                    active_final_multi,
                )
            )
            active_total_counts.name = sample_name
            logging.info(f"     [{sample_name}] Standard EM Iteration {i + 1}")
        else:
            try:
                active_total_counts, active_final_unique, active_final_multi = (
                    grouped_momentum_acceleration(
                        em_pipeline_context,
                        theta_history,
                        unique_mapper_groups,
                        multimapper_classes,
                        active_final_multi,
                        momentum_scaling_values=args.momentum_scaling,
                    )
                )
                active_total_counts.name = sample_name
                momentum_successes += 1
                logging.info(
                    f"     [{sample_name}] Joint Momentum EM Iteration {i + 1}"
                )
            except Exception as e:
                logging.warning(
                    f"     [{sample_name}] Momentum failed ({e}), using standard EM"
                )
                active_total_counts, active_final_unique, active_final_multi = (
                    standard_em_step(
                        active_total_counts,
                        unique_mapper_groups,
                        multimapper_classes,
                        active_final_multi,
                    )
                )
                active_total_counts.name = sample_name
                momentum_failures += 1

        theta_history.append(
            active_total_counts.copy()
        )  # FIXED: Use active_total_counts
        if len(theta_history) > 5:
            theta_history.pop(0)

        if adaptive_convergence_check(active_total_counts, previous_total, i + 1):
            logging.info(f"[{sample_name}] Converged after {i + 1} iterations.")
            logging.info(
                f"[{sample_name}] Momentum stats: {momentum_successes} successes, {momentum_failures} failures"
            )

            (
                current_total_counts,
                final_unique_counts,
                final_multimapper_counts,
                transcript_assignments,
                distinguishability_metrics,
            ) = _finalize_em_results()
            break
        if i == args.em_iterations - 1:
            logging.warning(
                f"[{sample_name}] Did not converge after {args.em_iterations} iterations."
            )

            (
                current_total_counts,
                final_unique_counts,
                final_multimapper_counts,
                transcript_assignments,
                distinguishability_metrics,
            ) = _finalize_em_results()
            break

    if args.output_confidence:
        # Calculate confidence metrics
        # Call external function with string IDs
        confidence_metrics = calculate_confidence_metrics(
            transcript_assignments,  # Already integers
            current_total_counts,
        )
    else:
        confidence_metrics = {}

    final_unique_counts.name = sample_name
    final_multimapper_counts.name = sample_name
    current_total_counts.name = sample_name

    junction_confidence_metrics = {}

    if args.output_confidence:
        junction_evidence = sample_cache_data.get("junction_evidence", {})
        junction_metrics = sample_cache_data.get("junction_metrics", {})

        if junction_metrics:
            junction_confidence_metrics = (
                calculate_junction_confidence_metrics_vectorized(
                    junction_evidence,  # Already integers
                    junction_metrics,  # Already integers
                    current_total_counts,  # Already integers
                )
            )
            logging.info(
                f"[{sample_name}] Calculated junction confidence for {len(junction_confidence_metrics)} transcripts"
            )

    # Clean up
    del unique_mapper_groups
    del multimapper_classes
    del theta_history[:-1]
    gc.collect()

    # Return all results (still with integer IDs)
    return (
        final_unique_counts,
        final_multimapper_counts,
        current_total_counts,
        confidence_metrics,
        distinguishability_metrics,
        junction_confidence_metrics,
        pre_em_total_counts,
        length_vector,
    )


def run_joint_em_pipeline_encoded(
    pipeline_context, args, combined_gtf_path, temp_dir, string_to_int, int_to_string
):
    """Modified pipeline with integer ID encoding."""

    # STEP 1: Calculate thresholds AND cache unique mapper results
    calculate_global_thresholds_and_cache_results_encoded(
        pipeline_context,
        args.bams,
        combined_gtf_path,
        args,
        temp_dir,
        string_to_int,
        int_to_string,
    )

    if args.use_bins:
        # If the user provided a global binning file, we load it and
        # OVERWRITE the locally-calculated thresholds in the pipeline_context.
        logging.info(
            f"--- Overwriting local thresholds with global ones from: {args.use_bins} ---"
        )
        try:
            with open(args.use_bins, "r") as f:
                bin_data = json.load(f)
                # over-write
                pipeline_context["global_thresholds"] = bin_data["global_thresholds"]
            logging.info("  Global thresholds loaded and applied successfully.")
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            logging.error(
                f"FATAL: Failed to load or parse the binning JSON file. Error: {e}"
            )
            sys.exit(1)
    else:
        # If no global file is provided
        logging.info(f"----- Using locally-calculated -----")

    if pipeline_context["global_thresholds"] is None:
        logging.error(
            "Failed to calculate or load global thresholds! Cannot proceed to EM."
        )
        sys.exit(1)

    # STEP 2: Run EM pipeline using cached results
    logging.info("\n=== Running EM Pipeline with Cached Results ===")
    threads_per_worker = max(1, args.threads // len(args.bams))
    logging.info(f"  -> Allocating {threads_per_worker} threads per sample")

    pool_args = []
    for bam in args.bams:
        sample_name = get_sample_name_from_bam(bam)

        sample_cache_data = pipeline_context["cached_results"].get(sample_name, {})

        global_data = {
            "global_thresholds": pipeline_context["global_thresholds"],
            "gene_map": pipeline_context.get("gene_map"),
            "te_map": pipeline_context.get("te_map"),
            "cache_manager": pipeline_context.get("cache_manager"),
        }

        pool_args.append(
            (
                sample_cache_data,
                global_data,
                bam,
                args,
                combined_gtf_path,
                temp_dir,
                threads_per_worker,
            )
        )

    # Process with initialized workers
    junction_int_to_string = pipeline_context["junction_int_to_string"]
    gene_map = pipeline_context["gene_map"]
    te_map = pipeline_context["te_map"]

    with multiprocessing.Pool(
        processes=min(args.threads, len(args.bams)),
        initializer=init_worker_downstream,
        initargs=(
            int_to_string,
            string_to_int,
            gene_map,
            te_map,
            junction_int_to_string,
            pipeline_context["transcript_info"],
        ),
    ) as pool:
        results = pool.map(run_joint_em_for_sample_with_cache_encoded, pool_args)

    # This is where you should add the cache cleanup:

    # Clean up cache - in light mode also drop junction_metrics (not needed without confidence output)
    cleanup_keys = [
        "initial_unique_counts",
        "modified_unique_counts",
        "unique_mapper_groups",
        "junction_evidence",
        "subset_coverage_territory_mapping",
    ]
    if getattr(args, "light", False):
        cleanup_keys.append("junction_metrics")
    for sample_name in pipeline_context["cached_results"]:
        sample_data = pipeline_context["cached_results"][sample_name]
        for key in cleanup_keys:
            if key in sample_data:
                del sample_data[key]
    gc.collect()

    unique_results = [res[0] for res in results]
    multimapper_results = [res[1] for res in results]
    total_results = [res[2] for res in results]
    confidence_results = [res[3] for res in results]
    distinguishability_results = [res[4] for res in results]
    junction_confidence_results = [res[5] for res in results]
    pre_em_total_counts = [res[6] for res in results]

    for sample_name, sample_data in pipeline_context["cached_results"].items():
        # Delete equivalence classes - not needed after EM
        if "unique_mapper_groups" in sample_data:
            del sample_data["unique_mapper_groups"]
        if "multimapper_classes" in sample_data:
            del sample_data["multimapper_classes"]

    gc.collect()

    # STEP 3: Process results - delay decoding for performance
    logging.info("\n=== PROCESSING RESULTS ===")

    # Combine results into DataFrames (still with integer IDs)
    final_unique_counts = pd.concat(unique_results, axis=1)
    final_multimapper_counts = pd.concat(multimapper_results, axis=1)
    final_total_counts = pd.concat(total_results, axis=1)
    # Clean up individual results to free memory

    sample_length_vectors = {
        sample_name: res[7]
        for sample_name, res in zip(final_unique_counts.columns, results)
    }
    del unique_results
    del multimapper_results
    gc.collect()

    # Save standard outputs - decode only when writing files
    if args.verbose_output:
        decoded_unique = decode_transcript_ids(final_unique_counts, int_to_string)
        decoded_unique.to_csv(
            f"{args.prefix}_unique_mapper_EM_counts.tsv.gz",
            sep="\t",
            compression={"method": "gzip", "compresslevel": 1},
        )
        del decoded_unique

        decoded_multi = decode_transcript_ids(final_multimapper_counts, int_to_string)
        decoded_multi.to_csv(
            f"{args.prefix}_multimapper_EM_counts.tsv.gz",
            sep="\t",
            compression={"method": "gzip", "compresslevel": 1},
        )
        del decoded_multi
        gc.collect()

    # Decode total counts for main output
    decoded_total_counts = decode_transcript_ids(final_total_counts, int_to_string)
    decoded_total_counts.round(6).to_csv(f"{args.prefix}_total_EM_counts.tsv", sep="\t")

    # STEP 4: Save pipeline metadata
    threshold_info = {
        "global_thresholds": pipeline_context["global_thresholds"],
        "samples_processed": [get_sample_name_from_bam(bam) for bam in args.bams],
        "momentum_scaling": {
            "low": args.momentum_scaling[0],
            "medium": args.momentum_scaling[1],
            "high": args.momentum_scaling[2],
        },
        "convergence_threshold": 0.0001,
        "optimization_method": "cached_results_with_integer_ids",
        "cached_samples": len(pipeline_context["cached_results"]),
        "junction_aware": True,
        "junction_weight": getattr(args, "junction_weight", 1.0),
        "junction_decay_exponent": getattr(args, "junction_decay_exponent", 1.0),
        "junction_metrics_tracked": True,
        "memory_optimized": True,
        "integer_id_mapping": True,
        "total_transcript_ids": len(int_to_string),
    }

    threshold_path = f"{args.prefix}_joint_binning_info.json"
    with open(threshold_path, "w") as f:
        json.dump(threshold_info, f, indent=2)

    # Clean up
    pipeline_context["cached_results"].clear()

    logging.info(f"✅ Optimized pipeline complete")

    return (
        final_unique_counts,
        final_multimapper_counts,
        final_total_counts,
        confidence_results,
        distinguishability_results,
        junction_confidence_results,
        pre_em_total_counts,
        sample_length_vectors,
    )


def build_argument_parser():
    """Build and return the MAJEC pipeline argument parser."""
    parser = argparse.ArgumentParser(
        description="MAJEC - Momentum-Accelerated Junction-Enhanced Counting\n"
        "Requires preprocessed annotation file created by precompute_majec_annotations.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Group: Required I/O ---
    io_group = parser.add_argument_group("Required Input/Output")
    io_group.add_argument(
        "--annotation",
        help="Preprocessed annotation file (.pkl.gz) created by precompute_majec_annotations.py",
    )
    io_group.add_argument(
        "--bams", nargs="+", help="List of coordinate-sorted input BAM files"
    )
    io_group.add_argument(
        "--prefix",
        default="MAJEC_output",
        help="Output file prefix (default: MAJEC_output)",
    )

    # --- Group: Core Pipeline Settings ---
    core_group = parser.add_argument_group("Core Pipeline Settings")
    core_group.add_argument(
        "--strandedness",
        default=0,
        type=int,
        choices=[0, 1, 2],
        help="Strand specificity (default: 0)",
    )
    core_group.add_argument(
        "--threads",
        default=8,
        type=int,
        help="Total number of threads to use (default: 8)",
    )
    core_group.add_argument(
        "--paired_end", action="store_true", help="Data is paired-end"
    )
    core_group.add_argument("--tempdir", help="Directory for temporary files")
    core_group.add_argument(
        "--use_bins",
        type=str,
        default=None,
        help="Use pre-calculated expression thresholds from a specified JSON file for cohort processing.",
    )
    core_group.add_argument(
        "--config",
        help="Path to a JSON configuration file. Command-line options override file settings.",
    )

    # --- Group: EM Algorithm Parameters ---
    em_group = parser.add_argument_group("EM Algorithm Parameters")
    em_group.add_argument(
        "--em_iterations",
        default=150,
        type=int,
        help="Maximum iterations for EM (default: 150)",
    )
    em_group.add_argument(
        "--momentum_start",
        default=4,
        type=int,
        help="Iteration to start momentum acceleration (default: 4)",
    )
    em_group.add_argument(
        "--momentum_scaling",
        nargs=3,
        type=float,
        default=[1.5, 1.0, 0.7],
        metavar=("LOW", "MEDIUM", "HIGH"),
        help="Momentum scaling factors for expression groups (default: 1.5 1.0 0.7)",
    )

    # --- Group: Priors & Evidence Penalties ---
    priors_group = parser.add_argument_group("Evidence Priors & Penalties")
    priors_group.add_argument(
        "--junction_weight",
        default=3.0,
        type=float,
        help="Weight for unique junction evidence (default: 3.0)",
    )
    priors_group.add_argument(
        "--junction_decay_exponent",
        type=float,
        default=1.0,
        help="Exponent to penalize shared junctions (default: 1.0)",
    )
    priors_group.add_argument(
        "--use_tsl_penalty",
        action="store_true",
        help="Apply penalties based on Transcript Support Level (TSL)",
    )
    priors_group.add_argument(
        "--tsl_penalty_values",
        type=str,
        help='Custom TSL penalties as JSON string, e.g. \'{"1":1.0,"2":0.8}\'',
    )
    priors_group.add_argument(
        "--use_junction_completeness",
        action="store_true",
        help="Penalize transcripts with incomplete junction evidence",
    )
    priors_group.add_argument(
        "--junction_completeness_min_score",
        type=float,
        default=0.0001,
        help="Minimum completeness score to avoid zeroing transcripts (default: 0.0001)",
    )
    priors_group.add_argument(
        "--junction_completeness_overdispersion",
        type=float,
        default=0.05,
        help="Overdispersion parameter for junction variance model (default: 0.05)",
    )
    priors_group.add_argument(
        "--library_type",
        choices=["dT", "polyA", "random", "none"],
        default="WARNING_UNSPECIFIED",
        help="Library preparation method (REQUIRED for completeness model)",
    )
    priors_group.add_argument(
        "--terminal_relax",
        action="store_true",
        help="Relaxes completeness penalty at transcript termini based on library type",
    )
    priors_group.add_argument(
        "--terminal_recovery_rate",
        type=float,
        default=2,
        help="Controls recovery steepness for terminal relaxation (default: 2)",
    )
    priors_group.add_argument(
        "--use_subset_penalty",
        action="store_true",
        help="Penalize subset isoforms based on superset evidence",
    )
    priors_group.add_argument(
        "--subset_evidence_threshold",
        type=float,
        default=1.25,
        help="Factor to adjust evidence required to NOT penalize a subset isoform (default: 1.25)",
    )
    priors_group.add_argument(
        "--subset_penalty_min",
        type=float,
        default=0.001,
        help="Minimum penalty factor for subset isoforms (default: 0.001)",
    )
    priors_group.add_argument(
        "--subset_penalty_overdispersion",
        type=float,
        default=0.1,
        help="Overdispersion for subset penalty variance model (default: 0.1)",
    )
    priors_group.add_argument(
        "--use_subset_coverage_data",
        action="store_true",
        help="Use read coverage data to inform subset penalties (requires annotation with --generate_rescue_features)",
    )
    priors_group.add_argument(
        "--use_paired_rescue",
        action="store_true",
        default=True,
        help="Use paired junction evidence to rescue low-count junctions (default: True)",
    )
    priors_group.add_argument(
        "--no_paired_rescue",
        action="store_false",
        dest="use_paired_rescue",
        help="Disable paired junction rescue",
    )
    priors_group.add_argument(
        "--paired_rescue_decay",
        type=float,
        default=-1.5,
        help="Exponential decay constant for paired rescue confidence weight (default: -1.5). "
        "More negative = stricter (only unique pairs help). Less negative = more permissive.",
    )

    # --- Group: Length Scaling & TPM ---
    length_group = parser.add_argument_group("Length Scaling & TPM Calculation")
    length_group.add_argument(
        "--output_tpm",
        action="store_true",
        help="Calculate and output TPM values in addition to counts.",
    )
    length_group.add_argument(
        "--frag_stats_dir",
        type=str,
        default=None,
        help="Directory containing JSON fragment length statistics files from get_cDNA_fragment_stats.py.",
    )
    length_group.add_argument(
        "--mean_fragment_length",
        type=float,
        default=None,
        help="A single, global mean fragment length to use for all samples.",
    )

    # --- Group: Confidence & Output Settings ---
    output_group = parser.add_argument_group("Confidence & Output Settings")
    output_group.add_argument(
        "--output_confidence",
        action="store_true",
        help="Output confidence metrics based on assignment entropy",
    )
    output_group.add_argument(
        "--confidence_min_expression",
        type=float,
        default=1.0,
        help="Minimum expression level for confidence calculations (default: 1.0)",
    )
    output_group.add_argument(
        "--analyze_competition",
        action="store_true",
        help="Analyze and report transcript competition patterns",
    )
    output_group.add_argument(
        "--calculate_group_confidence",
        action="store_true",
        help="Calculate confidence metrics for aggregated gene/TE groups",
    )
    output_group.add_argument(
        "--verbose_output",
        action="store_true",
        help="Generate all detailed breakdown and intermediate files",
    )
    output_group.add_argument(
        "--dump_em_classes",
        action="store_true",
        help="Save EM equivalency classes for debugging",
    )
    output_group.add_argument(
        "--light",
        action="store_true",
        help="Light mode: skip prior tracking, confidence metrics, and verbose outputs "
        "for reduced memory usage and faster execution",
    )

    # --- Group: Caching ---
    cache_group = parser.add_argument_group("Caching")
    cache_group.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cached featureCounts results if available",
    )
    cache_group.add_argument(
        "--cache_dir",
        default="pipeline_cache",
        help="Directory for cache files (default: pipeline_cache)",
    )
    cache_group.add_argument(
        "--rebuild_cache",
        action="store_true",
        help="Force rebuild of cache even if valid",
    )
    cache_group.add_argument(
        "--read_only_cache",
        action="store_true",
        help="Use cache in read-only mode; safer for batch parameter sweeps",
    )

    return parser


def parse_args_with_config(parser):
    """Parse arguments, applying config file defaults if provided."""
    partial_args, _ = parser.parse_known_args()

    if partial_args.config:
        try:
            with open(partial_args.config, "r") as f:
                config_values = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            parser.error(
                f"Error reading or parsing config file '{partial_args.config}': {e}"
            )
        parser.set_defaults(**config_values)

    args = parser.parse_args(sys.argv[1:])

    # Validate required args (deferred from argparse to allow --config to supply them)
    if not args.annotation:
        parser.error("--annotation is required (via command line or --config)")
    if not args.bams:
        parser.error("--bams is required (via command line or --config)")

    Path(args.prefix).parent.mkdir(parents=True, exist_ok=True)
    return args


def validate_args(args):
    """Validate argument combinations and resolve BAM paths. Exits on failure."""
    # Resolve BAM paths to absolute
    args.bams = [os.path.abspath(bam) for bam in args.bams]
    for bam_path in args.bams:
        if not os.path.exists(bam_path):
            logging.error(f"FATAL: BAM file not found: {bam_path}")
            sys.exit(1)

    if args.frag_stats_dir and args.mean_fragment_length:
        logging.error(
            "FATAL: --frag_stats_dir and --mean_fragment_length are mutually exclusive."
        )
        sys.exit(1)

    if args.tsl_penalty_values:
        try:
            args.tsl_penalty_values = json.loads(args.tsl_penalty_values)
        except Exception:
            logging.error("Invalid JSON for --tsl_penalty_values")
            sys.exit(1)

    if args.terminal_relax and args.library_type == "WARNING_UNSPECIFIED":
        logging.error(
            "--terminal_relax specified but no --library_type provided. "
            "Please specify --library_type or remove --terminal_relax"
        )
        sys.exit(1)

    if args.calculate_group_confidence and not args.output_confidence:
        logging.warning(
            "--calculate_group_confidence requires --output_confidence; disabling group confidence"
        )
        args.calculate_group_confidence = False

    if getattr(args, "light", False):
        overridden = []
        for flag in [
            "verbose_output",
            "output_confidence",
            "analyze_competition",
            "calculate_group_confidence",
            "dump_em_classes",
        ]:
            if getattr(args, flag, False):
                overridden.append(f"--{flag}")
                setattr(args, flag, False)
        if overridden:
            logging.warning(f"--light mode: disabling {', '.join(overridden)}")
        logging.info(
            "Light mode enabled: skipping prior tracking, confidence metrics, and verbose outputs"
        )

    if args.frag_stats_dir and args.output_tpm:
        logging.info(
            f"Using fragment length statistics from directory: {args.frag_stats_dir}"
        )
    else:
        logging.warning(
            "No fragment length data provided via --frag_stats_dir. "
            "Falling back to annotated transcript lengths for normalization."
        )


def calculate_tpm(final_counts, sample_length_vectors):
    """Calculate TPM values from final EM counts and per-sample length vectors."""
    tpm_df = pd.DataFrame(index=final_counts.index, dtype=float)
    for col_name in final_counts.columns:
        lengths_kb = sample_length_vectors[col_name] / 1000.0
        rpk = final_counts[col_name].div(lengths_kb).fillna(0)
        scaling_factor = rpk.sum() / 1_000_000
        tpm_df[col_name] = rpk / scaling_factor if scaling_factor > 0 else 0
    return tpm_df


def main():
    parser = build_argument_parser()
    args = parse_args_with_config(parser)
    setup_logging(args.prefix)

    start_time = time.time()
    command_string = " ".join([shlex.quote(arg) for arg in sys.argv])
    logging.info("==================================================")
    logging.info("              MAJEC Pipeline Started              ")
    logging.info("==================================================")
    logging.info(f" Execution command:")
    logging.info(f"  {command_string}")
    logging.info(f"Processing {len(args.bams)} BAM files with {args.threads} threads.")
    logging.info(f"  -> Strand specificity: {args.strandedness}")
    logging.info(f"  -> Paired-end mode: {'Yes' if args.paired_end else 'No'}")
    logging.info(f"  -> Output prefix: {args.prefix}")

    validate_args(args)

    try:
        (
            string_to_int,
            int_to_string,
            junction_string_to_int,
            junction_int_to_string,
            gene_map,
            te_map,
            junction_map,
            transcript_junction_order,
            transcript_info,
            transcript_expected_junctions,
            subset_relationships,
            subset_coverage_territory_mapping,
            splice_competitor_map,
            metadata,
        ) = load_preprocessed_annotations(args.annotation)
    except Exception as e:
        logging.error(f"Failed to load annotation file: {e}")
        sys.exit(1)

    logging.info("Annotation & Feature Status:")
    logging.info(f"  -> Loaded annotation: {os.path.basename(args.annotation)}")
    logging.info(f"  -> Total transcripts in annotation: {metadata['n_transcripts']:,}")

    # Check if TSL penalty requested but no TSL data
    if args.use_tsl_penalty and not metadata.get("has_tsl_data", False):
        logging.warning("--use_tsl_penalty specified but annotation has no TSL data")
        logging.warning("TSL penalties will not be applied")
        args.use_tsl_penalty = False

    # Initialize pipeline context with new data
    pipeline_context = {
        "junction_map": junction_map,
        "junction_string_to_int": junction_string_to_int,
        "junction_int_to_string": junction_int_to_string,
        "transcript_junction_order": (
            transcript_junction_order
            if (args.use_junction_completeness or args.use_subset_penalty)
            else None
        ),
        "cached_results": {},
        "global_thresholds": None,
        "gene_map": gene_map,
        "te_map": te_map,
        "transcript_info": transcript_info,  # Includes TSL and transcript lengths
        "transcript_expected_junctions": transcript_expected_junctions,  # Not sure if needed anymore
        "subset_relationships": subset_relationships,  # For competition analysis
        "subset_coverage_territory_mapping": subset_coverage_territory_mapping,  # For coverage-based subset penalties
        "splice_competitor_map": splice_competitor_map,  # For competition analysis
        "annotation_metadata": metadata,
    }

    # Initialize cache manager if using cache
    if args.use_cache or args.rebuild_cache:
        cache_manager = PipelineCache(args.cache_dir)
        pipeline_context["cache_manager"] = cache_manager
        logging.info(f"Cache manager initialized with directory: {args.cache_dir}")

    if args.use_subset_coverage_data and (
        not metadata["subset_coverage_features_bed_path"]
        or not metadata["genome_file_path"]
    ):
        logging.error(
            "--use_subset_coverage_data specified but annotation lacks required coverage feature data"
        )
        logging.error(
            "Please recreate the annotation with --generate_rescue_features or disable this option"
        )
        sys.exit(1)

    # Run pipeline
    with tempfile.TemporaryDirectory(dir=args.tempdir) as temp_dir:
        logging.info(f"Using temporary directory: {temp_dir}")

        # Get combined GTF path from annotation metadata
        try:
            combined_gtf_path = get_combined_gtf_path(metadata, temp_dir)
        except FileNotFoundError as e:
            logging.error(str(e))
            logging.error(
                "The GTF files referenced in the annotation are not found at their original paths"
            )
            logging.error(
                "Please ensure the GTF files are accessible or recreate the annotation"
            )
            sys.exit(1)

        # Run optimized pipeline with integer IDs
        results = run_joint_em_pipeline_encoded(
            pipeline_context,
            args,
            combined_gtf_path,
            temp_dir,
            string_to_int,
            int_to_string,
        )

        if results[0] is None:
            logging.error("Joint EM pipeline failed")
            sys.exit(1)

        (
            final_unique_counts,
            final_multimapper_counts,
            final_total_counts,
            confidence_results,
            distinguishability_results,
            junction_confidence_results,
            pre_em_total_counts,
            sample_length_vectors,
        ) = results

        # Save enhanced confidence metrics
        if args.output_confidence or pipeline_context["junction_map"]:
            logging.info("Saving comprehensive confidence metrics in parallel...")

            # Convert DataFrame columns to list of Series for compatibility
            total_results = [
                final_total_counts[col] for col in final_total_counts.columns
            ]

            # Create gene_to_transcripts_map with INTEGER LocusIDs
            gene_to_transcripts_map = None
            if pipeline_context["junction_map"]:
                gene_map = pipeline_context["gene_map"]
                if gene_map is not None:
                    # This creates a map with integer transcript IDs!
                    gene_to_transcripts_map = (
                        gene_map.groupby("AggregateID")["LocusID"].apply(list).to_dict()
                    )

            # Prepare arguments for parallel saving
            pool_args = []
            for i, sample_counts in enumerate(total_results):
                sample_name = sample_counts.name
                sample_pre_em_total_counts = pre_em_total_counts[i]
                # Get the per-sample junction metrics
                sample_junction_metrics = (
                    pipeline_context["cached_results"]
                    .get(sample_name, {})
                    .get("junction_metrics", {})
                )

                # THE REFINED args_tuple:
                # We pass all the DYNAMIC per-sample data and the SMALL, STATIC config flags.
                # The LARGE static maps (int_to_string, etc.) have been REMOVED from this tuple.
                args_tuple = (
                    sample_name,
                    sample_counts,
                    confidence_results[i],
                    distinguishability_results[i],
                    junction_confidence_results[i],
                    sample_junction_metrics,
                    sample_pre_em_total_counts,
                    gene_to_transcripts_map,
                    args.prefix,
                    args.verbose_output,
                    int_to_string,
                    junction_int_to_string,
                )
                pool_args.append(args_tuple)

            # Create the Pool. It uses YOUR preferred downstream initializer.
            # The initargs EXACTLY match the signature of init_worker_downstream.
            if args.output_confidence:
                with multiprocessing.Pool(
                    processes=min(args.threads, len(total_results))
                ) as pool:
                    # The pool.map call remains the same, passing the list of tuples
                    parallel_results = pool.map(
                        process_and_save_sample_outputs, pool_args
                    )

                parallel_confidence_results = [res[0] for res in parallel_results]

                run_file_map = {}
                sample_names_from_run = [res[1] for res in parallel_results]

                for sample_name in sample_names_from_run:
                    # The main process now constructs all the paths.
                    # This is the single source of truth for naming conventions.
                    run_file_map[sample_name] = {
                        "priors": os.path.abspath(
                            f"{args.prefix}_{sample_name}_prior_adjustments.tsv.gz"
                        ),
                        "metrics_sparse": os.path.abspath(
                            f"{args.prefix}_{sample_name}_transcript_metrics_SPARSE.tsv.gz"
                        ),
                        "confidence_dense": os.path.abspath(
                            f"{args.prefix}_{sample_name}_counts_with_confidence.tsv.gz"
                        ),
                        "high_confidence": os.path.abspath(
                            f"{args.prefix}_{sample_name}_high_confidence_counts.tsv.gz"
                        ),
                    }

                # Create comprehensive transcript confidence summary
                if parallel_confidence_results and any(
                    r is not None for r in parallel_confidence_results
                ):
                    transcript_summary = pd.concat(
                        [r for r in parallel_confidence_results if r is not None],
                        axis=1,
                    )

                    # Decode the integer transcript IDs to strings for the output file
                    transcript_summary = decode_transcript_ids(
                        transcript_summary, int_to_string
                    )

                    output_file = f"{args.prefix}_transcript_confidence_summary.tsv.gz"
                    transcript_summary.to_csv(
                        output_file,
                        sep="\t",
                        compression={"method": "gzip", "compresslevel": 1},
                    )
                    logging.info(
                        f"✅ Saved transcript confidence summary for {len(parallel_confidence_results)} samples"
                    )
            else:
                logging.info(
                    "Skipping confidence metrics output as --output_confidence is not set."
                )
                run_file_map = {}

            if args.output_tpm:
                logging.info("Calculating TPM values from final counts...")
                tpm_df = calculate_tpm(final_total_counts, sample_length_vectors)
                decoded_tpm_df = decode_transcript_ids(tpm_df, int_to_string)
                tpm_filepath = f"{args.prefix}_total_EM_TPM.tsv"
                decoded_tpm_df.round(6).to_csv(tpm_filepath, sep="\t")
                logging.info(f"Final TPM estimates saved to: {tpm_filepath}")

            if args.analyze_competition:
                summary, complex_genes = analyze_competition_patterns(
                    distinguishability_results,
                    args.prefix,
                    int_to_string=int_to_string,
                )

        # --- AGGREGATION STEP ---
        logging.info("Aggregating final counts by gene symbol and TE subfamily...")
        gene_map = pipeline_context["gene_map"]
        te_map = pipeline_context["te_map"]

        if gene_map is not None:
            final_aggregated_counts = _aggregate_genes_and_tes(
                final_total_counts, gene_map, te_map
            )

            if not final_aggregated_counts.empty:
                agg_filepath = f"{args.prefix}_total_EM_aggregated_counts.tsv"
                final_aggregated_counts.to_csv(agg_filepath, sep="\t")
                logging.info(
                    f"Final aggregated counts saved to: {agg_filepath} ({len(final_aggregated_counts)} groups)"
                )
            else:
                logging.error("No aggregated counts to save!")

            # Gene/TE level TPM aggregation
            if not final_aggregated_counts.empty and args.output_tpm:
                logging.info("Aggregating TPMs by gene symbol and TE subfamily...")
                final_aggregated_tpm = _aggregate_genes_and_tes(
                    tpm_df, gene_map, te_map
                )

                if not final_aggregated_tpm.empty:
                    agg_tpm_filepath = f"{args.prefix}_total_EM_aggregated_TPM.tsv"
                    final_aggregated_tpm.to_csv(agg_tpm_filepath, sep="\t")
                    logging.info(f"Final aggregated TPMs saved to: {agg_tpm_filepath}")

            # Add group confidence metrics (using integer versions)
            if args.calculate_group_confidence and not final_aggregated_counts.empty:
                logging.info("\n=== CALCULATING GROUP-LEVEL CONFIDENCE METRICS ===")
                try:
                    group_metrics = run_group_confidence_from_pipeline_parallel_optimized(
                        final_total_counts=final_total_counts,  # Integer IDs
                        final_aggregated_counts=final_aggregated_counts,
                        confidence_results=confidence_results,  # Integer IDs
                        distinguishability_results=distinguishability_results,  # Integer IDs
                        junction_confidence_results=junction_confidence_results,  # Integer IDs
                        gene_map=gene_map,  # Integer IDs
                        te_map=te_map,  # Integer IDs
                        args=args,
                        int_to_string=int_to_string,
                        string_to_int=string_to_int,
                    )
                except Exception as e:
                    logging.error(f"Failed to calculate group confidence metrics: {e}")
                    import traceback

                    traceback.print_exc()
        else:
            logging.error(
                "Could not perform final aggregation due to GTF parsing errors."
            )

    # Write the COMPLETE run manifest file
    if (
        "cache_path_manifest" in pipeline_context
        and pipeline_context["cache_path_manifest"]
    ):
        manifest_path = f"{args.prefix}_run_manifest.json"
        manifest_data = {
            "run_prefix": args.prefix,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "annotation_file": os.path.abspath(args.annotation),
            "cache_files": pipeline_context["cache_path_manifest"],
            "summary_files": {
                "transcript_counts": os.path.abspath(
                    f"{args.prefix}_total_EM_counts.tsv"
                ),
                "transcript_tpm": os.path.abspath(f"{args.prefix}_total_EM_TPM.tsv"),
                "aggregated_counts": os.path.abspath(
                    f"{args.prefix}_total_EM_aggregated_counts.tsv"
                ),
                "aggregated_tpm": os.path.abspath(
                    f"{args.prefix}_total_EM_aggregated_TPM.tsv"
                ),
                "group_confidence": os.path.abspath(
                    f"{args.prefix}_group_confidence_comprehensive.tsv.gz"
                ),
            },
            "per_sample_files": run_file_map,
        }
        # Add error handling in case some files weren't generated
        for key, path in manifest_data["summary_files"].items():
            if not os.path.exists(path):
                logging.warning(
                    f"Manifest warning: summary file for '{key}' not found at '{path}'."
                )

        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=4)  # Use indent=4 for better readability
        logging.info(f"✅ Saved COMPLETE run manifest to: {manifest_path}")

    elapsed_time = time.time() - start_time
    logging.info(f"\n=== PIPELINE COMPLETED IN {elapsed_time/60:.1f} minutes ===")
    logging.info(
        f"Junction-aware: {'Yes' if pipeline_context['junction_map'] else 'No'}"
    )
    logging.info(f"Annotation: {os.path.basename(args.annotation)}")
    logging.info(f"Transcripts: {len(int_to_string):,}")


if __name__ == "__main__":
    main()
