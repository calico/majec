#!/usr/bin/env python
"""
Fixed version of group_confidence_metrics_parallel_integrated.py
Main change: Adaptive parallelization for single sample processing
"""

import pandas as pd
import numpy as np
import logging
import json
from collections import defaultdict, Counter
import multiprocessing
import time


def calculate_gini_coefficient(values):
    """
    Calculate Gini coefficient for expression dominance.
    0 = perfect equality, 1 = perfect inequality
    """
    if len(values) == 0 or np.sum(values) == 0:
        return np.nan

    # Sort values
    sorted_values = np.sort(values)
    n = len(values)

    # Calculate Gini
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (
        n + 1
    ) / n


def calculate_effective_copy_number(expression_values):
    """
    Calculate effective copy number (perplexity of expression distribution).
    Indicates how many transcripts effectively contribute to expression.
    """
    if len(expression_values) == 0 or np.sum(expression_values) == 0:
        return 0

    # Normalize to probabilities
    probs = expression_values / np.sum(expression_values)

    # Calculate entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # Convert to effective number
    return 2**entropy


def process_sample_vectorized(args):
    """
    Process all groups for a sample using vectorized operations.
    Unchanged from original.
    """
    (
        sample,
        sample_data,
        group_to_transcripts,
        aggregated_counts,
        gene_groups,
        te_groups,
        params,
    ) = args

    # Extract data once
    transcript_data = sample_data.get("transcript_data", {})
    confidence_data = sample_data.get("confidence_data", {})
    distinguishability_data = sample_data.get("distinguishability_data", {})
    junction_data = sample_data.get("junction_data", {})

    int_to_string = params.get("int_to_string")

    # Pre-allocate results dictionary
    group_metrics = {}

    # Get all group IDs that exist in aggregated counts
    valid_groups = [
        gid for gid in group_to_transcripts.keys() if gid in aggregated_counts.index
    ]

    # Process all groups for this sample
    for group_id in valid_groups:
        transcripts = group_to_transcripts[group_id]
        aggregated_count = aggregated_counts.loc[group_id, sample]

        # Vectorized expression filtering
        expr_values = np.array([transcript_data.get(t, 0) for t in transcripts])
        mask = expr_values >= params["min_expression_threshold"]

        if not np.any(mask):
            # No expressed transcripts - use minimal metrics
            group_metrics[group_id] = {
                "aggregated_count": aggregated_count,
                "n_expressed_transcripts": 0,
                "n_total_transcripts": len(transcripts),
                "weighted_unique_fraction": 0.0,
                "weighted_dominant_reads_fraction": 0.0,
                "weighted_shared_fraction": 0.0,
                "strong_evidence_fraction": 0.0,
                "weighted_confidence": 0,
                "high_conf_fraction": 0,
                "gini_coefficient": np.nan,
                "effective_copies": 0,
                "dominant_transcript": None,
                "dominant_fraction": 0,
                "has_junction_validation": False,
                "group_type": "gene" if group_id in gene_groups else "te",
                "intra_group_competition": 0,
                "inter_group_competition": 0,
                "main_external_competitor": None,
            }
            continue

        # Work with expressed transcripts only
        expressed_transcripts = np.array(transcripts)[mask]
        expressed_values = expr_values[mask]

        # Vectorized confidence calculation
        conf_scores = []
        for t in expressed_transcripts:
            if t in confidence_data:
                conf_scores.append(
                    (transcript_data.get(t, 0), confidence_data[t]["confidence_score"])
                )

        if conf_scores:
            weights = np.array([x[0] for x in conf_scores])
            scores = np.array([x[1] for x in conf_scores])

            if params["log_weight"]:
                weights = np.log1p(weights)

            weighted_conf = np.average(scores, weights=weights)
            high_conf_mask = scores > 0.8
            high_conf_fraction = (
                np.sum(weights[high_conf_mask]) / np.sum(weights)
                if np.sum(weights) > 0
                else 0
            )
        else:
            weighted_conf = 0
            high_conf_fraction = 0

        # Get the 'unique_fraction' for each expressed transcript from confidence_data.
        unique_fraction_scores = np.array(
            [
                confidence_data.get(t, {}).get("unique_fraction", 0.0)
                for t in expressed_transcripts
            ]
        )
        shared_fraction_scores = np.array(
            [
                confidence_data.get(t, {}).get("shared_read_fraction", 0.0)
                for t in expressed_transcripts
            ]
        )

        # Take expression weighted average
        group_weighted_unique_fraction = np.average(
            unique_fraction_scores, weights=expressed_values
        )
        group_weighted_shared_fraction = np.average(
            shared_fraction_scores, weights=expressed_values
        )

        # Get the 'dominant_reads' count for each expressed transcript from distinguishability_data.
        # This is an absolute count, not a fraction.
        dominant_reads_counts = np.array(
            [
                distinguishability_data.get(t, {}).get("dominant_reads", 0.0)
                for t in expressed_transcripts
            ]
        )

        # Sum the absolute dominant reads from all expressed transcripts in the group.
        total_dominant_reads_in_group = np.sum(dominant_reads_counts)
        group_weighted_dominant_reads_fraction = (
            total_dominant_reads_in_group / aggregated_count
        )

        # Competition analysis
        dist_scores = []
        intra_fracs = []
        inter_fracs = []
        external_competitor_votes = []
        abs_intergene_dist = []

        for t_id, t_expr in zip(expressed_transcripts, expressed_values):
            t_metrics = distinguishability_data.get(t_id, {})

            # Aggregate the competition fractions
            dist_scores.append(t_metrics.get("distinguishability_score", 1.0))
            intra_fracs.append(t_metrics.get("intra_gene_competition_frac", 0.0))
            inter_fracs.append(t_metrics.get("inter_gene_competition_frac", 0.0))
            abs_intergene_dist.append(
                t_metrics.get("ambiguous_fraction_abs_inter_dist", 1.0)
            )
            strongest_external = t_metrics.get("strongest_external_competitor_gene")
            if strongest_external is not None:
                # Add a weighted vote to our list
                external_competitor_votes.append((strongest_external, t_expr))

        main_external_competitor = None
        if external_competitor_votes:
            # Sum the expression "votes" for each external gene
            weighted_counts = defaultdict(float)
            for competitor, weight in external_competitor_votes:
                weighted_counts[competitor] += weight

            # The winner is the one with the highest summed expression weight
            main_external_competitor = max(
                weighted_counts.items(), key=lambda item: item[1]
            )[0]

        group_intra_competition = np.average(intra_fracs, weights=expressed_values)
        group_inter_competition = np.average(inter_fracs, weights=expressed_values)
        group_total_distinguishability = np.average(
            dist_scores, weights=expressed_values
        )
        group_total_abs_intergene_dist = np.average(
            abs_intergene_dist, weights=expressed_values
        )

        # Distribution metrics - already optimized
        gini = calculate_gini_coefficient(expressed_values)
        eff_copies = calculate_effective_copy_number(expressed_values)

        # Dominance
        dominant_idx = np.argmax(expressed_values)
        dominant_transcript_int = expressed_transcripts[dominant_idx]

        # Decode the dominant transcript ID for output
        if int_to_string and 0 <= int(dominant_transcript_int) < len(int_to_string):
            dominant_transcript = int_to_string[int(dominant_transcript_int)]
        else:
            dominant_transcript = str(int(dominant_transcript_int))  # Fallback

        dominant_fraction = expressed_values[dominant_idx] / np.sum(expressed_values)

        # Junction validation - vectorized check
        has_junction = any(
            junction_data.get(t, {}).get("has_unique_junction_support", False)
            for t in expressed_transcripts
        )

        # Store results
        group_metrics[group_id] = {
            "aggregated_count": aggregated_count,
            "n_expressed_transcripts": len(expressed_values),
            "n_total_transcripts": len(transcripts),
            "weighted_unique_fraction": group_weighted_unique_fraction,
            "weighted_dominant_reads_fraction": group_weighted_dominant_reads_fraction,
            "weighted_shared_fraction": group_weighted_shared_fraction,
            "strong_evidence_fraction": group_weighted_unique_fraction
            + group_weighted_dominant_reads_fraction,
            "weighted_confidence": weighted_conf,
            "high_conf_fraction": high_conf_fraction,
            "ambiguous_fraction_group_distinguishability": group_total_distinguishability,
            "intra_group_competition": group_intra_competition,
            "inter_group_competition": group_inter_competition,
            "holistic_group_distinguishability": (1 - group_weighted_shared_fraction)
            + (group_weighted_shared_fraction * group_total_distinguishability),
            "holistic_group_external_distinguishability": (
                1 - group_weighted_shared_fraction
            )
            + (group_weighted_shared_fraction * group_total_abs_intergene_dist),
            "ambiguous_fraction_external_distinguishability": group_total_abs_intergene_dist,
            "main_external_competitor": main_external_competitor,
            "gini_coefficient": gini,
            "effective_copies": eff_copies,
            "dominant_transcript": dominant_transcript,
            "dominant_fraction": dominant_fraction,
            "is_dominated": dominant_fraction > params["dominance_threshold"],
            "has_junction_validation": has_junction,
            "group_type": "gene" if group_id in gene_groups else "te",
        }

    return sample, group_metrics


def process_group_chunk(args):
    """
    Process a chunk of groups instead of all groups for a sample.
    This enables parallelization when processing a single sample.
    """
    (
        chunk_id,
        group_chunk,
        sample,
        sample_data,
        aggregated_counts,
        gene_groups,
        te_groups,
        params,
    ) = args

    # Create a subset of group_to_transcripts for this chunk
    group_to_transcripts_chunk = dict(group_chunk)

    # Reuse the sample processing logic but with just this chunk
    _, group_metrics = process_sample_vectorized(
        (
            sample,
            sample_data,
            group_to_transcripts_chunk,
            aggregated_counts,
            gene_groups,
            te_groups,
            params,
        )
    )

    return chunk_id, group_metrics


def save_outputs_parallel_optimized(
    all_sample_metrics,
    aggregated_counts,
    samples,
    output_prefix,
    summary_stats,
    n_processes=4,
    compression_level=1,
):
    """
    Make output - just the comprehensive file.
    """
    logging.info("Creating output files with optimized DataFrame building...")
    total_start_time = time.time()

    # Flatten the nested structure into records
    records = []
    for sample, sample_data in all_sample_metrics.items():
        for group_id, metrics in sample_data.items():
            record = {"sample": sample, "group_id": group_id}
            record.update(metrics)
            records.append(record)

    # Create a single DataFrame with all data
    all_data = pd.DataFrame(records)

    # Prepare comprehensive report
    comp_start = time.time()
    logging.info("Preparing comprehensive report...")

    comprehensive_df = all_data.copy()

    # Reorder columns for better readability
    column_order = ["sample", "group_id"] + [
        col for col in comprehensive_df.columns if col not in ["sample", "group_id"]
    ]
    comprehensive_df = comprehensive_df[column_order]

    comprehensive_file = f"{output_prefix}_group_confidence_comprehensive.tsv.gz"
    comprehensive_df.to_csv(
        comprehensive_file,
        sep="\t",
        index=False,
        compression={"method": "gzip", "compresslevel": compression_level},
    )

    logging.info(f"Wrote comprehensive report")

    # Save summary stats
    summary_file = f"{output_prefix}_group_confidence_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_stats, f, indent=2)

    total_elapsed = time.time() - total_start_time
    return total_elapsed


def calculate_aggregated_confidence_metrics_parallel_optimized(
    aggregated_counts=None,
    transcript_counts=None,
    confidence_results=None,
    distinguishability_results=None,
    junction_confidence_results=None,
    group_to_transcripts=None,
    gene_groups=None,
    te_groups=None,
    output_prefix="group_confidence",
    log_weight=False,
    dominance_threshold=0.5,
    min_expression_threshold=1.0,
    n_processes=None,
    int_to_string=None,
    string_to_int=None,
    **kwargs,
):
    """
    Fixed version with adaptive parallelization strategy.
    """

    mode_str = "Full"
    logging.info(
        f"=== Calculating Group-Level Confidence Metrics (Parallel Optimized, {mode_str} Mode) ==="
    )

    if n_processes is None:
        n_processes = min(multiprocessing.cpu_count(), 8)

    # Prepare data
    samples = aggregated_counts.columns.tolist()
    n_groups = len(group_to_transcripts)
    n_samples = len(samples)

    logging.info(
        f"Processing {n_samples} samples × {n_groups} groups = {n_samples * n_groups:,} combinations"
    )

    # Convert data to sample-indexed format for efficient access
    sample_data_dict = {}

    for i, sample in enumerate(samples):
        sample_data = {
            "transcript_data": {},
            "confidence_data": {},
            "distinguishability_data": {},
            "junction_data": {},
        }

        # Extract transcript counts
        if isinstance(transcript_counts, pd.DataFrame):
            if sample in transcript_counts.columns:
                sample_data["transcript_data"] = transcript_counts[sample].to_dict()
        elif isinstance(transcript_counts, list) and i < len(transcript_counts):
            data = transcript_counts[i]
            sample_data["transcript_data"] = (
                data.to_dict() if isinstance(data, pd.Series) else data
            )

        # Extract confidence data
        if isinstance(confidence_results, list) and i < len(confidence_results):
            sample_data["confidence_data"] = confidence_results[i]
        elif isinstance(confidence_results, dict):
            sample_data["confidence_data"] = confidence_results.get(sample, {})

        # Extract distinguishability data
        if isinstance(distinguishability_results, list) and i < len(
            distinguishability_results
        ):
            sample_data["distinguishability_data"] = distinguishability_results[i]
        elif isinstance(distinguishability_results, dict):
            sample_data["distinguishability_data"] = distinguishability_results.get(
                sample, {}
            )

        # Extract junction data
        if isinstance(junction_confidence_results, list) and i < len(
            junction_confidence_results
        ):
            sample_data["junction_data"] = junction_confidence_results[i]
        elif isinstance(junction_confidence_results, dict):
            sample_data["junction_data"] = junction_confidence_results.get(sample, {})

        sample_data_dict[sample] = sample_data

    # Parameters
    params = {
        "log_weight": log_weight,
        "dominance_threshold": dominance_threshold,
        "min_expression_threshold": min_expression_threshold,
        "int_to_string": int_to_string,
        "string_to_int": string_to_int,
    }

    start_time = time.time()

    # ADAPTIVE PARALLELIZATION
    if n_samples == 1 and n_groups > 1000:
        # SINGLE SAMPLE: Parallelize over groups instead of samples
        logging.info(
            f"Single sample detected - using GROUP-PARALLEL strategy with {n_processes} processes"
        )

        sample = samples[0]
        sample_data = sample_data_dict[sample]

        # Convert groups to list for chunking
        all_groups = list(group_to_transcripts.items())

        # Create chunks of groups
        chunk_size = max(
            100, len(all_groups) // n_processes
        )  # At least 100 groups per chunk
        group_chunks = []
        for i in range(0, len(all_groups), chunk_size):
            group_chunks.append(all_groups[i : i + chunk_size])

        logging.info(
            f"Split {len(all_groups)} groups into {len(group_chunks)} chunks (~{chunk_size} groups/chunk)"
        )

        # Create work items - one per chunk
        work_items = [
            (
                i,
                chunk,
                sample,
                sample_data,
                aggregated_counts,
                gene_groups,
                te_groups,
                params,
            )
            for i, chunk in enumerate(group_chunks)
        ]

        # Process chunks in parallel
        with multiprocessing.Pool(processes=min(n_processes, len(work_items))) as pool:
            chunk_results = pool.map(process_group_chunk, work_items)

        # Merge results from all chunks
        all_sample_metrics = {sample: {}}
        for chunk_id, chunk_metrics in chunk_results:
            all_sample_metrics[sample].update(chunk_metrics)

    else:
        # MULTIPLE SAMPLES: Use original sample-parallel strategy
        logging.info(
            f"Using optimized sample-level processing with {min(n_processes, n_samples)} processes"
        )

        # Create work items - one per sample
        work_items = [
            (
                sample,
                sample_data_dict[sample],
                group_to_transcripts,
                aggregated_counts,
                gene_groups,
                te_groups,
                params,
            )
            for sample in samples
        ]

        # Process samples in parallel
        if len(samples) > 1:
            with multiprocessing.Pool(processes=min(n_processes, len(samples))) as pool:
                results = pool.map(process_sample_vectorized, work_items)

            # Convert results to dictionary
            all_sample_metrics = dict(results)
        else:
            # Single sample - no need for multiprocessing overhead
            all_sample_metrics = dict([process_sample_vectorized(work_items[0])])

    processing_time = time.time() - start_time
    items_per_sec = (len(samples) * n_groups) / processing_time

    logging.info(
        f"Processed {len(samples) * n_groups:,} group-sample combinations in {processing_time:.1f}s "
        f"({items_per_sec:.0f} items/sec)"
    )

    # Save outputs using existing format
    logging.info("Creating output files...")

    # Prepare summary stats
    summary_stats = {
        "total_groups": len(aggregated_counts),
        "gene_groups": len(gene_groups) if gene_groups else 0,
        "te_groups": len(te_groups) if te_groups else 0,
        "mode": f"parallel_optimized_{mode_str.lower()}",
        "parallelization": {
            "strategy": "group-parallel" if n_samples == 1 else "sample-parallel",
            "n_processes": n_processes,
            "total_items": len(samples) * n_groups,
            "processing_time": processing_time,
            "items_per_second": items_per_sec,
        },
        "parameters": {
            "log_weight": log_weight,
            "dominance_threshold": dominance_threshold,
            "min_expression_threshold": min_expression_threshold,
        },
    }

    # Use the output function
    io_elapsed = save_outputs_parallel_optimized(
        all_sample_metrics,
        aggregated_counts,
        samples,
        output_prefix,
        summary_stats,
        n_processes=min(5, n_processes),
        compression_level=1,
    )

    logging.info(
        f"Total time: {processing_time + io_elapsed:.1f}s "
        f"(processing: {processing_time:.1f}s, I/O: {io_elapsed:.1f}s)"
    )

    return all_sample_metrics


def run_group_confidence_from_pipeline_parallel_optimized(
    final_total_counts,
    final_aggregated_counts,
    confidence_results,
    distinguishability_results,
    junction_confidence_results,
    gene_map,
    te_map,
    args,
    int_to_string=None,
    string_to_int=None,
):
    """
    Optimized pipeline integration function.
    """
    # Prepare mappings
    if te_map is not None and not te_map.empty:
        all_mappings = pd.concat([gene_map, te_map])
        te_groups = set(te_map["AggregateID"].unique())
    else:
        all_mappings = gene_map
        te_groups = set()

    group_to_transcripts = (
        all_mappings.groupby("AggregateID")["LocusID"].apply(list).to_dict()
    )
    gene_groups = (
        set(gene_map["AggregateID"].unique())
        if gene_map is not None and not gene_map.empty
        else set()
    )

    # Use all available threads
    n_processes = getattr(args, "threads", multiprocessing.cpu_count())

    # Call optimized function
    return calculate_aggregated_confidence_metrics_parallel_optimized(
        aggregated_counts=final_aggregated_counts,
        transcript_counts=final_total_counts,
        confidence_results=confidence_results,
        distinguishability_results=distinguishability_results,
        junction_confidence_results=junction_confidence_results,
        group_to_transcripts=group_to_transcripts,
        gene_groups=gene_groups,
        te_groups=te_groups,
        output_prefix=args.prefix,
        log_weight=getattr(args, "group_confidence_log_weight", False),
        dominance_threshold=getattr(args, "group_confidence_dominance_threshold", 0.5),
        min_expression_threshold=getattr(args, "confidence_min_expression", 1.0),
        n_processes=n_processes,
        int_to_string=int_to_string,
        string_to_int=string_to_int,
    )
