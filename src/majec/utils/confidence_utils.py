import pandas as pd
from collections import defaultdict
import time
import numpy as np
import logging
import itertools
from collections import Counter


def decode_transcript_ids(data, int_to_string, copy=True):
    """
    Convert integer transcript IDs back to strings.

    Args:
        data: Can be DataFrame, Series, dict, or dict of dicts
        int_to_string: List mapping integer IDs to strings
        copy: Whether to copy the data structure

    Returns:
        Data structure with string IDs
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy() if copy else data
        if df.index.dtype in ["int64", "int32"]:  # Integer index
            df.index = df.index.map(
                lambda x: int_to_string[x] if 0 <= x < len(int_to_string) else "UNKNOWN"
            )
        return df

    elif isinstance(data, pd.Series):
        s = data.copy() if copy else data
        if s.index.dtype in ["int64", "int32"]:  # Integer index
            s.index = s.index.map(
                lambda x: int_to_string[x] if 0 <= x < len(int_to_string) else "UNKNOWN"
            )
        s.name = data.name
        return s

    elif isinstance(data, dict):
        # Check what type of keys we have
        first_key = next(iter(data.keys())) if data else None

        if first_key and isinstance(first_key, tuple):
            # Dict with tuple keys (equivalence classes)
            decoded = {}
            for feature_tuple, value in data.items():
                decoded_tuple = tuple(
                    int_to_string[f] if 0 <= f < len(int_to_string) else "UNKNOWN"
                    for f in feature_tuple
                )
                decoded[decoded_tuple] = value
            return decoded

        elif first_key and isinstance(first_key, frozenset):
            # NEW: Dict with frozenset keys (multimapper classes)
            decoded = {}
            for feature_set, value in data.items():
                decoded_set = frozenset(
                    int_to_string[f] if 0 <= f < len(int_to_string) else f"UNKNOWN_{f}"
                    for f in feature_set
                )
                decoded[decoded_set] = value
            return decoded

    return data


def decode_dataframe_columns(df, int_to_string_map, columns_to_decode):
    """
    Decodes integer IDs into strings for specified columns in a DataFrame.

    This is a general-purpose decoder that can handle:
    - single integers
    - lists, tuples, sets, or frozensets of integers
    - dictionaries where the keys are integers
    """
    if df.empty or not int_to_string_map:
        return df

    df_copy = df.copy()

    def _decoder(item):
        """Helper function to decode various data structures."""
        # Check for non-data values first
        if item is None or (isinstance(item, float) and np.isnan(item)):
            return item

        if isinstance(item, float) and item.is_integer():
            item = int(item)

        if isinstance(item, (int, np.integer)):
            if 0 <= item < len(int_to_string_map):
                return int_to_string_map[item]
            else:
                return f"UNKNOWN_ID_{item}"

        if isinstance(item, (list, tuple, set, frozenset)):
            return type(item)(
                (
                    int_to_string_map[i]
                    if 0 <= i < len(int_to_string_map)
                    else f"UNKNOWN_ID_{i}"
                )
                for i in item
            )

        elif isinstance(item, dict):
            # The map is a list, so we use bracket indexing with a safety check
            return {
                (
                    int_to_string_map[k]
                    if 0 <= k < len(int_to_string_map)
                    else f"UNKNOWN_ID_{k}"
                ): v
                for k, v in item.items()
            }
        elif isinstance(item, (int, np.integer)):
            # The map is a list, so we use bracket indexing with a safety check
            if 0 <= item < len(int_to_string_map):
                return int_to_string_map[item]
            else:
                return f"UNKNOWN_ID_{item}"

        else:
            # Return other types (like strings, floats) as is
            return item

    for col in columns_to_decode:
        if col in df_copy.columns:
            # Use .apply() to run our decoder on every element in the column
            df_copy[col] = df_copy[col].apply(_decoder)

    return df_copy


def generate_isoform_discrimination_report(
    junction_confidence_df,
    gene_to_transcripts_map,
    junction_metrics,
    prefix,
    sample_name,
    int_to_string=None,
    junction_int_to_string=None,
):
    """
    Generate reports showing which isoforms can be discriminated by junctions.

    Args:
        junction_confidence_df: Dict of junction confidence metrics per transcript
        gene_to_transcripts_map: Dict mapping gene_id -> list of transcript_ids
        junction_metrics: Dict of detailed junction metrics (includes observed_junctions)
        prefix: Output file prefix
        sample_name: Sample name for output files
    """
    discrimination_reports = []

    for gene_id, transcript_list in gene_to_transcripts_map.items():
        if len(transcript_list) < 2:
            continue  # Skip single-isoform genes

        # Only analyze transcripts that have junction data
        transcripts_with_junctions = [
            t
            for t in transcript_list
            if t in junction_confidence_df and t in junction_metrics
        ]

        if len(transcripts_with_junctions) < 2:
            continue

        report = analyze_gene_isoform_discrimination(
            gene_id, transcripts_with_junctions, junction_metrics
        )
        discrimination_reports.append(report)

    # Save detailed report if there's meaningful data
    if discrimination_reports:
        # Summary statistics
        summary_data = []
        for report in discrimination_reports:
            summary_data.append(
                {
                    "gene": report["gene"],
                    "n_isoforms": report["n_isoforms"],
                    "n_with_unique_junctions": report["n_with_unique_junctions"],
                    "n_discriminable_pairs": sum(
                        1 for p in report["isoform_pairs"] if p["can_discriminate"]
                    ),
                    "total_pairs": len(report["isoform_pairs"]),
                }
            )

        summary_df = decode_transcript_ids(pd.DataFrame(summary_data), int_to_string)
        summary_output = f"{prefix}_{sample_name}_isoform_discrimination_summary.tsv"
        decode_transcript_ids(summary_df, int_to_string).to_csv(
            summary_output, sep="\t", index=False
        )

        logging.info(
            f"[{sample_name}] Generated isoform discrimination report for {len(discrimination_reports)} genes"
        )

        # Optional: Save detailed pairwise comparisons for genes with many isoforms
        detailed_data = []
        for report in discrimination_reports:
            if len(report["isoform_pairs"]) > 1:  # Multi-isoform genes
                for pair in report["isoform_pairs"]:
                    pair_data = pair.copy()
                    pair_data["gene"] = report["gene"]
                    detailed_data.append(pair_data)

        if detailed_data:
            # 1. Decode transcript IDs as before
            detailed_df = decode_transcript_ids(
                pd.DataFrame(detailed_data), int_to_string
            )

            # 2. Decode the junction sets using the newly passed map
            if junction_int_to_string:
                # First, ensure we are actually saving the sets and not just their lengths.
                # (We must have also modified analyze_gene_isoform_discrimination to save the integer sets)
                columns_to_decode = ["unique_to_t1", "unique_to_t2", "shared_junctions"]
                detailed_df = decode_dataframe_columns(
                    detailed_df, junction_int_to_string, columns_to_decode
                )

            # 3. Save the fully decoded DataFrame
            detailed_output = (
                f"{prefix}_{sample_name}_isoform_discrimination_detailed.tsv.gz"
            )
            detailed_df.to_csv(
                detailed_output,
                sep="\t",
                compression={"method": "gzip", "compresslevel": 1},
                index=False,
            )


def analyze_gene_isoform_discrimination(gene_id, transcript_list, junction_metrics):
    """
    Analyze junction-based discrimination for a single gene's isoforms.

    Args:
        gene_id: Gene identifier
        transcript_list: List of transcript IDs to compare
        junction_metrics: Dict with 'observed_junctions' sets per transcript
    """
    report = {
        "gene": gene_id,
        "n_isoforms": len(transcript_list),
        "n_with_unique_junctions": 0,
        "isoform_pairs": [],
    }

    # Count isoforms with unique junctions
    for transcript in transcript_list:
        if transcript in junction_metrics:
            n_unique = junction_metrics[transcript].get("n_unique_junctions", 0)
            if n_unique > 0:
                report["n_with_unique_junctions"] += 1

    # Pairwise comparison using pre-computed junction sets
    for t1, t2 in itertools.combinations(transcript_list, 2):
        # Get observed junctions for each transcript
        t1_junctions = junction_metrics.get(t1, {}).get("observed_junctions", {}).keys()
        t2_junctions = junction_metrics.get(t2, {}).get("observed_junctions", {}).keys()

        # Ensure they're sets
        if not isinstance(t1_junctions, set):
            t1_junctions = set(t1_junctions)
        if not isinstance(t2_junctions, set):
            t2_junctions = set(t2_junctions)

        # Calculate discrimination
        unique_to_t1 = t1_junctions - t2_junctions
        unique_to_t2 = t2_junctions - t1_junctions
        shared = t1_junctions & t2_junctions

        # Discrimination score
        discrimination_score = (len(unique_to_t1) + len(unique_to_t2)) / (
            len(shared) + 1
        )

        report["isoform_pairs"].append(
            {
                "transcript1": t1,
                "transcript2": t2,
                "unique_to_t1": len(unique_to_t1),
                "unique_to_t2": len(unique_to_t2),
                "shared_junctions": len(shared),
                "discrimination_score": discrimination_score,
                "can_discriminate": len(unique_to_t1) > 0 or len(unique_to_t2) > 0,
            }
        )

    return report


def track_assignment_entropy_vectorized(
    unique_groups, multi_groups, current_priors, min_expression=1.0, sample_name=None
):
    """
    Track assignment entropy using vectorized operations instead of parallel processing.
    More efficient for within-sample processing.
    """
    start_time = time.time()
    sample_name = current_priors.name if hasattr(current_priors, "name") else "unknown"

    # Pre-filter by expression level
    expressed_mask = current_priors >= min_expression
    expressed_transcripts = set(current_priors[expressed_mask].index)
    total_transcripts = len(current_priors)
    tracked_transcripts = len(expressed_transcripts)
    transcript_unique_reads = defaultdict(float)
    transcript_shared_reads = defaultdict(float)

    logging.info(
        f"[{sample_name}] Tracking entropy for {tracked_transcripts:,}/{total_transcripts:,} "
        f"transcripts with expression >= {min_expression} (vectorized)"
    )

    # Initialize assignment tracking
    transcript_assignments = defaultdict(lambda: defaultdict(float))

    # Convert priors to numpy array for faster access
    priors_array = current_priors.values
    priors_index = current_priors.index.values
    priors_lookup = {idx: i for i, idx in enumerate(priors_index)}

    # Process unique mapper groups
    for feature_tuple, read_count in unique_groups.items():
        if len(feature_tuple) == 1:
            transcript = feature_tuple[0]
            if transcript in expressed_transcripts:
                transcript_assignments[transcript]["unique_singleton"] += read_count
                transcript_unique_reads[transcript] += read_count
        else:
            # Vectorized weight calculation
            features = np.array(feature_tuple)
            feature_indices = np.array([priors_lookup.get(f, -1) for f in features])
            valid_mask = feature_indices >= 0

            if np.any(valid_mask):
                valid_indices = feature_indices[valid_mask]
                valid_features = features[valid_mask]

                # Check if any are expressed
                expressed_mask_local = np.array(
                    [f in expressed_transcripts for f in valid_features]
                )
                if np.any(expressed_mask_local):
                    weights = priors_array[valid_indices]
                    total_weight = np.sum(weights)

                    if total_weight > 0:
                        class_id = f"um_{hash(feature_tuple)}"
                        contributions = (weights / total_weight) * read_count

                        for feat, contrib, is_expr in zip(
                            valid_features, contributions, expressed_mask_local
                        ):
                            if is_expr:
                                transcript_assignments[feat][class_id] += contrib
                                transcript_shared_reads[feat] += read_count

    # Process multi-mapper groups
    for feature_set, read_count in multi_groups.items():
        if len(feature_set) == 1:
            transcript_id = next(iter(feature_set))
            transcript_assignments[transcript_id]["unique_multimapper"] += read_count
            transcript_shared_reads[feat] += read_count
        else:
            # Similar vectorized processing
            features = np.array(list(feature_set))
            feature_indices = np.array([priors_lookup.get(f, -1) for f in features])
            valid_mask = feature_indices >= 0

            if np.any(valid_mask):
                valid_indices = feature_indices[valid_mask]
                valid_features = features[valid_mask]

                expressed_mask_local = np.array(
                    [f in expressed_transcripts for f in valid_features]
                )
                if np.any(expressed_mask_local):
                    weights = priors_array[valid_indices]
                    total_weight = np.sum(weights)

                    if total_weight > 0:
                        class_id = f"mm_{hash(frozenset(feature_set))}"
                        contributions = (weights / total_weight) * read_count

                        for feat, contrib, is_expr in zip(
                            valid_features, contributions, expressed_mask_local
                        ):
                            if is_expr:
                                transcript_assignments[feat][class_id] += contrib
                                transcript_shared_reads[feat] += read_count

    elapsed = time.time() - start_time
    logging.info(
        f"[{sample_name}] Entropy tracking completed in {elapsed:.1f} seconds (vectorized)"
    )

    return dict(transcript_assignments)


def calculate_confidence_metrics(
    transcript_assignments, total_counts, min_count_threshold=1.0  # shared_fractions,
):
    """
    Calculate entropy-based confidence metrics for each transcript.
    """
    confidence_metrics = {}

    for transcript, assignments in transcript_assignments.items():
        if (
            transcript not in total_counts
            or total_counts[transcript] < min_count_threshold
        ):
            continue

        # Calculate entropy across assignment sources
        total_assigned = sum(assignments.values())

        if total_assigned > 0:
            entropy = 0
            n_sources = len(assignments)

            # Calculate Shannon entropy
            for source, count in assignments.items():
                p = count / total_assigned
                if p > 0:
                    entropy -= p * np.log2(p)

            # Useful derived metrics
            unique_singleton_fraction = (
                assignments.get("unique_singleton", 0) / total_assigned
            )
            unique_multimapper_fraction = (
                assignments.get("unique_multimapper", 0) / total_assigned
            )
            # Derive the shared fraction. This is the only way to guarantee correctness.
            shared_frac = 1.0 - (
                unique_singleton_fraction + unique_multimapper_fraction
            )
            max_entropy = np.log2(n_sources) if n_sources > 1 else 0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            transcript_counts = float(total_counts[transcript])

            confidence_metrics[transcript] = {
                "entropy": entropy,
                "confidence_score": 1 / (1 + entropy),  # Simple transformation
                "n_sources": n_sources,
                "unique_fraction": unique_singleton_fraction,
                "normalized_entropy": normalized_entropy,
                "total_count": transcript_counts,
                "shared_read_fraction": shared_frac,
                "unique_multimapper_fraction": unique_multimapper_fraction,
            }

    return confidence_metrics


def calculate_differential_confidence_vectorized(
    unique_mapper_groups,
    multimapper_classes,
    current_priors,
    expressed_transcripts,
    gene_map=None,
    te_map=None,
    sample_name=None,
    dominance_threshold=0.5,
):
    """
    Fully vectorized version without internal parallelization.
    More efficient when called from already-parallel sample processing.
    """
    start_time = time.time()
    if sample_name is None:
        sample_name = (
            current_priors.name if hasattr(current_priors, "name") else "unknown"
        )

    logging.info(
        f"[{sample_name}] Calculating distinguishability for {len(expressed_transcripts):,} "
        f"transcripts (vectorized)"
    )

    # Create transcript-to-gene lookup
    transcript_to_gene = {}
    transcript_to_type = {}

    if gene_map is not None:
        gene_dict = gene_map.set_index("LocusID")["AggregateID"].to_dict()
        transcript_to_gene.update(gene_dict)
        for transcript in gene_dict:
            transcript_to_type[transcript] = "gene"

    if te_map is not None:
        te_dict = te_map.set_index("LocusID")["AggregateID"].to_dict()
        transcript_to_gene.update(te_dict)
        for transcript in te_dict:
            transcript_to_type[transcript] = "TE"

    # Convert to arrays for vectorized operations
    priors_array = current_priors.values
    priors_index = current_priors.index.values
    priors_lookup = {idx: i for i, idx in enumerate(priors_index)}

    # Collect all competitive relationships
    all_transcript_data = defaultdict(list)
    abs_inter_dist_scores = defaultdict(list)
    all_transcript_competitors = defaultdict(lambda: defaultdict(float))
    # Combine the two dictionaries explicitly to be safe.

    def _process_one_group_type(groups_dict):
        for feature_set, read_count in groups_dict.items():

            if len(feature_set) < 2:
                continue

            # Convert to array
            features = np.array(list(feature_set))

            # Get indices and filter
            feature_indices = np.array([priors_lookup.get(f, -1) for f in features])

            valid_mask = feature_indices >= 0

            if not np.any(valid_mask):
                continue

            valid_indices = feature_indices[valid_mask]
            valid_features = features[valid_mask]

            # Check expression
            expressed_mask = np.array(
                [f in expressed_transcripts for f in valid_features]
            )

            if not np.any(expressed_mask):
                continue

            # Vectorized weight calculation
            weights = priors_array[valid_indices]
            total_weight = np.sum(weights)

            if total_weight < 1e-9:
                continue

            allocations = weights / total_weight

            # For each expressed transcript in this class
            expressed_features = valid_features[expressed_mask]
            expressed_allocations = allocations[expressed_mask]

            class_genes = {transcript_to_gene.get(t_id) for t_id in valid_features}
            class_genes = {g for g in class_genes if g is not None}
            is_inter_genic_class = len(class_genes) > 1

            gene_allocations = None
            if is_inter_genic_class:
                # This expensive calculation now happens only ONCE per inter-genic class.
                gene_allocations = defaultdict(float)
                for t_int, alloc in zip(valid_features, allocations):
                    gene_allocations[transcript_to_gene.get(t_int, "UNKNOWN")] += alloc

            # Vectorized distinguishability calculation
            for i, (transcript, my_allocation) in enumerate(
                zip(expressed_features, expressed_allocations)
            ):
                # Calculate differences with all others
                other_mask = np.ones(len(allocations), dtype=bool)
                other_mask[np.where(valid_features == transcript)[0]] = False

                dominant_reads = 0

                if my_allocation > dominance_threshold:
                    dominance_credit = (my_allocation - dominance_threshold) / (
                        1 - dominance_threshold
                    )
                    dominant_reads = dominance_credit * my_allocation * read_count

                other_allocations = allocations[other_mask]
                other_features = valid_features[other_mask]

                # Vectorized metrics
                diffs = np.abs(my_allocation - other_allocations)
                avg_diff = np.mean(diffs)
                min_diff = np.min(diffs)

                class_weight = read_count / priors_array[priors_lookup[transcript]]

                # --- Calculate Absolute Inter-Genic Score (with ELSE block) ---
                if is_inter_genic_class:
                    target_gene = transcript_to_gene.get(transcript, "UNKNOWN")
                    my_gene_total = gene_allocations[target_gene]
                    abs_score = abs(my_gene_total - (1.0 - my_gene_total))
                    abs_inter_dist_scores[transcript].append((abs_score, class_weight))

                else:
                    # FAST PATH: Intra-genic class gets a perfect score.
                    abs_inter_dist_scores[transcript].append((1.0, class_weight))

                for competitor in other_features:
                    # Accumulate the weight for this specific transcript-competitor interaction
                    all_transcript_competitors[transcript][competitor] += class_weight

                all_transcript_data[transcript].append(
                    {
                        "class_size": len(features),
                        "my_share": my_allocation,
                        "avg_distinguishability": avg_diff,
                        "min_distinguishability": min_diff,
                        "class_weight": class_weight,
                        "competitors": list(other_features),
                        "dominant_reads": dominant_reads,
                        "gene_allocations": (
                            dict(gene_allocations) if gene_allocations else None
                        ),
                    }
                )

    logging.info(
        f"[{sample_name}] Processing {len(unique_mapper_groups)} unique mapper groups for distinguishability..."
    )
    _process_one_group_type(unique_mapper_groups)

    logging.info(
        f"[{sample_name}] Processing {len(multimapper_classes)} multi-mapper groups for distinguishability..."
    )
    _process_one_group_type(multimapper_classes)

    final_scores = {}

    # Loop over ALL expressed transcripts to ensure none are missed
    for transcript in expressed_transcripts:

        my_gene = transcript_to_gene.get(transcript, "UNKNOWN")
        my_type = transcript_to_type.get(transcript, "unknown")

        competitive_sets = all_transcript_data.get(transcript)

        # --- Case 1: The transcript had NO competitive interactions (it was unique-only) ---
        if not competitive_sets:
            final_scores[transcript] = {
                "distinguishability_score": 1.0,
                "min_distinguishability": np.nan,
                "ambiguous_fraction_abs_inter_dist": np.nan,
                "n_competitive_classes": 0,
                "n_unique_competitors": 0,
                "hardest_competitor": None,
                "hardest_competitor_gene": None,
                "strongest_external_competitor_gene": None,  # New
                "hardest_competitor_weight_frac": np.nan,
                "intra_gene_competition_frac": np.nan,
                "inter_gene_competition_frac": np.nan,
                "competition_type": "unique",
                "my_gene": my_gene,
                "transcript_type": my_type,
                "dominant_reads": 0.0,
            }
            continue

        # --- Case 2: The transcript has competitive data. Aggregate it in a SINGLE PASS. ---

        # Initialize all accumulators for this transcript
        total_class_weight = 0.0
        weighted_avg_dist_sum = 0.0
        weighted_min_dist_sum = 0.0
        total_dom_reads = 0.0
        weighted_abs_inter_dist_sum = 0.0

        competitor_weights = defaultdict(float)
        external_gene_weights = defaultdict(float)

        for interaction in competitive_sets:
            class_weight = interaction["class_weight"]
            total_class_weight += class_weight

            # Aggregate original distinguishability metrics
            weighted_avg_dist_sum += (
                interaction["avg_distinguishability"] * class_weight
            )
            weighted_min_dist_sum += (
                interaction["min_distinguishability"] * class_weight
            )
            total_dom_reads += interaction["dominant_reads"]

            # Aggregate individual competitor weights
            for competitor in interaction["competitors"]:
                competitor_weights[competitor] += class_weight

            # Aggregate absolute inter-genic score and external gene weights
            gene_allocs = interaction.get("gene_allocations")
            if gene_allocs:  # This was an inter-genic class
                my_gene_total = gene_allocs.get(my_gene, 0.0)
                abs_score = abs(my_gene_total - (1.0 - my_gene_total))
                weighted_abs_inter_dist_sum += abs_score * class_weight

                # Accumulate weights for external genes
                for gene, allocation in gene_allocs.items():
                    if gene != my_gene:
                        external_gene_weights[gene] += class_weight * allocation
            else:  # This was an intra-genic class
                weighted_abs_inter_dist_sum += 1.0 * class_weight

        # --- Now, calculate the final values from the accumulated sums ---

        # Final scores
        weighted_avg = (
            weighted_avg_dist_sum / total_class_weight
            if total_class_weight > 0
            else 1.0
        )
        weighted_min = (
            weighted_min_dist_sum / total_class_weight
            if total_class_weight > 0
            else 1.0
        )
        final_abs_inter_dist = (
            weighted_abs_inter_dist_sum / total_class_weight
            if total_class_weight > 0
            else 1.0
        )

        # Final competitor analysis
        hardest_competitor_id, strongest_weight = (
            max(competitor_weights.items(), key=lambda item: item[1])
            if competitor_weights
            else (None, 0.0)
        )
        total_comp_weight = sum(competitor_weights.values())
        intra_weight = sum(
            w
            for c, w in competitor_weights.items()
            if transcript_to_gene.get(c) == my_gene
        )
        frac_intra = intra_weight / total_comp_weight if total_comp_weight > 0 else 0.0
        frac_inter = 1.0 - frac_intra

        strongest_external_gene = (
            max(external_gene_weights.items(), key=lambda item: item[1])[0]
            if external_gene_weights
            else None
        )

        # Classify competition type based on the strongest competitor
        competitor_gene = transcript_to_gene.get(hardest_competitor_id, "UNKNOWN")
        competitor_type = transcript_to_type.get(hardest_competitor_id, "unknown")

        if my_gene == competitor_gene and my_gene != "UNKNOWN":
            competition_type = "intra-genic"
        elif my_type == "TE" and competitor_type == "gene":
            competition_type = "TE-to-gene"
        elif my_type == "gene" and competitor_type == "TE":
            competition_type = "gene-to-TE"
        elif my_type == "TE" and competitor_type == "TE":
            if my_gene != competitor_gene:
                competition_type = "inter-TE"
            else:
                competition_type = "intra-TE"
        elif my_type == "gene" and competitor_type == "gene":
            competition_type = "inter-genic"
        else:
            competition_type = "unknown"

        final_scores[transcript] = {
            "distinguishability_score": weighted_avg,
            "min_distinguishability": weighted_min,
            "ambiguous_fraction_abs_inter_dist": final_abs_inter_dist,
            "n_competitive_classes": len(competitive_sets),
            "n_unique_competitors": len(competitor_weights),
            "hardest_competitor": hardest_competitor_id,
            "hardest_competitor_gene": transcript_to_gene.get(
                hardest_competitor_id, "UNKNOWN"
            ),
            "strongest_external_competitor_gene": strongest_external_gene,
            "hardest_competitor_weight_frac": (
                strongest_weight / total_comp_weight if total_comp_weight > 0 else 0.0
            ),
            "intra_gene_competition_frac": frac_intra,
            "inter_gene_competition_frac": frac_inter,
            "competition_type": competition_type,
            "my_gene": my_gene,
            "transcript_type": my_type,
            "dominant_reads": total_dom_reads,
        }

    if final_scores:
        comp_types = Counter(
            score["competition_type"] for score in final_scores.values()
        )
        logging.info(f"[{sample_name}] Competition type breakdown:")
        for comp_type, count in sorted(comp_types.items()):
            logging.info(
                f"  - {comp_type}: {count} transcripts ({count/len(final_scores)*100:.1f}%)"
            )

    elapsed = time.time() - start_time
    logging.info(
        f"[{sample_name}] Distinguishability calculation completed in {elapsed:.1f} seconds (vectorized)"
    )

    return final_scores


def save_results_with_confidence(
    current_total_counts,
    confidence_metrics,
    distinguishability_metrics,
    junction_confidence_metrics,
    prefix,
    sample_name,
    sample_pre_em_total_counts,
    verbose_output=False,
    full_feature_set=None,
    int_to_string=None,
):
    """
    Optimized version that works with sparse data (only non-zero features).
    Expands some outputs to full set if provided.

    Args:
        current_total_counts: Series with counts (can be sparse - only non-zero)
        confidence_metrics: Dict of confidence metrics
        distinguishability_metrics: Dict of distinguishability metrics
        junction_confidence_metrics: Dict of junction metrics
        prefix: Output file prefix
        sample_name: Sample name
        verbose_output: Whether to save detailed outputs
        full_feature_set: Complete list of all features (for consistent output)
    """
    import time

    start_time = time.time()

    # Work only with expressed features
    expressed_mask = current_total_counts > 0
    expressed_counts = current_total_counts[expressed_mask]
    n_expressed = len(expressed_counts)
    n_total = (
        len(current_total_counts) if full_feature_set is None else len(full_feature_set)
    )

    logging.info(
        f"[{sample_name}] Processing confidence for {n_expressed:,} expressed features "
        f"(out of {n_total:,} total)"
    )

    # Create sparse output dataframe with only expressed features
    output_df_sparse = expressed_counts.to_frame("count")

    # Add confidence metrics (already sparse - only for expressed features)
    if confidence_metrics:
        conf_df = pd.DataFrame.from_dict(confidence_metrics, orient="index")
        # Only join on the features that exist in both
        output_df_sparse = output_df_sparse.join(conf_df, how="left")

        # Fill NaN only for expressed features
        for col, fill_value in [
            ("confidence_score", 0),
            ("entropy", np.inf),
            ("n_sources", 0),
            ("unique_fraction", 0),
        ]:
            if col in output_df_sparse.columns:
                output_df_sparse[col] = output_df_sparse[col].fillna(fill_value)

        if "n_sources" in output_df_sparse.columns:
            output_df_sparse["n_sources"] = output_df_sparse["n_sources"].astype(int)

    # Add distinguishability metrics
    if distinguishability_metrics:
        dist_df = pd.DataFrame.from_dict(distinguishability_metrics, orient="index")

        # Add key columns
        key_dist_cols = [
            "distinguishability_score",
            "n_unique_competitors",
            "min_distinguishability",
            "n_competitive_classes",
            "hardest_competitor",
            "competition_type",
            "hardest_competitor_gene",
            "dominant_reads",
            "hardest_competitor_weight_frac",
            "ambiguous_fraction_abs_inter_dist",
            "intra_gene_competition_frac",
            "inter_gene_competition_frac",
        ]

        for col in key_dist_cols:
            if col in dist_df.columns:
                output_df_sparse[col] = dist_df[col]

        # Handle unique mappers
        if (
            "confidence_score" in output_df_sparse.columns
            and "distinguishability_score" in output_df_sparse.columns
        ):
            unique_mappers = output_df_sparse["confidence_score"] == 1.0
            output_df_sparse.loc[unique_mappers, "distinguishability_score"] = (
                output_df_sparse.loc[unique_mappers, "distinguishability_score"].fillna(
                    2.0
                )
            )
            output_df_sparse["distinguishability_score"] = output_df_sparse[
                "distinguishability_score"
            ].fillna(-1)

        # Save detailed distinguishability (already sparse)
        if verbose_output and not dist_df.empty:
            # Only work with features that have distinguishability data
            dist_expressed = dist_df.index.intersection(expressed_counts.index)
            if len(dist_expressed) > 0:
                dist_output_df = dist_df.loc[dist_expressed].copy()
                dist_output_df["count"] = expressed_counts[dist_expressed]
                dist_output_df = dist_output_df.sort_values("count", ascending=False)
                dist_output_df = decode_transcript_ids(dist_output_df, int_to_string)

                # Save sparse version
                output_file = (
                    f"{prefix}_{sample_name}_distinguishability_detailed.tsv.gz"
                )
                dist_output_df.to_csv(
                    output_file,
                    sep="\t",
                    compression={"method": "gzip", "compresslevel": 1},
                )
                logging.info(
                    f"[{sample_name}] Saved detailed distinguishability for "
                    f"{len(dist_output_df):,} features to {output_file}"
                )

    # Add junction confidence metrics
    if junction_confidence_metrics:
        junc_df = pd.DataFrame.from_dict(junction_confidence_metrics, orient="index")

        key_junction_cols = [
            "junction_evidence_fraction",
            "has_unique_junction_support",
            "junction_confidence_category",
        ]
        for col in key_junction_cols:
            if col in junc_df.columns:
                output_df_sparse[col] = junc_df[col]

        # Fill defaults
        if "junction_evidence_fraction" in output_df_sparse.columns:
            output_df_sparse["junction_evidence_fraction"] = output_df_sparse[
                "junction_evidence_fraction"
            ].fillna(0)
        if "has_unique_junction_support" in output_df_sparse.columns:
            output_df_sparse["has_unique_junction_support"] = junc_df[
                "has_unique_junction_support"
            ].astype("boolean")
            output_df_sparse["has_unique_junction_support"] = output_df_sparse[
                "has_unique_junction_support"
            ].fillna(False)
        if "junction_confidence_category" in output_df_sparse.columns:
            output_df_sparse["junction_confidence_category"] = output_df_sparse[
                "junction_confidence_category"
            ].fillna("no_junctions")

    # Add shared read fraction
    # output_df_sparse['share_read_fraction'] = add_shared_read_metrics(output_df_sparse, unique_groups=, multi_groups)

    # Add discord scores
    output_df_sparse["discord_score"] = calculate_junction_read_discord(
        sample_pre_em_total_counts, current_total_counts
    )

    # Add overall confidence category
    output_df_sparse["overall_confidence"] = categorize_overall_confidence(
        output_df_sparse
    )

    # Summary statistics on sparse data (MUCH faster)
    logging.info(f"[{sample_name}] Confidence summary:")
    logging.info(f"  - Total transcripts: {n_total:,}")
    logging.info(f"  - Expressed transcripts: {n_expressed:,}")

    if "confidence_score" in output_df_sparse.columns:
        high_conf_count = (output_df_sparse["confidence_score"] > 0.8).sum()
        logging.info(f"  - High entropy confidence (>0.8): {high_conf_count}")

    if "distinguishability_score" in output_df_sparse.columns:
        high_dist_count = (output_df_sparse["distinguishability_score"] > 0.8).sum()
        unique_count = (output_df_sparse["distinguishability_score"] == 2.0).sum()
        output_df_sparse["dominant_fraction"] = (
            output_df_sparse["dominant_reads"] / output_df_sparse["count"]
        )
        if "confidence_score" in output_df_sparse.columns:
            output_df_sparse["strong_evidence_fraction"] = (
                output_df_sparse["dominant_fraction"].fillna(0.0)
                + output_df_sparse["unique_fraction"].fillna(0.0)
                + output_df_sparse["unique_multimapper_fraction"].fillna(0.0)
            )
            output_df_sparse["strong_evidence_fraction"] = output_df_sparse[
                "strong_evidence_fraction"
            ].clip(0, 1)
        logging.info(f"  - High distinguishability (>0.8): {high_dist_count}")
        logging.info(f"  - Unique mappers (dist=2.0): {unique_count}")

    if "has_unique_junction_support" in output_df_sparse.columns:
        junction_count = output_df_sparse["has_unique_junction_support"].sum()
        logging.info(f"  - Junction validated: {junction_count}")

    if "entropy" in output_df_sparse.columns:
        finite_entropy = output_df_sparse["entropy"][
            output_df_sparse["entropy"] != np.inf
        ]
        if len(finite_entropy) > 0:
            logging.info(f"  - Mean entropy: {finite_entropy.mean():.3f}")

    # NOW expand to full feature set only for final outputs
    if full_feature_set is not None:
        logging.info(f"[{sample_name}] Expanding to full feature set for output...")

        # Create full output with zeros
        output_df_full = pd.DataFrame(index=full_feature_set)

        # Copy all columns from sparse to full, filling missing with defaults
        for col in output_df_sparse.columns:
            if col == "count":
                output_df_full[col] = 0.0
            elif col == "confidence_score":
                output_df_full[col] = 0.0
            elif col == "entropy":
                output_df_full[col] = np.inf
            elif col == "n_sources":
                output_df_full[col] = 0
            elif col == "unique_fraction":
                output_df_full[col] = 0.0
            elif col == "distinguishability_score":
                output_df_full[col] = -1.0
            elif col == "n_unique_competitors":
                output_df_full[col] = -1
            elif col == "competition_type":
                output_df_full[col] = "not_expressed"
            elif col == "has_unique_junction_support":
                output_df_full[col] = False
            elif col == "junction_confidence_category":
                output_df_full[col] = "no_junctions"
            elif col == "overall_confidence":
                output_df_full[col] = "not_expressed"
            elif col == "dominant_fraction":
                output_df_full[col] = 0.0
            elif col == "unique_multimapper_fraction":
                output_df_full[col] = 0.0
            elif col == "dominant_reads":
                output_df_full[col] = 0.0
            else:
                # Generic default
                output_df_full[col] = None

        # Create holistic distinguishability
        output_df_sparse.rename(
            columns={
                "distinguishability_score": "ambiguous_fraction_distinguishability"
            },
            inplace=True,
        )
        is_unique_only = output_df_sparse["shared_read_fraction"] == 0
        output_df_sparse.loc[
            is_unique_only, "ambiguous_fraction_distinguishability"
        ] = 1.0
        output_df_sparse["distinguishability_score"] = (
            1 - output_df_sparse["shared_read_fraction"]
        ) * 1.0 + (
            output_df_sparse["shared_read_fraction"]
            * output_df_sparse["ambiguous_fraction_distinguishability"]
        )
        output_df_sparse["abs_inter_dist"] = (
            1 - output_df_sparse["shared_read_fraction"]
        ) * 1.0 + (
            output_df_sparse["shared_read_fraction"]
            * output_df_sparse["ambiguous_fraction_abs_inter_dist"]
        )

        # Update with actual values from sparse dataframe
        output_df_full.update(output_df_sparse)

        # Fix dtypes after update
        if "n_sources" in output_df_full.columns:
            output_df_full["n_sources"] = output_df_full["n_sources"].astype(int)
        if "has_unique_junction_support" in output_df_full.columns:
            output_df_full["has_unique_junction_support"] = output_df_full[
                "has_unique_junction_support"
            ].astype(bool)

        output_df = output_df_full
    else:
        output_df = output_df_sparse

    if not output_df_sparse.empty:
        metrics_file = f"{prefix}_{sample_name}_transcript_metrics_SPARSE.tsv.gz"  # Added SPARSP in name to be clear

        if "hardest_competitor" in output_df_sparse.columns:
            output_df_sparse = decode_dataframe_columns(
                output_df_sparse,
                int_to_string,
                ["hardest_competitor"],  # Just this one column needs decoding
            )

        # Decode and save.
        decode_transcript_ids(output_df_sparse, int_to_string).to_csv(
            metrics_file, compression={"method": "gzip", "compresslevel": 1}, sep="\t"
        )
        logging.info(
            f"[{sample_name}] Saved sparse metrics for DB ingestion to {metrics_file}"
        )

    # Save outputs
    if verbose_output:
        # Only sort the sparse version for speed
        output_df_sorted = decode_transcript_ids(
            output_df.sort_values("count", ascending=False), int_to_string
        )
        output_df_sorted.to_csv(
            f"{prefix}_{sample_name}_counts_with_confidence.tsv.gz",
            compression={"method": "gzip", "compresslevel": 1},
            sep="\t",
        )

    # High confidence subset (work with sparse, then expand if needed)
    high_conf_mask = pd.Series(False, index=output_df_sparse.index)

    if "confidence_score" in output_df_sparse.columns:
        high_conf_mask |= output_df_sparse["confidence_score"] > 0.8
    if "distinguishability_score" in output_df_sparse.columns:
        high_conf_mask |= output_df_sparse["distinguishability_score"] > 0.8
    if "has_unique_junction_support" in output_df_sparse.columns:
        high_conf_mask |= output_df_sparse["has_unique_junction_support"] == True

    if high_conf_mask.any():
        # Directly create the high-confidence set from the already-filtered sparse data.
        # No need to ever look at the big 'output_df_full' again.
        high_conf_df = output_df_sparse[high_conf_mask]
    else:
        # If no transcripts meet the stringent confidence criteria,
        # fall back to a simple abundance filter. This is now an explicit choice.
        logging.warning(
            f"[{sample_name}] No transcripts met high-confidence criteria. "
            f"Falling back to top expressed transcripts for high_confidence_counts.tsv.gz file."
        )
        # Take the top 1000 expressed transcripts or all with count > 10, whichever is smaller
        abundant_transcripts = output_df_sparse[output_df_sparse["count"] > 10]
        if len(abundant_transcripts) > 1000:
            high_conf_df = output_df_sparse.n_largest(1000, "count")
        else:
            high_conf_df = abundant_transcripts

    if not high_conf_df.empty:
        decode_transcript_ids(high_conf_df.copy(), int_to_string).to_csv(
            f"{prefix}_{sample_name}_high_confidence_counts.tsv.gz",
            compression={"method": "gzip", "compresslevel": 1},
            sep="\t",
        )
    elapsed = time.time() - start_time
    logging.info(f"[{sample_name}] Confidence output saved in {elapsed:.1f}s")

    return output_df_sparse


def analyze_competition_patterns(
    distinguishability_results, output_prefix, int_to_string=None
):
    """
    Analyze and visualize competition patterns across samples.
    """
    # Collect all competition data
    all_data = []
    for sample_idx, sample_dist in enumerate(distinguishability_results):
        sample_name = f"Sample_{sample_idx}"
        for transcript, metrics in sample_dist.items():
            if "competition_type" in metrics:
                all_data.append(
                    {
                        "sample": sample_name,
                        "transcript": transcript,
                        "gene": metrics.get("my_gene", "UNKNOWN"),
                        "competition_type": metrics["competition_type"],
                        "distinguishability": metrics["distinguishability_score"],
                        "hardest_competitor": metrics.get("hardest_competitor"),
                        "competitor_gene": metrics.get("hardest_competitor_gene"),
                    }
                )

    df = pd.DataFrame(all_data)

    # Summary statistics
    summary = (
        df.groupby(["competition_type"])
        .agg({"distinguishability": ["mean", "median", "std", "count"]})
        .round(3)
    )

    decode_transcript_ids(summary, int_to_string).to_csv(
        f"{output_prefix}_competition_summary.tsv", sep="\t"
    )

    # Identify genes with complex competition
    gene_competition = (
        df.groupby("gene")["competition_type"].value_counts().unstack(fill_value=0)
    )
    complex_genes = gene_competition[gene_competition.sum(axis=1) > 10].sort_values(
        by="intra-genic", ascending=False
    )

    decode_transcript_ids(complex_genes, int_to_string).to_csv(
        f"{output_prefix}_complex_competition_genes.tsv", sep="\t"
    )

    logging.info(
        f"Competition analysis saved to {output_prefix}_competition_summary.tsv"
    )

    return summary, complex_genes


import numpy as np
import pandas as pd


def categorize_overall_confidence(df):
    """
    Create an overall confidence category based on all metrics.
    Now includes shared read fraction in the logic.
    Fixed to ensure all conditions are boolean Series with consistent types.
    """
    # Check which columns are available
    has_entropy = "confidence_score" in df.columns
    has_dist = "distinguishability_score" in df.columns
    has_junction = "has_unique_junction_support" in df.columns
    has_sharing = "shared_read_fraction" in df.columns

    # Build conditions based on available metrics
    conditions = []

    # Gold standard: high on multiple metrics AND low sharing
    if has_entropy and has_dist and has_junction and has_sharing:
        conditions.append(
            (df["confidence_score"] > 0.8)
            & (
                (df["distinguishability_score"] > 0.8)
                | (df["has_unique_junction_support"].astype(bool))
            )
            & (df["shared_read_fraction"] < 0.5)  # Require low sharing for gold
        )
    elif has_entropy and has_dist and has_junction:
        conditions.append(
            (df["confidence_score"] > 0.8)
            & (
                (df["distinguishability_score"] > 0.8)
                | (df["has_unique_junction_support"].astype(bool))
            )
        )

    # Good: decent metrics but allow more sharing
    good_parts = []
    if has_entropy:
        good_parts.append(df["confidence_score"] > 0.8)
    if has_dist:
        good_parts.append(df["distinguishability_score"] > 0.8)
    if has_junction:
        good_parts.append(df["has_unique_junction_support"].astype(bool))

    if good_parts:
        # Use logical OR across all good parts
        good_condition = good_parts[0].copy()
        for part in good_parts[1:]:
            good_condition = good_condition | part

        # Exclude extreme sharing from "good"
        if has_sharing:
            good_condition = good_condition & (df["shared_read_fraction"] < 0.9)
        conditions.append(good_condition)

    # Medium: moderate scores or high sharing
    medium_parts = []
    if has_entropy:
        medium_parts.append(df["confidence_score"] > 0.5)
    if has_dist:
        medium_parts.append(df["distinguishability_score"] > 0.5)
    if has_sharing:
        # High sharing automatically drops to medium
        medium_parts.append(df["shared_read_fraction"] > 0.9)

    if medium_parts:
        # Use logical OR across all medium parts
        medium_condition = medium_parts[0].copy()
        for part in medium_parts[1:]:
            medium_condition = medium_condition | part
        conditions.append(medium_condition)

    # Low: everything else - create a boolean Series of correct length
    conditions.append(pd.Series([True] * len(df), index=df.index))

    # Adjust choices based on number of conditions
    all_choices = ["gold_standard", "good", "medium", "low"]
    choices = all_choices[: len(conditions)]

    # Convert all conditions to numpy arrays of same dtype
    conditions_array = [cond.values.astype(bool) for cond in conditions]

    # Provide a string default value to match the dtype of choices
    return pd.Series(
        np.select(conditions_array, choices, default="low"), index=df.index
    )


def calculate_junction_read_discord(pre_em_counts, post_em_counts):
    """
    Continuous discord score with smooth expression-based scaling.
    """
    discord_scores = {}

    for transcript in pre_em_counts.index:
        if transcript in post_em_counts.index:
            pre = pre_em_counts[transcript]
            post = post_em_counts[transcript]

            if pre > 0:
                reduction = pre / (post + 1)

                if reduction > 1:
                    # Base discord (log of reduction)
                    raw_discord = np.log10(reduction)

                    # Expression-based scaling (smooth function)
                    # Higher expression needs more reduction for same discord
                    expression_scale = 1 + np.log10(pre + 10) / 4

                    # Scaled discord
                    discord = raw_discord / expression_scale

                    # Smooth capping function (approaches 4 asymptotically)
                    discord = 4 * discord / (1 + discord)
                else:
                    discord = 0

                discord_scores[transcript] = discord
            else:
                discord_scores[transcript] = 0

    return discord_scores


def add_shared_read_metrics(df, unique_groups, multi_groups):
    """
    Calculate what fraction of reads are shared vs unique.
    """
    shared_fraction = {}

    for transcript in df.index:
        unique_reads = sum(
            count
            for group, count in unique_groups.items()
            if len(group) == 1 and group[0] == transcript
        )

        shared_reads = sum(
            count
            for group, count in unique_groups.items()
            if transcript in group and len(group) > 1
        )
        shared_reads += sum(
            count for group, count in multi_groups.items() if transcript in group
        )

        total = unique_reads + shared_reads
        if total > 0:
            shared_fraction[transcript] = shared_reads / total
        else:
            shared_fraction[transcript] = 1.0  # No reads = maximally uncertain

    df["shared_read_fraction"] = pd.Series(shared_fraction)

    # Penalize distinguishability when sharing is extreme
    df["adjusted_distinguishability"] = df["distinguishability_score"] * (
        1 - df["shared_read_fraction"]
    )

    return df


def process_and_save_sample_outputs(args_tuple):
    """
    Worker function to save all outputs for a single sample.
    To be used with a multiprocessing Pool.
    """
    (
        sample_name,
        sample_counts,
        sample_confidence,
        sample_distinguishability,
        sample_junction_confidence,
        sample_junction_metrics,
        sample_pre_em_total_counts,
        gene_to_transcripts_map,
        prefix,
        verbose_output,
        int_to_string,
        junction_int_to_string,
    ) = args_tuple

    logging.info(
        f"[{sample_name}] Saving comprehensive confidence and junction reports..."
    )

    full_feature_set_for_sample = sample_counts.index

    # Save comprehensive results with all metrics
    output_df = save_results_with_confidence(
        sample_counts,
        sample_confidence,
        sample_distinguishability,
        sample_junction_confidence,
        prefix,
        sample_name,
        sample_pre_em_total_counts,
        verbose_output=verbose_output,
        full_feature_set=full_feature_set_for_sample,
        int_to_string=int_to_string,
    )

    # Generate isoform discrimination report if we have junction data
    if verbose_output and sample_junction_confidence and gene_to_transcripts_map:
        generate_isoform_discrimination_report(
            sample_junction_confidence,
            gene_to_transcripts_map,
            sample_junction_metrics,
            prefix,
            sample_name,
            int_to_string=int_to_string,  # Pass maps explicitly
            junction_int_to_string=junction_int_to_string,
        )

    # Create junction support score using the junction confidence data already passed in
    junction_support = pd.Series(0.0, index=output_df.index)

    if sample_junction_confidence:
        # Convert to DataFrame for vectorized operations
        junc_df = pd.DataFrame.from_dict(sample_junction_confidence, orient="index")
        # Align with output_df index
        junc_df = junc_df.reindex(output_df.index)

        # Vectorized scoring (with safe column checking)
        if "has_unique_junction_support" in junc_df.columns:
            unique_mask = junc_df["has_unique_junction_support"] == True
            junction_support[unique_mask] = 1.0

            if "junction_specificity_score" in junc_df.columns:
                specific_mask = (~unique_mask) & (
                    junc_df["junction_specificity_score"] > 0
                )
                junction_support[specific_mask] = 0.5 + (
                    0.4 * junc_df.loc[specific_mask, "junction_specificity_score"]
                )

                if "n_shared_junctions" in junc_df.columns:
                    ambiguous_mask = (
                        (~unique_mask)
                        & (junc_df["junction_specificity_score"] <= 0)
                        & (junc_df["n_shared_junctions"] > 0)
                    )
                    junction_support[ambiguous_mask] = 0.25

    # Create DataFrame with just the columns we want
    summary_data = {f"{sample_name}_count": output_df["count"]}
    if "confidence_score" in output_df.columns:
        summary_data[f"{sample_name}_confidence"] = output_df["confidence_score"]
    if "distinguishability_score" in output_df.columns:
        summary_data[f"{sample_name}_distinguishability"] = output_df[
            "distinguishability_score"
        ]
    if "junction_support" in locals():
        summary_data[f"{sample_name}_junction_support"] = junction_support

    return (pd.DataFrame(summary_data), sample_name)


# New wrapper functions that maintain the same interface but use vectorized implementations
def track_assignment_entropy_parallel(
    unique_groups,
    multi_groups,
    current_priors,
    min_expression=1.0,
    threads_per_sample=1,
    sample_name=None,
):
    """
    Wrapper that maintains the parallel interface but uses vectorized implementation.
    The threads_per_sample parameter is ignored since vectorization is more efficient.
    """
    if threads_per_sample > 1:
        logging.debug(
            f"Ignoring threads_per_sample={threads_per_sample}, using vectorized implementation"
        )

    return track_assignment_entropy_vectorized(
        unique_groups,
        multi_groups,
        current_priors,
        min_expression,
        sample_name=sample_name,
    )


def calculate_differential_confidence_parallel(
    unique_mapper_groups,
    multimapper_classes,
    current_priors,
    expressed_transcripts,
    gene_map=None,
    te_map=None,
    threads_per_sample=1,
    sample_name=None,
):
    """
    Wrapper that maintains the parallel interface but uses vectorized implementation.
    The threads_per_sample parameter is ignored since vectorization is more efficient.
    """
    if threads_per_sample > 1:
        logging.debug(
            f"Ignoring threads_per_sample={threads_per_sample}, using vectorized implementation"
        )

    return calculate_differential_confidence_vectorized(
        unique_mapper_groups,
        multimapper_classes,
        current_priors,
        expressed_transcripts,
        gene_map,
        te_map,
        sample_name=sample_name,
    )
