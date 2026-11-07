#!/usr/bin/env python
"""
Modified junction_processing.py - Fixed threading version
Main changes:
1. Replace multiprocessing.Pool with ThreadPoolExecutor to avoid nested process issues
2. Allow parallel processing even when called from worker processes
3. Add more aggressive parallelization for large files
"""

import pandas as pd
import numpy as np
import logging


def apply_junction_evidence(
    key_string,
    count,
    junction_string_to_int,
    junction_uniqueness_map,
    weighted_junction_evidence,
    junction_metrics,
    decay_exponent,
):
    """
    Apply evidence to transcripts based on junction mapping.
    This is the core logic shared between all versions.
    
    Returns True if junction was found in map, False otherwise.
    """
    junction_id = junction_string_to_int.get(key_string, -1)
    
    if junction_id != -1 and junction_id in junction_uniqueness_map:
        info = junction_uniqueness_map[junction_id]
        
        n_transcripts = info.get('n_transcripts', 1)
        uniqueness_weight = (1.0 / n_transcripts) ** decay_exponent
        final_evidence_score = count * uniqueness_weight 
        
        if final_evidence_score > 0:
            for transcript in info['transcripts']:
                weighted_junction_evidence[transcript] += final_evidence_score
                metrics = junction_metrics[transcript]
                metrics['junction_read_count'] += count
                if n_transcripts == 1:
                    metrics['unique_junction_reads'] += count
                else:
                    metrics['shared_junction_reads'] += count
                if junction_id not in metrics['observed_junctions']:
                    metrics['observed_junctions'][junction_id] = 0
                    if n_transcripts == 1:
                        metrics['n_unique_junctions'] += 1
                    else:
                        metrics['n_shared_junctions'] += 1
                
                metrics['observed_junctions'][junction_id] += count
        return True
    return False


def calculate_junction_evidence_from_metrics(junction_metrics, junction_map,
                                            junction_weight=1.0, decay_exponent=1.0):
    """
    Calculate junction evidence from pre-computed junction metrics.

    Note: junction_weight is accepted for API compatibility but NOT applied here.
    It is applied downstream in _apply_junction_boosts, matching the fresh path.

    Args:
        junction_metrics: Dict of transcript -> junction statistics
        junction_map: The junction uniqueness map
        junction_weight: Unused (kept for API compatibility)
        decay_exponent: Penalty exponent for shared junctions

    Returns:
        Dict of transcript -> raw evidence score (before junction_weight)
    """
    weighted_junction_evidence = {}
    
    for transcript, metrics in junction_metrics.items():
        if not metrics.get('observed_junctions'):
            continue

        total_evidence = 0.0

        # observed_junctions is now a dict of junction_key -> read_count
        for junction_key, junction_read_count in metrics['observed_junctions'].items():
            if junction_key in junction_map:
                junction_info = junction_map[junction_key]

                # Get junction sharing info
                n_transcripts = junction_info.get('n_transcripts', 1)

                # Calculate uniqueness weight with decay exponent
                uniqueness_weight = (1.0 / n_transcripts) ** decay_exponent

                # Calculate evidence using actual read count
                junction_score = junction_read_count * uniqueness_weight
                total_evidence += junction_score

        # Do NOT apply junction_weight here - it is applied downstream
        # in _apply_junction_boosts to match the fresh path behavior
        if total_evidence > 0:
            weighted_junction_evidence[transcript] = total_evidence
    
    return weighted_junction_evidence

# Keep other functions unchanged
def calculate_junction_confidence_metrics_vectorized(
    junction_evidence,
    junction_metrics_from_parsing, 
    total_counts
):
    """
    Post-process junction metrics using a vectorized approach.
    """
    if not junction_metrics_from_parsing:
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(junction_metrics_from_parsing, orient='index')
    
    # Add total counts and weighted scores
    df['total_count'] = total_counts.reindex(df.index, fill_value=0)
    df['weighted_junction_score'] = df.index.map(junction_evidence.get).fillna(0)
    
    # Vectorized calculations
    df['junction_evidence_fraction'] = df['junction_read_count'] / (df['total_count'] + 1)
    df['has_unique_junction_support'] = df['n_unique_junctions'] > 0
    
    # Junction specificity with safe division
    total_junction_reads = df['unique_junction_reads'] + df['shared_junction_reads']
    df['junction_specificity_score'] = np.divide(
        df['unique_junction_reads'], 
        total_junction_reads, 
        out=np.zeros_like(df['unique_junction_reads'], dtype=float), 
        where=total_junction_reads > 0
    )
    
    # Categories
    conditions = [
        df['n_unique_junctions'] > 0,
        df['junction_specificity_score'] > 0.5,
        df['n_shared_junctions'] > 0
    ]
    choices = [
        'unique_junction_validated',
        'specific_shared_junctions', 
        'ambiguous_shared_junctions'
    ]
    df['junction_confidence_category'] = np.select(conditions, choices, default='no_junctions')
    
    # Summary flags
    df['is_junction_supported'] = df['junction_read_count'] > 0
    df['is_highly_specific'] = df['junction_specificity_score'] > 0.8
    
    return df.to_dict(orient='index')

import numpy as np
import pandas as pd
import logging


def calculate_subset_penalties_holistic(
    modified_counts,
    subset_relationships,
    junction_metrics,
    junction_map,
    transcript_junction_order,
    min_penalty=0.1,
    subset_evidence_threshold=1.0,
    overdispersion=0.02,
    confidence_threshold=2.0,
    expression_scale_factor=20
):
    """
    Apply penalties to subset isoforms based on a true baseline established
    from all other transcripts' independent evidence.
    """

    # Helper function to calculate the baseline for a single transcript
    def _calculate_attributed_baseline(
        transcript_id,
        junctions_to_exclude,
        junction_metrics,
        junction_map,
        transcript_junction_order,
        modified_counts
    ):
        """
        Calculates per-junction expression baseline for a single transcript.

        Instead of a simple mean, it calculates the *attributable* reads at each
        independent junction based on the transcript's relative expression (the prior).
        """
        # Start with all expected junctions to avoid issues with missing junctions from the observed set 
        all_expected_junctions = transcript_junction_order.get(transcript_id, [])
        if not all_expected_junctions:
            return None

        # Find junctions that provide independent evidence (i.e., are not in the subset)
        independent_expected_junctions = [
            junc_key for junc_key in all_expected_junctions
            if junc_key not in junctions_to_exclude
        ]

        if not independent_expected_junctions:
            return None

        # Get the observed junctions
        observed_junctions = junction_metrics.get(transcript_id, {}).get('observed_junctions', {})

        attributed_counts = []
        target_expr = modified_counts.get(transcript_id, 0)

        for junc_key in independent_expected_junctions:
            # Get the observed count for this junction, defaulting to 0 if it was not seen.
            count = observed_junctions.get(junc_key, 0)
            if count == 0:
                attributed_counts.append(0.0)
                continue
        
            sharing_transcripts = junction_map.get(junc_key, {}).get('transcripts', [])
            if not sharing_transcripts: # This should never happen
                logging.error(f"INTERNAL ANNOTATION INCONSISTENCY: Junction {junc_key} exists in the transcript's "
                        "expected junction list but has no associated transcripts in the global junction_map. "
                        "This may indicate a corrupted annotation file or a bug in preprocess.py."
                    )
                continue

            # Sum the expression of all transcripts sharing this junction
            total_expr_at_junction = sum(modified_counts.get(tid, 0) for tid in sharing_transcripts)

            # Attribute reads based on relative expression
            attributed_count = count * (target_expr / total_expr_at_junction)
            attributed_counts.append(attributed_count)

        return np.mean(attributed_counts) if attributed_counts else None
    
    penalties_applied = 0
    penalty_details = {}

    for subset_id, details in subset_relationships.items():
        if subset_id not in modified_counts.index or modified_counts[subset_id] == 0:
            continue

        subset_detail = {
            'is_subset': True,
            'penalty': 1.0,
            'had_evidence': False,
            'reason': 'initial',
            'expected_from_others': 0,
            'actual_shared': 0,
            'z_score': np.nan
        }

        # --- Stage 1: Define the canonical set of junctions for comparison ---
        # Get the GROUND TRUTH list of junctions this subset is supposed to share.
        expected_shared_junctions = details.get('shared_junctions', [])

        subset_junction_keys = set(expected_shared_junctions)
        subset_detail['had_evidence'] = True

        contributing_transcripts = set()
        for junc_key in subset_junction_keys:
            sharing = junction_map.get(junc_key, {}).get('transcripts', [])
            contributing_transcripts.update(sharing)
        contributing_transcripts.discard(subset_id)

        if not contributing_transcripts:
            subset_detail['reason'] = 'no_other_transcripts_share_junctions'
            penalty_details[subset_id] = subset_detail
            continue

        # 1. Calculate the attributed baseline rate for each contributing transcript
        baseline_rates = {}
        for tid in contributing_transcripts:
            rate = _calculate_attributed_baseline(
                tid, subset_junction_keys, junction_metrics, junction_map, transcript_junction_order, modified_counts
            )
            if rate is not None:
                baseline_rates[tid] = rate

        if not baseline_rates:
            subset_detail['reason'] = 'no_contributors_had_independent_evidence'
            penalty_details[subset_id] = subset_detail
            continue
        
        observed_subset_junctions = junction_metrics.get(subset_id, {}).get('observed_junctions', {})
        # 2. Sum the baselines to create a total expected count
        total_expected_from_others = 0
        actual_shared_total = 0
        
        for junc_key in  expected_shared_junctions:
            count = observed_subset_junctions.get(junc_key, 0.0)
            actual_shared_total += count
            supporters = junction_map.get(junc_key, {}).get('transcripts', [])
            expected_at_this_junction = sum(
                baseline_rates[tid] for tid in supporters if tid in baseline_rates
            )
            total_expected_from_others += expected_at_this_junction

        subset_detail['expected_from_others'] = total_expected_from_others
        subset_detail['actual_shared'] = actual_shared_total
        subset_detail['n_baseline_contributors'] = len(baseline_rates)
        
        # modify expected by evidence threshold
        total_expected_from_others = total_expected_from_others * subset_evidence_threshold

        # 3. Compare and Penalize
        excess_reads = actual_shared_total - total_expected_from_others
        
        if total_expected_from_others <= 0:
            z_score = np.inf if excess_reads > 0 else -np.inf
        else:
            mean_baseline_per_junc = total_expected_from_others / len(expected_shared_junctions)
            effective_overdispersion = overdispersion * (1 + expression_scale_factor / (mean_baseline_per_junc + 1e-6))
            expected_variance = total_expected_from_others + effective_overdispersion * total_expected_from_others**2
            expected_sd = np.sqrt(expected_variance)
            z_score = excess_reads / (expected_sd + 1e-6)
        
        subset_detail['z_score'] = z_score

        if z_score < 0:
            penalty_factor = min_penalty
        elif z_score < confidence_threshold:
            penalty_factor = min_penalty + (1 - min_penalty) * (z_score / confidence_threshold)
        else:
            penalty_factor = 1.0

        original_count = modified_counts[subset_id]
        penalties_applied += 1

        subset_detail.update({
            'penalty': penalty_factor,
            'reason': 'penalty_applied',
            'original_count': original_count,
            'penalized_count': modified_counts[subset_id]
        })
        penalty_details[subset_id] = subset_detail

    logging.info(f"    Applied final subset penalties to {penalties_applied} transcripts")
    return penalty_details

def calculate_junction_completeness_position_aware_optimized(
    junction_metrics,
    transcript_junction_order,
    min_penalty=0.1,
    overdispersion=0.02,
    pseudocount=1,
    transcript_info=None,
    terminal_relax=False,
    terminal_recovery_rate=False,
    library_type='None',
    splice_competitor_map=None,
    junction_map=None,
    all_junction_counts=None,
    ):
    """
    Enhanced with multi-junction penalty logic (no new parameters exposed)
    """
    if splice_competitor_map is not None:
        all_junction_counts = {}
        for tid, metrics in junction_metrics.items():
            observed = metrics.get('observed_junctions', {})
            for junction, count in observed.items():
                # Take max if junction appears in multiple transcripts
                all_junction_counts[junction] = max(
                    all_junction_counts.get(junction, 0), count
                )
    # logging.info("making splice site maps...")  
    # donor_map, acceptor_map = build_splice_site_maps(junction_map)
    # logging.info("splice site map complete")


    # Pre-compute all the helper functions once
    def z_to_exponent(z):
        """Map z-score to exponent for ratio penalty."""
        if z >= -1.5:
            return 0.2
        elif z <= -6:
            return 1.0
        else:
            t = (-z - 1.5) / 4.5
            exponent = 0.2 + 0.8 * t**1.5
            return min(1.0, exponent)
    
    def calculate_combined_z(z_scores, weights, n_worst=3):
        """
        Smart combination that considers multiple bad junctions
        Internal parameters hardcoded for now
        """
        # Internal thresholds
        severity_threshold = -2.0
        
        # Take worst N junctions
        worst_n = min(n_worst, len(z_scores))
        worst_zs = z_scores[:worst_n]
        
        # Normalize weights for available junctions
        active_weights = weights[:worst_n]
        active_weights = np.array(active_weights) / np.sum(active_weights)
        
        # Calculate base weighted combination
        weighted_z = sum(w * z for w, z in zip(active_weights, worst_zs))
        
        # Count severely bad junctions
        bad_junctions = sum(1 for z in worst_zs if z < severity_threshold)
        
        if bad_junctions <= 1:
            # Single bad junction - protect against moderation
            return min(z_scores[0], weighted_z)
        else:
            # Multiple bad junctions - amplify the signal
            # Each additional bad junction beyond the first adds to severity
            penalty_multiplier = 1 + (bad_junctions - 1) * 0.15
            adjusted_z = weighted_z * penalty_multiplier
            
            # Still don't get more lenient than worst single junction
            return min(z_scores[0], adjusted_z)
    
    def calculate_penalty_with_recovery(counts, worst_indices, overdispersion, 
                                        pseudocount, min_penalty, strand, library_type, 
                                        terminal_relax, terminal_recovery_rate, 
                                        expected_junctions, splice_competitor_map, 
                                        all_junction_counts, raw_median_baseline):
        """Calculate penalty using median model with optional terminal recovery"""
        nonzero_counts = counts[counts > 0]
        if len(nonzero_counts) == 0:
            return min_penalty, -6.0, {'model': 'no_data'}
        
        median_coverage = raw_median_baseline
        used_rms=False
        if np.max(nonzero_counts) > 50 * median_coverage:
            median_coverage = np.sqrt(np.mean(nonzero_counts**2)) #  RMS instead of median when median way below max
            used_rms=True

        # Check for terminal recovery pattern
        recovery_point = None
        expected_recovery = None
        if terminal_relax and library_type != 'none' and len(counts) >= 3:
            recovery_point, expected_recovery = model_terminal_recovery(
                counts, strand, library_type, terminal_recovery_rate
            )
        
        # Calculate expected values with splice competition
        expected_values = np.full(len(counts), median_coverage)

        # Apply splice competition expectations
        if splice_competitor_map and all_junction_counts:
            for i, junction in enumerate(expected_junctions):
                if junction in splice_competitor_map:
                    competitors = splice_competitor_map[junction]
                    if competitors:
                        # Get competitor counts
                        competitor_counts = []
                        for comp_junction in competitors:
                            if comp_junction in all_junction_counts:
                                comp_count = all_junction_counts[comp_junction]
                                if comp_count > 0:
                                    competitor_counts.append(comp_count)
                        
                        if competitor_counts:
                            competitor_median = np.median(competitor_counts)
                            
                            # For single-junction transcripts, use only competitors
                            if len(counts) == 1:
                                expected_values[i] = competitor_median
                            else:
                                # For multi-junction, use the stricter standard
                                expected_values[i] = max(median_coverage, competitor_median)
        
        
        # Set expected values
        if expected_recovery is not None:
            # Use recovery model for dropout zone, median elsewhere
            expected_values = np.full(len(counts), median_coverage)
            for pos, expected in expected_recovery.items():
                expected_values[pos] = expected
            model_used = f"terminal_recovery_{library_type}"
        else:
            # Check if we used competition
            if np.any(expected_values != median_coverage):
                model_used = "median_with_competition"
            elif used_rms:
                model_used='RMS'
            else:
                model_used = "median"
        
        # Calculate z-scores for worst junctions
        z_scores = []
        for idx in worst_indices:
            count = counts[idx]
            expected = expected_values[idx]
            # Use slightly reduced overdispersion for splice competitors
            junction = expected_junctions[idx] if idx < len(expected_junctions) else None
            if junction and splice_competitor_map and junction in splice_competitor_map:
                # These share sequence context, so less technical variation
                effective_overdispersion = overdispersion * 0.5
            else:
                effective_overdispersion = overdispersion
                
            expected_sd = np.sqrt(expected + effective_overdispersion * expected**2) + 1e-6
            z_score = (count - expected) / expected_sd
            z_scores.append(z_score)

        # Default weights - more emphasis on worst, but consider others
        weights = [0.6, 0.3, 0.1]
        combined_z = calculate_combined_z(z_scores, weights)
        
        # Convert to penalty
        worst_count = counts[worst_indices[0]]
        expected_worst = expected_values[worst_indices[0]]
        ratio = (worst_count + pseudocount) / (expected_worst + pseudocount)
        exponent = z_to_exponent(combined_z)
        if worst_count < expected_worst:
            score = ratio ** exponent
        else:
            score = 1.0  # No penalty if meeting/exceeding expectations
                
        penalty_info = {
            'model': model_used,
            'recovery_point': recovery_point,
            'worst_z_scores': z_scores,
            'worst_expected_values': [expected_values[idx] for idx in worst_indices],
            'worst_actual_counts': [counts[idx] for idx in worst_indices],
            'median_coverage': median_coverage,
            'terminal_recovery_used': recovery_point is not None,
            'used_splice_competition': False,  # Will be set below
            'competitor_adjusted_positions': []
        }
        
        # Track which positions used splice competition
        competitor_positions = []
        for i, junction in enumerate(expected_junctions):
            if junction in splice_competitor_map and splice_competitor_map[junction]:
                # Check if this position actually got competitor-based expectation
                # (not overridden by terminal recovery)
                if expected_recovery is None or i not in expected_recovery:
                    if expected_values[i] != median_coverage:  # Actually used competition
                        competitor_positions.append(i)

        penalty_info['used_splice_competition'] = len(competitor_positions) > 0
        penalty_info['competitor_adjusted_positions'] = competitor_positions
        penalty_info['n_junctions_with_competitors'] = sum(1 for j in expected_junctions 
                                                        if j in splice_competitor_map 
                                                        and splice_competitor_map[j])

        return score, combined_z, penalty_info

    
    def model_terminal_recovery(counts, strand, library_type, recovery_rate=2.0):
        """
        Model expected recovery pattern at terminal exons based on library type.
        
        Parameters:
        -----------
        counts : np.array
            Junction read counts ordered 5' to 3' on the transcript
        strand : str
            '+' or '-' strand
        library_type : str
            'dT', 'polyA', 'random', or 'none'
        recovery_rate : float
            Controls recovery curve steepness (higher = more lenient)
            
        Returns:
        --------
        recovery_point : int or None
            Index where normal coverage begins
        expected_recovery : dict or None
            Expected counts in dropout zone {position: expected_count}
        """
        
        # No recovery model for 'none' library type
        if library_type == 'none':
            return None, None
        
        # Check if we have any data to work with
        nonzero_counts = counts[counts > 0]
        if len(nonzero_counts) == 0:
            return None, None
        
        median_coverage = np.median(nonzero_counts)
        normal_threshold = 0.8 * median_coverage
        
        # Determine dropout location based on library type and strand
        # dT/polyA: 5' dropout on + strand, 3' dropout on - strand
        # random: 3' dropout on + strand, 5' dropout on - strand
        if library_type in ['dT', 'polyA']:
            check_start = True  # Check from start on + strand
        elif library_type == 'random':
            check_start = False  # Check from start on - strand
        else:
            # Unknown library type
            return None, None
        
        # Find recovery point
        recovery_point = None
        
        if check_start:
            # Look for recovery from start of transcript
            for i in range(len(counts)):
                if counts[i] >= normal_threshold:
                    recovery_point = i
                    break
        else:
            # Look for recovery from end of transcript
            for i in range(len(counts)-1, -1, -1):
                if counts[i] >= normal_threshold:
                    recovery_point = i
                    break
        
        # No recovery point found or no dropout zone
        if recovery_point is None:
            return None, None
        
        # Check if there's actually a dropout zone
        if check_start and recovery_point == 0:
            return None, None  # No dropout at start
        if not check_start and recovery_point == len(counts) - 1:
            return None, None  # No dropout at end
        
        # Verify we have a real dropout pattern, not just random low coverage
        if check_start:
            dropout_zone = counts[:recovery_point]
            # Check for increasing trend toward recovery
            if len(dropout_zone) >= 2:
                # Simple check: is coverage generally increasing?
                first_half = dropout_zone[:len(dropout_zone)//2]
                second_half = dropout_zone[len(dropout_zone)//2:]
                first_half_mean = np.mean(first_half[first_half > 0]) if np.any(first_half > 0) else 0
                second_half_mean = np.mean(second_half[second_half > 0]) if np.any(second_half > 0) else 0
                
                if second_half_mean <= first_half_mean * 1.5:
                    # Not a clear recovery pattern
                    return None, None
        else:
            dropout_zone = counts[recovery_point+1:]
            # Check for decreasing trend away from recovery
            if len(dropout_zone) >= 2:
                first_half = dropout_zone[:len(dropout_zone)//2]
                second_half = dropout_zone[len(dropout_zone)//2:]
                first_half_mean = np.mean(first_half[first_half > 0]) if np.any(first_half > 0) else 0
                second_half_mean = np.mean(second_half[second_half > 0]) if np.any(second_half > 0) else 0
                
                if first_half_mean <= second_half_mean * 1.5:
                    # Not a clear dropout pattern
                    return None, None
        
        # Model expected counts in dropout zone
        expected_recovery = {}
        
        if check_start:
            # Dropout at start
            for i in range(recovery_point):
                distance_from_recovery = recovery_point - i
                # Exponential recovery model
                if recovery_rate > 0:
                    expected = median_coverage * (1 - np.exp(-recovery_rate / distance_from_recovery))
                else:
                    # recovery_rate = 0 means no recovery modeling (harsh)
                    expected = median_coverage
                expected_recovery[i] = max(1, expected)
        else:
            # Dropout at end
            for i in range(recovery_point + 1, len(counts)):
                distance_from_recovery = i - recovery_point
                # Exponential recovery model
                if recovery_rate > 0:
                    expected = median_coverage * (1 - np.exp(-recovery_rate / distance_from_recovery))
                else:
                    # recovery_rate = 0 means no recovery modeling (harsh)
                    expected = median_coverage
                expected_recovery[i] = max(1, expected)
        
        return recovery_point, expected_recovery

    
   # VECTORIZED PRE-FILTERING STEP
    # Build arrays for batch processing
    transcript_ids = []
    expected_junction_lists = []
    n_expected_list = []
   
    # Initialize stats
    stats = {
        'complete': 0,
        'partial': 0,
        'none': 0,
        'single_exon': 0,
        'mild_penalty': 0,
        'moderate_penalty': 0,
        'severe_penalty': 0,
        'simple_complete': 0,
        'complex_analyzed': 0,
        'terminal_recovery_used': 0,
        'pure_median_1pct': 0
    }

    for transcript_id, expected_junctions in transcript_junction_order.items():
        transcript_ids.append(transcript_id)
        expected_junction_lists.append(expected_junctions)
        n_expected_list.append(len(expected_junctions))
    
    transcript_ids = np.array(transcript_ids)
    n_expected_array = np.array(n_expected_list)
    
    # Categorize transcripts
    single_exon_mask = n_expected_array == 0
    multi_exon_mask = ~single_exon_mask

    # Initialize tracking lists
    no_junction_transcripts = []
    simple_complete_transcripts = []
    complex_transcripts = []
    
    # Initialize results
    scores = {}
    model_details = {}

    # Handle single-exon transcripts immediately 
    single_exon_tids = transcript_ids[single_exon_mask]
    scores.update(dict.fromkeys(single_exon_tids, 1.0))
    no_junction_transcripts.extend([(tid, 1.0) for tid in single_exon_tids])
    
    # Process multi-exon transcripts
    for i, tid in enumerate(transcript_ids[multi_exon_mask]):
        model_details[tid] = {}
        expected_junctions = expected_junction_lists[i]
        n_expected = n_expected_array[multi_exon_mask][i]
        
        observed = junction_metrics.get(tid, {})
        observed_junction_counts = observed.get('observed_junctions', {})

        counts = np.array([observed_junction_counts.get(j, 0) for j in expected_junctions])
        n_observed = np.sum(counts > 0)
        # 1. Calculate the baseline from the ORIGINAL raw counts first.
        nonzero_raw_counts = counts[counts > 0]
        raw_median_baseline = np.median(nonzero_raw_counts) if len(nonzero_raw_counts) > 0 else 0

        # All transcripts with junctions go through the full penalty model
        complex_transcripts.append((tid, expected_junctions, counts, raw_median_baseline))
    
    # Update stats
    stats['single_exon'] = int(np.sum(single_exon_mask))
    stats['simple_complete'] = len(simple_complete_transcripts)
    stats['complex_analyzed'] = len(complex_transcripts)
    
    # Process complex cases
    n_worst_junctions = 3  # Internal constant
    
    for tid, expected_junctions, counts, raw_median_baseline in complex_transcripts:
        n_expected = len(expected_junctions)
        n_observed = np.sum(counts > 0)
        
        # Get indices of worst N junctions
        worst_indices = np.argsort(counts)[:min(n_worst_junctions, len(counts))]
        worst_counts = counts[worst_indices]
        
        # Get strand information
        if transcript_info:
            # Get strand information from the reliable transcript_info dict
            strand = transcript_info.get(tid, {}).get('strand', '+')
        else:
            strand='+'
        
        # Initialize model details
        model_details[tid] = {
            'n_expected': n_expected,
            'n_observed': n_observed,
            'strand': strand,
            'worst_positions': worst_indices.tolist(),
            'worst_counts': worst_counts.tolist(),
            'n_worst_considered': len(worst_indices)
        }
        
        # Check <1% rule
        nonzero_counts = counts[counts > 0]
        if len(nonzero_counts) == 0:
            scores[tid] = min_penalty
            stats['none'] += 1
            stats['severe_penalty'] += 1
            model_details[tid]['model'] = 'no_junctions'
            model_details[tid]['penalty'] = min_penalty
            continue
        
        use_pure_median_1pct = np.any(nonzero_counts < 0.01 * raw_median_baseline)
        
        if use_pure_median_1pct:
            stats['pure_median_1pct'] += 1
            model_details[tid]['pure_median_1pct'] = True
        
        # Calculate penalty with terminal recovery
        penalty_score, z_score, penalty_info = calculate_penalty_with_recovery(
            counts, worst_indices, overdispersion, pseudocount, min_penalty,
            strand, library_type, terminal_relax, terminal_recovery_rate, 
            expected_junctions, splice_competitor_map, all_junction_counts, raw_median_baseline
        )
        
        # Apply minimum penalty
        final_score = max(penalty_score, min_penalty)
        scores[tid] = final_score
        
        # Update model details
        model_details[tid].update(penalty_info)
        model_details[tid]['penalty'] = final_score
        model_details[tid]['z_score'] = z_score
        
        # Update stats
        if penalty_info.get('terminal_recovery_used', False):
            stats['terminal_recovery_used'] += 1
        
        if penalty_info.get('used_splice_competition', False):
            stats['splice_competition_used'] = stats.get('splice_competition_used', 0) + 1
        
        # Add RMS tracking
        if penalty_info.get('model') == 'RMS':
            stats['rms_used'] = stats.get('rms_used', 0) + 1
                
        if n_observed == n_expected and z_score > -2:
            stats['complete'] += 1
        else:
            stats['partial'] += 1
        
        if z_score > -2:
            stats['mild_penalty'] += 1
        elif z_score > -4:
            stats['moderate_penalty'] += 1
        else:
            stats['severe_penalty'] += 1
    
    # Add simple complete transcripts
    for tid, score, n_expected, n_observed, z_score in simple_complete_transcripts:
        model_details[tid] = {
            'model': 'simple_complete',
            'penalty': score,
            'n_expected': n_expected,
            'n_observed': n_observed,
            'z_score': z_score
        }
    
    # Add no junction transcripts
    for tid, score in no_junction_transcripts:
        if score == 1.0:
            model_details[tid] = {'model': 'single_exon', 'penalty': 1.0}
        else:
            model_details[tid] = {'model': 'no_junctions', 'penalty': min_penalty}
    
    # Log summary statistics
    logging.info(f"Junction completeness calculated for {len(scores)} transcripts")
    logging.info(f"  Pre-filtered: {stats['simple_complete']} simple complete, "
                f"{stats['none']} no junctions, {stats['single_exon']} single-exon")
    logging.info(f"  Complex analysis: {stats['complex_analyzed']} transcripts")
    logging.info(f"  Model: median with terminal recovery")
    if stats['terminal_recovery_used'] > 0:
        logging.info(f"  Terminal recovery applied: {stats['terminal_recovery_used']} transcripts")
    if stats.get('rms_used', 0) > 0:
        logging.info(f"  RMS baseline used: {stats['rms_used']} transcripts")
    if 'splice_competition_used' in stats:
        logging.info(f"  Splice competition applied: {stats['splice_competition_used']} transcripts")
    if stats['pure_median_1pct'] > 0:
        logging.info(f"  <1% rule triggered: {stats['pure_median_1pct']} transcripts")
    logging.info(f"  Completeness: {stats['complete']} complete, {stats['partial']} partial")
    
    return scores, stats, model_details

