#!/usr/bin/env python
# Standard library imports
import logging
import sys
import os
import subprocess
import re
import pickle
from collections import defaultdict

# Third-party imports
import numpy as np
import pandas as pd



def load_precomputed_junction_map(pickle_file):
    """Load pre-computed junction uniqueness map."""
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    logging.info(f"Loaded junction map with {data['metadata']['total_junctions']:,} junctions")
    logging.info(f"  ({data['metadata']['unique_junctions']:,} unique, "
                f"{data['metadata']['shared_junctions']:,} shared)")
    
    return data

def setup_logging(prefix):
    """Sets up logging to both file and console."""
    log_file = f"{prefix}_pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file at: {log_file}")

def safe_subprocess_run(command, description="", **kwargs):
    """Run subprocess with proper error handling."""
    try:
        # Add executable='/bin/bash' to ensure a compatible shell is used
        result = subprocess.run(command, shell=True, check=True,
                                capture_output=True, text=True,
                                executable='/bin/bash', **kwargs)
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Error {description}: {e}\nSTDERR: {e.stderr}\nSTDOUT: {e.stdout}")
        raise
    
def get_sample_name_from_bam(bam_path):
    """
    A robust function to derive a clean sample name from a BAM file path.

    This function is the single source of truth for sample naming in the MAJEC pipeline.
    It performs the following steps:
    1. Gets the basename of the file (e.g., "my_sample_Aligned.sortedByCoord.out.bam").
    2. Uses a regular expression to remove a chain of common bioinformatics suffixes,
       including aligner outputs, sorting, and file extensions.
    3. Removes a few common prefixes (e.g., "sorted_").

    Args:
        bam_path (str): The full path to the BAM file.

    Returns:
        str: The cleaned, robust sample name.

    Examples:
        >>> get_sample_name_from_bam("/path/to/sample1_Aligned.sortedByCoord.out.bam")
        'sample1'
        >>> get_sample_name_from_bam("other_sample.dedup.sorted.bam")
        'other_sample'
        >>> get_sample_name_from_bam("sorted_weird-name.bam")
        'weird-name'
    """
    if not bam_path:
        return ""

    filename = os.path.basename(bam_path)

    # List of common suffix "tokens" to be removed.
    # The order does not matter here. They are joined into a single regex.
    # This list is easily extensible.
    suffix_tokens = [
        # STAR-specific suffixes
        'Aligned', 'sortedByCoord', 'toTranscriptome', 'out',
        # Common processing suffixes
        'sortedByName', 'sorted', 'dedup', 'mkdup',
        # Paired-end / Single-end indicators
        'pe', 'se',
        # Common file extensions
        'bam', 'sam', 'cram'
    ]

    # Create a regex pattern: (?:[._](?:token1|token2|...))+$
    # This matches one or more occurrences of:
    #   - A separator (dot or underscore)
    #   - Followed by any of the tokens in our list
    #   - Anchored to the end of the string ($)
    #   - The outer (?:...)+ makes it match chains like ".sorted.bam"
    pattern = r'(?:[._](?:' + '|'.join(suffix_tokens) + '))+$'
    
    # Use re.sub to remove the matched pattern. We use IGNORECASE for robustness.
    sample_name = re.sub(pattern, '', filename, flags=re.IGNORECASE)

    # Additionally, handle a few common prefixes
    # This is done separately as they are less common and simpler to handle.
    common_prefixes = [
        'sorted_', 'dedup_'
    ]
    for prefix in common_prefixes:
        if sample_name.lower().startswith(prefix):
            sample_name = sample_name[len(prefix):]
            break # Only strip one prefix

    return sample_name

def calculate_effective_length_distributional(transcript_lengths, frag_mean, frag_sd):
    """
    Calculates effective lengths with scipy truncnorm or numpy fallback.
    """
    if frag_sd is None or frag_sd <= 0:
        return (transcript_lengths - frag_mean + 1).clip(lower=1)
    
    min_len = int(max(1, frag_mean - 4 * frag_sd))
    max_len = int(frag_mean + 4 * frag_sd)
    frag_lengths = np.arange(min_len, max_len + 1)
    
    try:
        # Try scipy truncnorm (most accurate)
        from scipy.stats import truncnorm
        a = (min_len - frag_mean) / frag_sd
        b = (max_len - frag_mean) / frag_sd
        probs = truncnorm.pdf(frag_lengths, a, b, loc=frag_mean, scale=frag_sd)
        
    except ImportError:
        # Numpy fallback - approximate truncnorm
        # Calculate normal PDF
        probs = np.exp(-0.5 * ((frag_lengths - frag_mean) / frag_sd) ** 2)
        
        # For truncated normal, we need to account for the truncation
        # This is approximate but pretty close for ±4 SD range
        logging.warning("scipy not available - using approximate truncated normal", 
                     RuntimeWarning)
    
    # Normalize probabilities
    probs /= probs.sum()
    
    T = transcript_lengths.values[:, np.newaxis]
    possible_starts = np.maximum(T - frag_lengths[np.newaxis, :] + 1, 0)
    eff_lengths = possible_starts @ probs
    
    return pd.Series(eff_lengths, index=transcript_lengths.index)

def create_convergence_groups(pipeline_context,theta, percentile_thresholds=[25, 75]):
    """Create convergence groups using expression thresholds."""

    if pipeline_context['global_thresholds'] is None:
        logging.warning("Global expression thresholds not calculated! Using fallback per-sample binning.")
        # Fallback to old behavior
        if len(theta) == 0:
            return {'low': np.array([]), 'medium': np.array([]), 'high': np.array([])}
        
        low_threshold = np.percentile(theta[theta > 0], percentile_thresholds[0]) if np.sum(theta > 0) > 0 else 1.0
        high_threshold = np.percentile(theta[theta > 0], percentile_thresholds[1]) if np.sum(theta > 0) > 0 else 100.0
    else:
        low_threshold = pipeline_context['global_thresholds']['low_threshold']
        high_threshold = pipeline_context['global_thresholds']['high_threshold']
    
    if len(theta) == 0:
        return {'low': np.array([]), 'medium': np.array([]), 'high': np.array([])}
    
    # Create boolean masks using global thresholds
    low_expr = theta <= low_threshold
    high_expr = theta >= high_threshold
    medium_expr = ~low_expr & ~high_expr
    
    # Convert to index arrays for efficient indexing
    groups = {
        'low': np.where(low_expr)[0],
        'medium': np.where(medium_expr)[0], 
        'high': np.where(high_expr)[0]
    }
    
    return groups


def expectation_step_grouped(grouped_data, current_priors, pseudo_count=1e-6):
    """
    Args:
        grouped_data: Dict mapping feature_tuple -> read_count
        current_priors: Series with current abundance estimates
        pseudo_count: Small value to prevent division by zero
        
    Returns:
        Series with new expected counts (same as original)
    """
    priors_dict = current_priors.to_dict()
    apportioned_dict = defaultdict(float)

  
    for feature_tuple, read_count in grouped_data.items():
        features = list(feature_tuple)
        
        # Add pseudo-count for numerical stability
        weights = [max(priors_dict.get(feat, 0), pseudo_count) for feat in features]
        total_weight = sum(weights)
        
        min_total_weight = pseudo_count * len(features) * 1.1  # Small buffer
        
        if total_weight > min_total_weight:
            # Normal proportional allocation
            for feat, weight in zip(features, weights):
                apportioned_dict[feat] += (weight / total_weight) * read_count
        else:
            # Uniform fallback for zero-weight cases
            if len(features) > 0:
                uniform_allocation = read_count / len(features)
                for feat in features:
                    apportioned_dict[feat] += uniform_allocation
    
    # Convert back to Series (same as original)
    new_counts = pd.Series(0.0, index=current_priors.index, dtype=np.float64)
    
    # Efficient update without intermediate Series creation
    for feat, count in apportioned_dict.items():
        if feat in new_counts.index:
            new_counts[feat] = count 
   
    return new_counts



def standard_em_step(theta, unique_groups, multi_groups, previous_multi_counts):
   
    # E-step for unique mappers using current total counts (same logic)
    unique_counts = expectation_step_grouped(unique_groups, theta)
    
    # E-step for multi-mappers using unique + previous multi as priors (same logic)
    priors_for_mm = unique_counts.add(previous_multi_counts, fill_value=0)
    multi_counts = expectation_step_grouped(multi_groups, priors_for_mm)
    
    # Return new total counts (same as original)
    return unique_counts.add(multi_counts, fill_value=0), unique_counts, multi_counts

# def standard_em_step(theta, unique_groups, multi_groups, previous_multi_counts):
#     logging.info("Performing standard EM step with unified prior...")
#     # Combine with normalized keys (convert everything to frozenset)
#     all_groups = {}
    
#     # Add unique groups (convert tuple keys to frozenset)
#     for key, count in unique_groups.items():
#         normalized_key = frozenset(key) if not isinstance(key, frozenset) else key
#         all_groups[normalized_key] = all_groups.get(normalized_key, 0) + count
    
#     # Add multi groups
#     for key, count in multi_groups.items():
#         normalized_key = frozenset(key) if not isinstance(key, frozenset) else key
#         all_groups[normalized_key] = all_groups.get(normalized_key, 0) + count
    
#     # Single E-step with unified prior
#     new_counts = expectation_step_grouped(all_groups, theta)
    
#     # For decomposition (if needed for logging)
#     unique_counts = expectation_step_grouped(unique_groups, theta)
#     multi_counts = new_counts - unique_counts
    
#     return new_counts, unique_counts, multi_counts


def grouped_momentum_acceleration(pipeline_context, theta_history, unique_groups, multi_groups, previous_multi_counts,
                                         base_momentum=0.3, momentum_scaling_values=[1.5, 1.0, 0.7]):

    momentum_scaling = {'low': momentum_scaling_values[0], 
                        'medium': momentum_scaling_values[1], 
                        'high': momentum_scaling_values[2]}

    if len(theta_history) < 3:
        # Need at least 3 iterates for momentum
        return standard_em_step(theta_history[-1], unique_groups, multi_groups, previous_multi_counts)
    
    try:
        # Calculate velocity 
        current_theta = theta_history[-1]
        velocity = current_theta - theta_history[-2]
        prev_velocity = theta_history[-2] - theta_history[-3]

        # Create expression-based groups 
        groups = create_convergence_groups(pipeline_context, current_theta)
        
        # Apply momentum 
        momentum_factors = np.full_like(velocity, base_momentum)
        
        for group_name, indices in groups.items():
            if len(indices) > 0:
                # Momentum adaptation logic 
                group_velocity = velocity.iloc[indices]
                group_prev_velocity = prev_velocity.iloc[indices]
                
                velocity_correlation = np.corrcoef(group_velocity, group_prev_velocity)[0,1] if len(indices) > 1 else 1.0
                velocity_correlation = np.nan_to_num(velocity_correlation, nan=0.0)
                
                group_change_rate = np.mean(np.abs(group_velocity))
                base_group_momentum = base_momentum * momentum_scaling.get(group_name, 1.0)
                
                if velocity_correlation > 0.5 and group_change_rate > 1e-6:
                    group_momentum = base_group_momentum
                elif velocity_correlation < -0.5:
                    group_momentum = base_group_momentum * 0.3
                else:
                    group_momentum = base_group_momentum * 0.6
                
                group_momentum = min(group_momentum, 0.6)
                momentum_factors[indices] = group_momentum
        
        # Apply momentum 
        theta_momentum = current_theta + momentum_factors * velocity
        theta_momentum = theta_momentum.clip(lower=1e-10)
        
        # Validate momentum step 
        relative_change = np.abs(theta_momentum - current_theta) / (current_theta + 1e-8)
        max_change = np.max(relative_change)
        
        if max_change > 1.0:
            return standard_em_step(current_theta, unique_groups, multi_groups, previous_multi_counts)
        
        # Use EM step for final decomposition
        _, momentum_unique, momentum_multi = standard_em_step(theta_momentum, unique_groups, multi_groups, previous_multi_counts)
        
        return theta_momentum, momentum_unique, momentum_multi
        
    except (ValueError, np.linalg.LinAlgError) as e:
        # Fallback to standard step
        return standard_em_step(theta_history[-1], unique_groups, multi_groups, previous_multi_counts)

def adaptive_convergence_check(current_total, previous_total, iteration, 
                                      min_iterations=15, relative_threshold=0.0001):
   
    if iteration < min_iterations:
        return False
    
    # Overall relative change 
    total_change = abs(current_total.sum() - previous_total.sum()) / (previous_total.sum() + 1e-6)
    
    # Adaptive threshold for meaningful features
    expressed_features = current_total[current_total > 0]
    if len(expressed_features) > 0:
        # Use 1.0 or 10th percentile of expressed features, whichever is higher
        adaptive_threshold = max(1.0, np.percentile(expressed_features, 10))
    else:
        adaptive_threshold = 1.0
    
    meaningful_mask = current_total >= adaptive_threshold
    
    if meaningful_mask.sum() > 0:
        meaningful_current = current_total[meaningful_mask]
        meaningful_previous = previous_total[meaningful_mask]
        
        feature_changes = abs(meaningful_current - meaningful_previous) / (meaningful_previous + 1e-6)
        max_feature_change = feature_changes.max()
        mean_feature_change = feature_changes.mean()
        p95_feature_change = np.percentile(feature_changes, 95)
        
        logging.info(f"     Convergence check (iter {iteration}): "
                    f"Total={total_change:.6f}, Max={max_feature_change:.6f}, "
                    f"Mean={mean_feature_change:.6f}, 95th={p95_feature_change:.6f}")
        logging.info(f"     Meaningful features: {meaningful_mask.sum():,}/{len(current_total):,} "
                    f"(threshold≥{adaptive_threshold:.1f})")
        
        # Convergence criteria
        converged = (
            (total_change < relative_threshold) and
            (max_feature_change < relative_threshold * 5) and  # Less strict than 10
            (mean_feature_change < relative_threshold * 0.5) and  # More strict than 1
            (p95_feature_change < relative_threshold * 2)  # Less strict than 3
        )
        
        if converged:
            logging.info(f"     ✅ CONVERGENCE ACHIEVED: All criteria satisfied")
            logging.info(f"       Final state: {meaningful_mask.sum()} meaningful features, "
                        f"total expression: {current_total.sum():.0f}")
        
        return converged
    
    # Fallback: if no meaningful features, just check total change
    converged = total_change < relative_threshold
    if converged:
        logging.info(f"     ✅ CONVERGENCE ACHIEVED: Total change criteria satisfied")
    
    return converged


