import argparse
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from scipy import stats
from typing import Dict, List, Tuple, Optional
import zipfile
import plotly.io as pio
import io
import json
import subprocess

try:
    import kaleido

    STATIC_EXPORT_AVAILABLE = True
except ImportError:
    STATIC_EXPORT_AVAILABLE = False
    print(
        "Warning: Install kaleido for SVG/PDF export capability. pip install kaleido or conda install python-kaleido"
    )


try:
    from PyPDF2 import PdfMerger

    HAVE_PYPDF2 = True
except ImportError:
    HAVE_PYPDF2 = False
    print("WARNING: PyPDF2 not installed. PDF merging will be disabled.")


# ==============================================================================
# 1. DATA ACCESS & PROCESSING LAYER
# ==============================================================================


def fetch_gene_data_raw(db_conn, gene_id_or_transcript_id, sample_ids):
    """
    Fetches all raw data for a given gene and a list of sample IDs.
    Returns data in long format without pivoting.

    Parameters:
    -----------
    db_conn : sqlite3.Connection
        Database connection
    gene_id_or_transcript_id : str
        Gene ID or transcript ID to fetch
    sample_ids : list of int
        List of sample_id_int values to fetch data for

    Returns:
    --------
    dict with long-format DataFrames for each data type
    """
    print(
        f"Fetching raw data for '{gene_id_or_transcript_id}' across {len(sample_ids)} samples..."
    )

    # --- Step 1: Resolve ID to gene_id ---
    df_check = pd.read_sql_query(
        "SELECT gene_id FROM transcript_annotations WHERE transcript_id_str = ? OR gene_id = ?",
        db_conn,
        params=(gene_id_or_transcript_id, gene_id_or_transcript_id),
    )
    if not df_check.empty:
        gene_id = df_check["gene_id"].iloc[0]
        print(f"  -> Resolved to gene: '{gene_id}'")
    else:
        print(
            f"  ERROR: Identifier '{gene_id_or_transcript_id}' not found.",
            file=sys.stderr,
        )
        return None

    # --- Step 2: Get base annotations ---
    print("  -> Fetching transcript annotations...")
    query_base = "SELECT * FROM transcript_annotations WHERE gene_id = ?"
    df_annotations = pd.read_sql_query(query_base, db_conn, params=(gene_id,))

    if df_annotations.empty:
        print(f"  WARNING: No transcripts found for gene '{gene_id}'.")
        return None

    # Get transcript IDs for filtering
    transcript_ids = tuple(df_annotations["transcript_id_int"].tolist())
    transcript_ids_sql = f"({','.join(map(str, transcript_ids))})"

    # Get sample IDs for filtering
    sample_ids_sql = f"({','.join(map(str, sample_ids))})"

    # --- Step 3: Fetch transcript metrics (long format) ---
    print(f"  -> Fetching transcript metrics for samples: {sample_ids}")
    # starts from 'transcript_annotations' and LEFT JOINs everything else.

    query_transcripts = f"""
        SELECT
            -- Start with the IDs from the annotation table
            ta.transcript_id_int,
            sm.sample_id_int,
            sm.sample_id,

            -- Now get all the quantitative data, which will be NULL if a transcript
            -- is not present in the fct or priors tables for a given sample.
            fct.tpm,
            fct.count,
            p.initial_count,
            p.final_count,
            p.original_subset_penalty,
            p.adjusted_subset_penalty,
            p.completeness_penalty,
            p.tsl_penalty,
            p.n_observed_junctions,
            p.n_expected_junctions,
            p.territory_confidence,
            p.territory_evidence_ratio
        FROM
            -- START with the source of truth: all transcripts for our gene
            (SELECT transcript_id_int FROM transcript_annotations WHERE transcript_id_int IN {transcript_ids_sql}) AS ta
        
        -- Create all combinations of these transcripts with our target samples
        CROSS JOIN
            (SELECT sample_id_int, sample_id FROM sample_metadata WHERE sample_id_int IN {sample_ids_sql}) AS sm

        -- Now, LEFT JOIN the data. This is the key.
        LEFT JOIN
            final_counts_tpm AS fct ON ta.transcript_id_int = fct.transcript_id_int AND sm.sample_id_int = fct.sample_id_int
        LEFT JOIN
            priors AS p ON ta.transcript_id_int = p.transcript_id_int AND sm.sample_id_int = p.sample_id_int;
    """
    df_transcripts_long = pd.read_sql_query(query_transcripts, db_conn)

    # --- Step 4: Fetch junction data (long format) ---
    print("  -> Fetching junction evidence...")

    query_junctions = f"""
        SELECT
            tj.transcript_id_int,
            tj.junction_order,
            ja.junction_id_int,
            ja.junction_id_str,
            ja.is_unique,
            ja.n_transcripts_sharing,
            jc.sample_id_int,
            sm.sample_id,
            COALESCE(jc.read_count, 0) AS read_count
        FROM transcript_junctions AS tj
        JOIN junction_annotations AS ja 
            ON tj.junction_id_int = ja.junction_id_int
        CROSS JOIN (SELECT DISTINCT sample_id_int FROM sample_metadata 
                   WHERE sample_id_int IN {sample_ids_sql}) AS samples
        LEFT JOIN junction_counts AS jc
            ON tj.junction_id_int = jc.junction_id 
            AND jc.sample_id_int = samples.sample_id_int
        LEFT JOIN sample_metadata AS sm
            ON samples.sample_id_int = sm.sample_id_int
        WHERE tj.transcript_id_int IN {transcript_ids_sql}
        ORDER BY tj.transcript_id_int, jc.sample_id_int, tj.junction_order
    """
    df_junctions_long = pd.read_sql_query(query_junctions, db_conn)

    # --- Step 5: Fetch territories (long format) ---
    print("  -> Fetching territory data...")

    query_territories = f"""
        SELECT
            tt.transcript_id_int,
            tt.territory_role,
            tr.chr,
            tr.start,
            tr.end,
            tr.length,
            tc.sample_id_int,
            sm.sample_id,
            COALESCE(tc.mean_coverage, 0) AS mean_coverage
        FROM transcript_territories AS tt
        JOIN territory_regions AS tr 
            ON tt.region_id = tr.region_id
        CROSS JOIN (SELECT DISTINCT sample_id_int FROM sample_metadata 
                   WHERE sample_id_int IN {sample_ids_sql}) AS samples
        LEFT JOIN territory_coverage AS tc
            ON tt.region_id = tc.region_id 
            AND tc.sample_id_int = samples.sample_id_int
        LEFT JOIN sample_metadata AS sm
            ON samples.sample_id_int = sm.sample_id_int
        WHERE tt.transcript_id_int IN {transcript_ids_sql}
    """
    df_territories_long = pd.read_sql_query(query_territories, db_conn)

    # --- Step 6: Fetch normalization factors ---
    print("  -> Fetching size factors...")

    query_size_factors = f"""
        SELECT 
            sn.sample_id_int,
            sm.sample_id,
            sn.size_factor
        FROM sample_normalization AS sn
        JOIN sample_metadata AS sm
            ON sn.sample_id_int = sm.sample_id_int
        WHERE sn.sample_id_int IN {sample_ids_sql}
    """
    df_size_factors = pd.read_sql_query(query_size_factors, db_conn)

    # --- Step 7: Fetch aggregated gene-level data ---
    print("  -> Fetching gene-level aggregated data...")

    query_aggregated = f"""
        SELECT 
            act.sample_id_int,
            sm.sample_id,
            act.count,
            act.tpm
        FROM aggregated_counts_tpm AS act
        JOIN sample_metadata AS sm
            ON act.sample_id_int = sm.sample_id_int
        WHERE act.group_id = ?
          AND act.sample_id_int IN {sample_ids_sql}
    """
    df_aggregated_long = pd.read_sql_query(query_aggregated, db_conn, params=(gene_id,))

    # --- Step 8: Get subset/superset relationships ---
    print("  -> Fetching subset relationships...")

    query_subsets = f"""
        SELECT
            ss.subset_transcript_id_int,
            ss.superset_transcript_id_int,
            subset_ta.transcript_id_str AS subset_id_str,
            superset_ta.transcript_id_str AS superset_id_str
        FROM subset_supersets AS ss
        JOIN transcript_annotations AS subset_ta 
            ON ss.subset_transcript_id_int = subset_ta.transcript_id_int
        JOIN transcript_annotations AS superset_ta 
            ON ss.superset_transcript_id_int = superset_ta.transcript_id_int
        WHERE ss.subset_transcript_id_int IN {transcript_ids_sql}
           OR ss.superset_transcript_id_int IN {transcript_ids_sql}
    """
    df_subset_links = pd.read_sql_query(query_subsets, db_conn)

    print("...Data fetch complete.")

    return {
        "gene_id": gene_id,
        "annotations": df_annotations,
        "transcripts_long": df_transcripts_long,
        "junctions_long": df_junctions_long,
        "territories_long": df_territories_long,
        "size_factors": df_size_factors,
        "aggregated_long": df_aggregated_long,
        "subset_links": df_subset_links,
    }


def fetch_gene_data_raw_with_junction_context(
    db_conn, gene_id_or_transcript_id, sample_ids
):
    """
    Enhanced fetch that includes other gene contexts for junctions in the query gene.
    """
    # Get base data as normal
    raw_data = fetch_gene_data_raw(db_conn, gene_id_or_transcript_id, sample_ids)

    if raw_data is None:
        return None

    # Get unique junction IDs from the query gene
    unique_junction_ids = (
        raw_data["junctions_long"]["junction_id_int"].unique().tolist()
    )

    if not unique_junction_ids:
        return raw_data

    # Find all genes that share these junctions - fixed query
    placeholders = ",".join(["?"] * len(unique_junction_ids))

    # First get the junction-gene mappings
    query = f"""
    WITH JunctionGenes AS (
        SELECT DISTINCT
            ja.junction_id_int,
            ja.junction_id_str,
            ta.gene_id
        FROM transcript_junctions tj
        JOIN transcript_annotations ta ON tj.transcript_id_int = ta.transcript_id_int
        JOIN junction_annotations ja ON tj.junction_id_int = ja.junction_id_int
        WHERE tj.junction_id_int IN ({placeholders})
    ),
    JunctionCounts AS (
        SELECT 
            junction_id_str,
            COUNT(DISTINCT gene_id) as n_genes,
            GROUP_CONCAT(gene_id, ';') as all_genes
        FROM JunctionGenes
        GROUP BY junction_id_str
    )
    SELECT 
        jg.junction_id_str,
        jg.gene_id,
        jc.n_genes,
        jc.all_genes
    FROM JunctionGenes jg
    JOIN JunctionCounts jc ON jg.junction_id_str = jc.junction_id_str
    ORDER BY jg.junction_id_str, jg.gene_id
    """

    junction_gene_map = pd.read_sql_query(query, db_conn, params=unique_junction_ids)

    # Create lookup dictionaries
    junction_to_genes = {}
    junction_to_n_genes = {}
    multi_gene_junctions = set()
    other_genes = set()

    for junc_str, group in junction_gene_map.groupby("junction_id_str"):
        genes = group["gene_id"].unique().tolist()
        junction_to_genes[junc_str] = genes
        junction_to_n_genes[junc_str] = group["n_genes"].iloc[0]

        if len(genes) > 1:
            multi_gene_junctions.add(junc_str)
            other_genes.update([g for g in genes if g != raw_data["gene_id"]])

    # Add to raw_data
    raw_data["junction_gene_map"] = junction_to_genes
    raw_data["multi_gene_junctions"] = multi_gene_junctions
    raw_data["other_genes_with_shared_junctions"] = list(other_genes)

    # Add to junctions_long for easy access in visualization
    raw_data["junctions_long"]["all_genes"] = raw_data["junctions_long"][
        "junction_id_str"
    ].map(lambda x: "; ".join(junction_to_genes.get(x, [])))
    raw_data["junctions_long"]["n_genes"] = raw_data["junctions_long"][
        "junction_id_str"
    ].map(lambda x: junction_to_n_genes.get(x, 1))

    if len(multi_gene_junctions) > 0:
        print(
            f"  -> Found {len(multi_gene_junctions)} junctions shared with other genes: {', '.join(other_genes)}"
        )

    return raw_data


def aggregate_replicate_data(raw_data, group_A_ids, group_B_ids=None):
    """
    Takes the dictionary of raw, long-format data from the fetcher and
    aggregates it by the defined replicate groups (A and B).

    Returns a new dictionary of aggregated, wide-format DataFrames ready for plotting.
    """
    print("--- Aggregating raw data by replicate group ---")

    # --- Normalize the raw counts ect ---
    print("  -> Applying size factor normalization...")

    # Assign group labels ('A' or 'B') to each row
    def assign_group(sample_id):
        if sample_id in group_A_ids:
            return "A"
        if group_B_ids and sample_id in group_B_ids:
            return "B"
        return None

    df_transcripts_long = raw_data["transcripts_long"]
    df_junctions_long = raw_data["junctions_long"]
    df_territories_long = raw_data["territories_long"]
    df_aggregated_long = raw_data["aggregated_long"]
    df_size_factors = raw_data["size_factors"]

    data_to_normalize = {
        "transcripts": (df_transcripts_long, ["count", "initial_count", "final_count"]),
        "junctions": (df_junctions_long, ["read_count"]),
        "territories": (df_territories_long, ["mean_coverage"]),
        "aggregated": (df_aggregated_long, ["count"]),
    }

    for key, (df, cols) in data_to_normalize.items():
        df_merged = pd.merge(df, df_size_factors, on="sample_id_int", how="left")
        df_merged["size_factor"] = df_merged["size_factor"].fillna(1.0)
        df_merged["group"] = df_merged["sample_id_int"].apply(assign_group)
        for col in cols:
            df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce")
            df_merged[f"norm_{col}"] = df_merged[col] / df_merged["size_factor"]
        # Update the original DataFrame in the dictionary
        raw_data[f"{key}_long"] = df_merged

    # Define the aggregation operations for each metric
    # For penalties, we typically just want the mean.
    # For counts/TPM & coverage, mean, std, and N are all useful.
    agg_functions = {
        "transcripts": {
            "tpm": ["mean", "std", "count"],
            "norm_count": ["mean", "std", "count"],
            "norm_initial_count": ["mean", "std", "count"],
            "norm_final_count": ["mean", "std", "count"],
            "original_subset_penalty": ["mean", "std", "count"],
            "adjusted_subset_penalty": ["mean", "std", "count"],
            "completeness_penalty": ["mean", "std", "count"],
            "tsl_penalty": ["mean"],
            "n_observed_junctions": ["mean"],
            "n_expected_junctions": ["mean"],
            "territory_confidence": ["mean", "std", "count"],
            "territory_evidence_ratio": ["mean", "std", "count"],
        },
        "junctions": {"norm_read_count": ["mean", "std", "count"]},
        "territories": {"norm_mean_coverage": ["mean", "std", "count"]},
        "aggregated": {
            "norm_count": ["mean", "std", "count"],
            "tpm": ["mean", "std", "count"],
        },
    }
    agged_dfs = {}
    for key in data_to_normalize:
        print(f"aggregating {key}...")
        df = raw_data[f"{key}_long"]
        for c_key in agg_functions[key].keys():
            df[c_key] = pd.to_numeric(df[c_key], errors="coerce")
        # Perform the groupby and aggregation
        if key == "aggregated":
            df_agg = df.groupby(["group"]).agg(agg_functions[key])
            agged_dfs[f"{key}_agg"] = df_agg
        elif key == "junctions":
            df_agg = df.groupby(["transcript_id_int", "junction_id_str", "group"]).agg(
                agg_functions[key]
            )
            df_unstacked = df_agg.unstack()
            df_unstacked.columns = [
                "_".join(col).strip() for col in df_unstacked.columns.values
            ]
            agged_dfs[f"{key}_agg"] = df_unstacked
        elif key == "territories":
            df_agg = df.groupby(
                ["transcript_id_int", "territory_role", "start", "end", "group"]
            ).agg(agg_functions[key])
            df_unstacked = df_agg.unstack()
            df_unstacked.columns = [
                "_".join(col).strip() for col in df_unstacked.columns.values
            ]
            agged_dfs[f"{key}_agg"] = df_unstacked

        else:
            df_agg = df.groupby(["transcript_id_int", "group"]).agg(agg_functions[key])

            # Unstack to pivot the 'group' level into columns
            df_unstacked = df_agg.unstack()

            # Flatten the resulting MultiIndex columns
            df_unstacked.columns = [
                "_".join(col).strip() for col in df_unstacked.columns.values
            ]
            agged_dfs[f"{key}_agg"] = pd.merge(
                raw_data["annotations"],
                df_unstacked,
                on="transcript_id_int",
                how="left",
            )

    # Add junction-gene map info to aggregated df
    if (
        "junctions_agg" in agged_dfs
        and "all_genes" in raw_data["junctions_long"].columns
    ):
        # Get unique junction metadata from long form
        junction_metadata = raw_data["junctions_long"][
            ["junction_id_str", "all_genes", "n_genes"]
        ].drop_duplicates()
        junction_metadata = junction_metadata.set_index("junction_id_str")

        # Map to the aggregated dataframe using the junction_id_str level of the multi-index
        junction_index_level = agged_dfs["junctions_agg"].index.get_level_values(
            "junction_id_str"
        )

        # Add columns by mapping from the junction metadata
        agged_dfs["junctions_agg"]["all_genes"] = junction_index_level.map(
            junction_metadata["all_genes"].to_dict()
        )
        agged_dfs["junctions_agg"]["n_genes"] = junction_index_level.map(
            junction_metadata["n_genes"].to_dict()
        )

        # Fill any missing values with defaults
        agged_dfs["junctions_agg"]["all_genes"] = agged_dfs["junctions_agg"][
            "all_genes"
        ].fillna("")
        agged_dfs["junctions_agg"]["n_genes"] = agged_dfs["junctions_agg"][
            "n_genes"
        ].fillna(1)

    return agged_dfs, raw_data


def calculate_junction_confidence_intervals(agged_dfs, raw_data, n_boot=1000):
    """
    Calculate confidence intervals for delta PSI after aggregation is done.
    """

    if "norm_read_count_mean_B" not in agged_dfs["junctions_agg"].columns:
        return agged_dfs

    df_junctions_long = raw_data["junctions_long"].copy()

    # Remove rows with NaN sample_id_int
    df_junctions_long = df_junctions_long.dropna(subset=["sample_id_int"])

    # Calculate total junction reads per sample (just for this gene)
    sample_totals = df_junctions_long.groupby("sample_id_int")["norm_read_count"].sum()

    # Add PSI for each junction-sample combination using a merge instead of apply
    df_totals = pd.DataFrame(sample_totals).reset_index()
    df_totals.columns = ["sample_id_int", "total_reads"]

    df_junctions_long = df_junctions_long.merge(
        df_totals, on="sample_id_int", how="left"
    )
    df_junctions_long["psi"] = df_junctions_long["norm_read_count"] / (
        df_junctions_long["total_reads"] + 1e-6
    )

    ci_results = []

    for tid, junc_id in agged_dfs["junctions_agg"].index:
        # Get PSI values for each replicate
        psi_A = df_junctions_long[
            (df_junctions_long["transcript_id_int"] == tid)
            & (df_junctions_long["junction_id_str"] == junc_id)
            & (df_junctions_long["group"] == "A")
        ]["psi"].values

        psi_B = df_junctions_long[
            (df_junctions_long["transcript_id_int"] == tid)
            & (df_junctions_long["junction_id_str"] == junc_id)
            & (df_junctions_long["group"] == "B")
        ]["psi"].values

        # Calculate delta PSI and confidence intervals
        if len(psi_A) >= 2 and len(psi_B) >= 2:
            mean_delta = np.log2((np.mean(psi_B) + 0.001) / (np.mean(psi_A) + 0.001))

            # Bootstrap CI (more robust than parametric)
            n_boot = n_boot
            delta_psis = []
            for _ in range(n_boot):
                boot_A = np.random.choice(psi_A, size=len(psi_A), replace=True)
                boot_B = np.random.choice(psi_B, size=len(psi_B), replace=True)
                delta = np.log2((np.mean(boot_B) + 0.001) / (np.mean(boot_A) + 0.001))
                delta_psis.append(delta)

            ci_lower = np.percentile(delta_psis, 2.5)
            ci_upper = np.percentile(delta_psis, 97.5)
        else:
            mean_delta = (
                np.log2((np.mean(psi_B) + 0.001) / (np.mean(psi_A) + 0.001))
                if len(psi_B) > 0
                else 0
            )
            ci_lower = np.nan
            ci_upper = np.nan

        ci_results.append(
            {
                "transcript_id_int": tid,
                "junction_id_str": junc_id,
                "delta_psi_lfc": mean_delta,
                "delta_psi_ci_lower": ci_lower,
                "delta_psi_ci_upper": ci_upper,
                "is_significant": not np.isnan(ci_lower)
                and (ci_lower > 0 or ci_upper < 0),
            }
        )

    df_ci = pd.DataFrame(ci_results).set_index(["transcript_id_int", "junction_id_str"])
    agged_dfs["junctions_agg"] = agged_dfs["junctions_agg"].join(df_ci)

    return agged_dfs


def calculate_junction_statistics(agged_dfs, raw_data):
    """
    Perform statistical tests on junction usage between conditions.
    """

    df_junctions_long = raw_data["junctions_long"]

    # Get junction counts per sample per group
    junctions_by_group = df_junctions_long.groupby(
        ["transcript_id_int", "junction_id_str", "group"]
    )

    results = []

    for (tid, junc_id, group), group_data in junctions_by_group:
        if group == "A":
            counts_A = group_data["norm_read_count"].values
        elif group == "B":
            counts_B = group_data["norm_read_count"].values

    # For each junction, run appropriate test
    for tid, junc_id in (
        df_junctions_long[["transcript_id_int", "junction_id_str"]]
        .drop_duplicates()
        .values
    ):
        counts_A = df_junctions_long[
            (df_junctions_long["transcript_id_int"] == tid)
            & (df_junctions_long["junction_id_str"] == junc_id)
            & (df_junctions_long["group"] == "A")
        ]["norm_read_count"].values
        counts_B = df_junctions_long[
            (df_junctions_long["transcript_id_int"] == tid)
            & (df_junctions_long["junction_id_str"] == junc_id)
            & (df_junctions_long["group"] == "B")
        ]["norm_read_count"].values

        # Option 1: T-test on PSI values (if sufficient replicates)
        if len(counts_A) >= 3 and len(counts_B) >= 3:
            # Calculate PSI for each replicate
            total_A = (
                df_junctions_long[(df_junctions_long["group"] == "A")]
                .groupby("sample_id_int")["norm_read_count"]
                .sum()
            )
            total_B = (
                df_junctions_long[(df_junctions_long["group"] == "B")]
                .groupby("sample_id_int")["norm_read_count"]
                .sum()
            )

            psi_A = counts_A / (total_A.values + 1e-6)
            psi_B = counts_B / (total_B.values + 1e-6)

            # Welch's t-test (doesn't assume equal variance)
            t_stat, p_value = stats.ttest_ind(psi_A, psi_B, equal_var=False)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(psi_A) + np.var(psi_B)) / 2)
            cohens_d = (np.mean(psi_B) - np.mean(psi_A)) / (pooled_std + 1e-6)

        # Option 2: Mann-Whitney U for small sample sizes
        elif len(counts_A) >= 2 and len(counts_B) >= 2:
            u_stat, p_value = stats.mannwhitneyu(
                counts_A, counts_B, alternative="two-sided"
            )
            t_stat = u_stat
            cohens_d = np.nan

        # Option 3: Fisher's exact test for presence/absence if very few replicates
        else:
            present_A = np.sum(counts_A > 0)
            absent_A = len(counts_A) - present_A
            present_B = np.sum(counts_B > 0)
            absent_B = len(counts_B) - present_B

            odds_ratio, p_value = stats.fisher_exact(
                [[present_A, absent_A], [present_B, absent_B]]
            )
            t_stat = odds_ratio
            cohens_d = np.nan

        # Multiple testing correction will be needed later
        results.append(
            {
                "transcript_id_int": tid,
                "junction_id_str": junc_id,
                "mean_A": np.mean(counts_A),
                "mean_B": np.mean(counts_B),
                "p_value": p_value,
                "test_statistic": t_stat,
                "effect_size": cohens_d,
                "n_reps_A": len(counts_A),
                "n_reps_B": len(counts_B),
            }
        )

    df_stats = pd.DataFrame(results)

    # Multiple testing correction (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests

    if len(df_stats) > 0:
        _, df_stats["p_adjusted"], _, _ = multipletests(
            df_stats["p_value"], method="fdr_bh", alpha=0.05
        )

    return df_stats


def calculate_expression_statistics(agged_dfs, norm_data):
    """
    Calculate expression statistics from replicate data.
    """

    df_transcripts_long = norm_data["transcripts_long"]

    expr_stats = []

    for tid in agged_dfs["transcripts_agg"]["transcript_id_int"].unique():
        # Get normalized counts for each replicate
        tid_data = df_transcripts_long[df_transcripts_long["transcript_id_int"] == tid]

        counts_A = tid_data[tid_data["group"] == "A"]["norm_count"].dropna().values
        counts_B = tid_data[tid_data["group"] == "B"]["norm_count"].dropna().values

        # Calculate statistics on counts
        if len(counts_A) > 0 and len(counts_B) > 0:
            mean_A = np.mean(counts_A)
            mean_B = np.mean(counts_B)

            # Log2 fold change
            expr_lfc = np.log2((mean_B + 1) / (mean_A + 1))

            # Statistical test
            if len(counts_A) >= 2 and len(counts_B) >= 2:
                _, p_value = stats.ttest_ind(counts_A, counts_B, equal_var=False)
            else:
                p_value = 1.0
        else:
            expr_lfc = 0
            p_value = 1.0

        expr_stats.append(
            {
                "transcript_id_int": tid,
                "expr_lfc": expr_lfc,
                "expr_pvalue": p_value,
                "expr_significant": p_value < 0.05 and abs(expr_lfc) > 1,
            }
        )

    # Add to aggregated data
    df_expr_stats = pd.DataFrame(expr_stats)
    agged_dfs["transcripts_agg"] = agged_dfs["transcripts_agg"].merge(
        df_expr_stats, on="transcript_id_int", how="left"
    )

    # --- Gene-level statistics ---
    df_aggregated_long = norm_data["aggregated_long"]
    if "aggregated_agg" in agged_dfs and not df_aggregated_long.empty:
        # Get gene-level counts for each replicate
        gene_counts_A = (
            df_aggregated_long[df_aggregated_long["group"] == "A"]["norm_count"]
            .dropna()
            .values
        )
        gene_counts_B = (
            df_aggregated_long[df_aggregated_long["group"] == "B"]["norm_count"]
            .dropna()
            .values
        )

        if len(gene_counts_A) > 0 and len(gene_counts_B) > 0:
            gene_mean_A = np.mean(gene_counts_A)
            gene_mean_B = np.mean(gene_counts_B)

            # Log2 fold change
            gene_expr_lfc = np.log2((gene_mean_B + 1) / (gene_mean_A + 1))

            # Statistical test
            if len(gene_counts_A) >= 2 and len(gene_counts_B) >= 2:
                _, gene_p_value = stats.ttest_ind(
                    gene_counts_A, gene_counts_B, equal_var=False
                )
            else:
                gene_p_value = 1.0
        else:
            gene_expr_lfc = 0
            gene_p_value = 1.0

        # Add gene-level stats to aggregated_agg dataframe
        agged_dfs["gene_stats"] = {
            "expr_lfc": gene_expr_lfc,
            "expr_pvalue": gene_p_value,
            "expr_significant": gene_p_value < 0.05 and abs(gene_expr_lfc) > 1,
        }

    return agged_dfs


# ==============================================================================
# 2. VISUALIZATION LAYER
# ==============================================================================


def create_junction_arc_plot_aggregated(agged_dfs, raw_data, gene_id, track_spacing=3):
    """
    Creates junction arc plot using aggregated replicate data with confidence intervals.
    """

    def parse_junction(junc_key):
        parts = junc_key.split(":")
        chrom = parts[0]
        coords = parts[1].split("-")
        start = int(coords[0])
        end = int(coords[1])
        strand = parts[2] if len(parts) > 2 else "+"
        return chrom, start, end, strand

    print("--- Creating Junction Arc Plot with Aggregated Data ---")

    # Extract aggregated dataframes
    df_transcripts = agged_dfs["transcripts_agg"]
    df_junctions = agged_dfs["junctions_agg"].reset_index()

    # Check for sample B data
    has_sample_B = "norm_read_count_mean_B" in df_junctions.columns

    # Sort transcripts by mean TPM
    sort_col = (
        "tpm_mean_A" if "tpm_mean_A" in df_transcripts.columns else "norm_count_mean_A"
    )
    df_transcripts_sorted = df_transcripts.sort_values(
        by=sort_col, ascending=True, na_position="first"
    )

    # Get coordinate mapping
    coords = set()
    for j_str in df_junctions["junction_id_str"].unique():
        _, start, end, _ = parse_junction(j_str)
        coords.add(start)
        coords.add(end)

    sorted_coords = sorted(list(coords))
    coord_map = {coord: i for i, coord in enumerate(sorted_coords)}

    fig = go.Figure()

    # Find max values for scaling
    max_tpm = (
        pd.to_numeric(df_transcripts["tpm_mean_A"], errors="coerce").max()
        if "tpm_mean_A" in df_transcripts.columns
        else 1.0
    )
    max_read_count = (
        df_junctions["norm_read_count_mean_A"].max()
        if "norm_read_count_mean_A" in df_junctions.columns
        else 1.0
    )
    if has_sample_B:
        max_read_count = max(
            max_read_count, df_junctions["norm_read_count_mean_B"].max()
        )

    # Hover collectors
    arc_hover_x = []
    arc_hover_y = []
    arc_hover_text = []
    y_tick_vals = []
    y_tick_text = []
    y_position = 0

    all_shapes = []
    # Draw each transcript
    for _, tx_row in df_transcripts_sorted.iterrows():
        y_position += track_spacing
        tid_int = tx_row["transcript_id_int"]
        tid_str = tx_row["transcript_id_str"]

        # Get transcript junctions
        tx_junctions = df_junctions[df_junctions["transcript_id_int"] == tid_int]
        if tx_junctions.empty:
            continue

        y_tick_vals.append(y_position)
        y_tick_text.append(tid_str)

        # Use mean TPM for line styling
        tpm_mean_A = pd.to_numeric(tx_row.get("tpm_mean_A", 0), errors="coerce")
        if pd.isna(tpm_mean_A):
            tpm_mean_A = 0.0

        line_width = 2 + 4 * (tpm_mean_A / max_tpm)
        color_intensity = 0.4 + 0.6 * (tpm_mean_A / max_tpm)
        line_color = f"rgba(112, 128, 144, {color_intensity})"

        # Enhanced hover text with stats
        hover_text = f"<b>{tid_str}</b><br>"
        hover_text += f"TPM A: {tx_row.get('tpm_mean_A', 0):.2f}"
        if "tpm_std_A" in tx_row:
            hover_text += f" ± {tx_row.get('tpm_std_A', 0):.2f}"
        if "tpm_count_A" in tx_row:
            hover_text += f" (n={tx_row.get('tpm_count_A', 0):.0f})"
        hover_text += f"<br>Count A: {tx_row.get('norm_count_mean_A', 0):.0f}"

        if has_sample_B:
            hover_text += f"<br>TPM B: {tx_row.get('tpm_mean_B', 0):.2f}"
            if "tpm_std_B" in tx_row:
                hover_text += f" ± {tx_row.get('tpm_std_B', 0):.2f}"
            if "tpm_count_B" in tx_row:
                hover_text += f" (n={tx_row.get('tpm_count_B', 0):.0f})"
            hover_text += f"<br>Count B: {tx_row.get('norm_count_mean_B', 0):.0f}"

        # Draw transcript line
        tx_coords = set()
        for j_str in tx_junctions["junction_id_str"]:
            _, start, end, _ = parse_junction(j_str)
            tx_coords.add(start)
            tx_coords.add(end)

        fig.add_trace(
            go.Scatter(
                x=[coord_map[c] for c in sorted(tx_coords)],
                y=[y_position] * len(tx_coords),
                mode="lines+markers",
                line=dict(color=line_color, width=line_width),
                marker=dict(symbol="line-ns-open", size=12, color=line_color),
                name=tid_str,
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False,
            )
        )

        # Draw junction arcs
        for _, junc_row in tx_junctions.iterrows():
            j_str = junc_row["junction_id_str"]
            _, start, end, _ = parse_junction(j_str)
            x0, x1 = coord_map[start], coord_map[end]

            # Sample A arc
            read_mean_A = junc_row.get("norm_read_count_mean_A", 0)
            read_std_A = junc_row.get("norm_read_count_std_A", 0)
            n_reps_A = junc_row.get("norm_read_count_count_A", 0)

            base_arc_height = 0.2
            arc_height_A = base_arc_height + 1.5 * (
                np.log1p(read_mean_A) / np.log1p(max_read_count)
            )
            color_intensity_A = 0.3 + 0.7 * (read_mean_A / max_read_count)
            arc_color_A = f"rgba(178, 34, 34, {color_intensity_A})"
            line_width_A = 1.5 + 4 * (read_mean_A / max_read_count)

            fill_color = None
            if "n_genes" in junc_row and junc_row["n_genes"] > 1:
                fill_color = "rgba(255, 0, 255, 0.1)"

            # Junction hover text
            hover_text = f"<b>Junction:</b> {j_str}<br>"
            hover_text += f"Count A: {read_mean_A:.1f}"
            if n_reps_A > 0:
                hover_text += f" ± {read_std_A:.1f} (n={n_reps_A:.0f})"

            if has_sample_B:
                read_mean_B = junc_row.get("norm_read_count_mean_B", 0)
                read_std_B = junc_row.get("norm_read_count_std_B", 0)
                n_reps_B = junc_row.get("norm_read_count_count_B", 0)
                hover_text += f"<br>Count B: {read_mean_B:.1f}"
                if n_reps_B > 0:
                    hover_text += f" ± {read_std_B:.1f} (n={n_reps_B:.0f})"

                if "delta_psi_lfc" in junc_row:
                    hover_text += (
                        f"<br><b>ΔPSI (log2):</b> {junc_row['delta_psi_lfc']:.2f}"
                    )
                    if "delta_psi_ci_lower" in junc_row and not pd.isna(
                        junc_row["delta_psi_ci_lower"]
                    ):
                        hover_text += f" (95% CI: {junc_row['delta_psi_ci_lower']:.2f}, {junc_row['delta_psi_ci_upper']:.2f})"
                    if "is_significant" in junc_row:
                        hover_text += f"<br>Significant: {'Yes' if junc_row['is_significant'] else 'No'}"

                if read_mean_A > 0:
                    arc_path_A = f"M {x0} {y_position} C {x0+0.5*(x1-x0)} {y_position+arc_height_A}, {x1-0.5*(x1-x0)} {y_position+arc_height_A}, {x1} {y_position}"
                    all_shapes.append(
                        {
                            "type": "path",
                            "path": arc_path_A,
                            "fillcolor": fill_color,
                            "line": {"color": arc_color_A, "width": line_width_A},
                            "opacity": 0.8,
                        }
                    )
                else:
                    arc_path_A = f"M {x0} {y_position} C {x0+0.5*(x1-x0)} {y_position+base_arc_height*3}, {x1-0.5*(x1-x0)} {y_position+base_arc_height*3}, {x1} {y_position}"
                    all_shapes.append(
                        {
                            "type": "path",
                            "path": arc_path_A,
                            "line": {"color": "#CCCCCC", "width": 1, "dash": "dot"},
                        }
                    )
                
                if "n_genes" in junc_row and junc_row["n_genes"] > 1:
                    other_genes = junc_row.get("all_genes", "").replace(";", ", ")
                    hover_text += f"<br><b>⚠ Multi-gene junction ({junc_row['n_genes']} genes)</b>"
                    hover_text += f"<br>Genes: {other_genes}"

                arc_hover_x.append(x0 + 0.5 * (x1 - x0))
                arc_hover_y.append(y_position + 0.5 * arc_height_A)
                arc_hover_text.append(hover_text)

                # Sample B arc
                if has_sample_B:
                    arc_height_B = base_arc_height + 1.5 * (
                        np.log1p(read_mean_B) / np.log1p(max_read_count)
                    )
                    line_width_B = 1.5 + 4 * (read_mean_B / max_read_count)
                    color_intensity_B = 0.3 + 0.7 * (read_mean_B / max_read_count)
                    arc_color_B = f"rgba(0, 0, 205, {color_intensity_B})"

                    if read_mean_B > 0:
                        arc_path_B = f"M {x0} {y_position} C {x0+0.5*(x1-x0)} {y_position-arc_height_B}, {x1-0.5*(x1-x0)} {y_position-arc_height_B}, {x1} {y_position}"
                        all_shapes.append(
                            {
                                "type": "path",
                                "fillcolor": fill_color,
                                "path": arc_path_B,
                                "line": {"color": arc_color_B, "width": line_width_B},
                                "opacity": 0.8,
                            }
                        )
                    else:
                        arc_path_B = f"M {x0} {y_position} C {x0+0.5*(x1-x0)} {y_position-base_arc_height*3}, {x1-0.5*(x1-x0)} {y_position-base_arc_height*3}, {x1} {y_position}"
                        all_shapes.append(
                            {
                                "type": "path",
                                "path": arc_path_B,
                                "line": {"color": "#CCCCCC", "width": 1, "dash": "dot"},
                            }
                        )

                    # Delta PSI bar with confidence interval visualization
                    if (
                        "delta_psi_lfc" in junc_row
                        and abs(junc_row["delta_psi_lfc"]) > 0.1
                    ):
                        bar_x = x0 + 0.5 * (x1 - x0)
                        bar_height = abs(junc_row["delta_psi_lfc"]) * 0.3
                        bar_color = (
                            "#FF6B35" if junc_row["delta_psi_lfc"] > 0 else "#00A6ED"
                        )
                        ci_fill_color = (
                            f"rgba(255, 107, 53, 0.2)"
                            if junc_row["delta_psi_lfc"] > 0
                            else f"rgba(0, 166, 237, 0.2)"
                        )

                        # 2. Draw the Confidence Interval as an "Error Bar" on top
                        if "delta_psi_ci_lower" in junc_row and not pd.isna(
                            junc_row["delta_psi_ci_lower"]
                        ):
                            ci_lower = junc_row["delta_psi_ci_lower"]
                            ci_upper = junc_row["delta_psi_ci_upper"]

                            # Calculate the y-coordinates for the start and end of the error bar line
                            y0_ci_line = y_position + ci_lower * 0.3
                            y1_ci_line = y_position + ci_upper * 0.3

                            # Draw the main vertical line of the error bar
                            all_shapes.append(
                                {
                                    "type": "line",
                                    "x0": bar_x,
                                    "x1": bar_x,
                                    "y0": y0_ci_line,
                                    "y1": y1_ci_line,
                                    "line": {"color": "black", "width": 1},
                                }
                            )

                            # Draw the bottom cap of the error bar
                            all_shapes.append(
                                {
                                    "type": "line",
                                    "x0": bar_x - 0.05,
                                    "x1": bar_x + 0.05,
                                    "y0": y0_ci_line,
                                    "y1": y0_ci_line,
                                    "line": {"color": "black", "width": 1},
                                }
                            )

                            # Draw the top cap of the error bar
                            all_shapes.append(
                                {
                                    "type": "line",
                                    "x0": bar_x - 0.05,
                                    "x1": bar_x + 0.05,
                                    "y0": y1_ci_line,
                                    "y1": y1_ci_line,
                                    "line": {"color": "black", "width": 1},
                                }
                            )

                        # Mean bar (solid)
                        opacity = 0.8 if junc_row.get("is_significant", False) else 0.4
                        all_shapes.append(
                            {
                                "type": "rect",
                                "x0": bar_x - 0.15,
                                "x1": bar_x + 0.15,
                                "y0": y_position - bar_height,
                                "y1": y_position + bar_height,
                                "fillcolor": bar_color,
                                "opacity": opacity,
                                "line": {"width": 0},
                            }
                        )

                    # Add invisible hover layer
                    fig.add_trace(
                        go.Scatter(
                            x=arc_hover_x,
                            y=arc_hover_y,
                            mode="markers",
                            marker=dict(color="rgba(0,0,0,0)", size=15),
                            hoverinfo="text",
                            hovertext=arc_hover_text,
                            showlegend=False,
                        )
                    )

    fig.update_layout(shapes=all_shapes)

    # Create legend elements using dummy traces
    legend_traces = []

    # Junction arc legend items
    legend_traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="rgba(178, 34, 34, 0.8)"),
            name="Sample A junctions (above)",
            showlegend=True,
        )
    )

    legend_traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="rgba(0, 0, 205, 0.8)"),
            name="Sample B junctions (below)",
            showlegend=True,
        )
    )

    legend_traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#CCCCCC", width=2, dash="dot"),
            name="Missing junctions",
            showlegend=True,
        )
    )

    # Delta PSI bars
    legend_traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="#FF6B35", symbol="square"),
            name="↑ B > A usage (ΔPSI > 0)",
            showlegend=True,
        )
    )

    legend_traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="#004E89", symbol="square"),
            name="↓ A > B usage (ΔPSI < 0)",
            showlegend=True,
        )
    )

    # Transcript line intensity
    legend_traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="rgba(112, 128, 144, 0.3)", width=2),
            name="Low TPM transcript",
            showlegend=True,
        )
    )

    legend_traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="rgba(112, 128, 144, 1.0)", width=6),
            name="High TPM transcript",
            showlegend=True,
        )
    )

    # Add all legend traces to figure
    for trace in legend_traces:
        fig.add_trace(trace)

    # Update layout to position legend
    fig.update_layout(
        legend=dict(
            title=dict(text="<b>Visual Elements</b>", font=dict(size=12)),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=10),
        ),
        showlegend=True,  # Changed from False to True
        margin=dict(r=200),  # Add right margin for legend
    )
    # --- 4. Final Layout and Axis Configuration ---
    fig.update_layout(
        title_text=f"Junction Arc Analysis for {gene_id}",
        xaxis=dict(
            tickmode="array",
            tickvals=list(coord_map.values()),
            ticktext=[f"{c/1e6:.2f}M" for c in sorted_coords],
            title="Genomic Position",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_tick_vals,
            ticktext=y_tick_text,
            title="Transcript",
        ),
        showlegend=False,
        height=max(400, len(df_transcripts_sorted) * 75),
    )

    fig.update_layout(
        title_text=f"Junction Arc Analysis for {gene_id}",
        # --- NEW: Cleaner, lighter theme ---
        plot_bgcolor="white",  # Set background to white
        paper_bgcolor="white",
        xaxis=dict(
            tickmode="array",
            tickvals=list(coord_map.values()),
            ticktext=[f"{c/1e6:.2f}M" for c in sorted_coords],
            title="Genomic Position",
            gridcolor="lightgrey",  # Make grid lines lighter
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_tick_vals,
            ticktext=y_tick_text,
            gridcolor="lightgrey",  # Make grid lines lighter
        ),
        showlegend=True,
        height=max(400, len(df_transcripts_sorted) * 75),
    )

    return fig


def create_differential_junction_heatmap(agged_dfs, gene_id):
    """
    Create a heatmap showing DIFFERENTIAL junction usage with expression context.
    """

    df_transcripts = agged_dfs["transcripts_agg"]
    df_junctions = agged_dfs["junctions_agg"].reset_index()

    # Get all unique junctions and sort them
    all_junctions = df_junctions["junction_id_str"].unique()

    def parse_junction(junc_key):
        parts = junc_key.split(":")
        chrom = parts[0]
        coords = parts[1].split("-")
        start = int(coords[0])
        end = int(coords[1])
        strand = parts[2] if len(parts) > 2 else "+"
        return chrom, start, end, strand

    sorted_junctions = sorted(all_junctions, key=lambda j: parse_junction(j)[1:3])

    if sorted_junctions and parse_junction(sorted_junctions[0])[3] == "-":
        sorted_junctions = sorted_junctions[::-1]

    # Build junction level data
    heatmap_data = []
    significance_data = []
    transcript_info = []
    counts_data_A = []
    counts_data_B = []
    jgene_info_data = []
    n_jgenes_data = []

    for _, transcript_row in df_transcripts.iterrows():
        tid_int = transcript_row["transcript_id_int"]
        tid_str = transcript_row["transcript_id_str"]
        transcript_junctions = df_junctions[
            df_junctions["transcript_id_int"] == tid_int
        ]

        row_values = []
        row_sig = []
        row_counts_A = []
        row_counts_B = []
        row_jgene_info = []
        row_n_jgenes = []
        for junc in sorted_junctions:
            junc_data = transcript_junctions[
                transcript_junctions["junction_id_str"] == junc
            ]

            if not junc_data.empty:
                if "delta_psi_lfc" in junc_data.columns:
                    delta_psi = junc_data["delta_psi_lfc"].iloc[0]
                    is_sig = (
                        junc_data["is_significant"].iloc[0]
                        if "is_significant" in junc_data.columns
                        else False
                    )
                    # Get the mean counts
                    mean_count_A = (
                        junc_data["norm_read_count_mean_A"].iloc[0]
                        if "norm_read_count_mean_A" in junc_data.columns
                        else 0
                    )
                    mean_count_B = (
                        junc_data["norm_read_count_mean_B"].iloc[0]
                        if "norm_read_count_mean_B" in junc_data.columns
                        else 0
                    )
                    jgene_info = junc_data["all_genes"].iloc[0]
                    n_jgenes = junc_data["n_genes"].iloc[0]
                else:
                    delta_psi = 0
                    is_sig = False
                    mean_count_A = 0
                    mean_count_B = 0
                    jgene_info = ""
                    n_jgenes = 1

                row_values.append(delta_psi)
                row_sig.append(is_sig)
                row_counts_A.append(mean_count_A)  # Store these
                row_counts_B.append(mean_count_B)  # Store these
                row_jgene_info.append(jgene_info)
                row_n_jgenes.append(int(n_jgenes))
            else:
                row_values.append(np.nan)
                row_sig.append(False)
                row_counts_A.append(0)
                row_counts_B.append(0)
                row_jgene_info.append("")
                row_n_jgenes.append(1)

        heatmap_data.append(row_values)
        significance_data.append(row_sig)
        counts_data_A.append(row_counts_A)
        counts_data_B.append(row_counts_B)
        jgene_info_data.append(row_jgene_info)
        n_jgenes_data.append(row_n_jgenes)

        # Get expression values
        tpm_A = transcript_row.get("tpm_mean_A", 0)
        tpm_B = transcript_row.get("tpm_mean_B", 0)

        transcript_info.append(
            {
                "tid_int": tid_int,
                "tid_str": tid_str,
                "tpm_A": tpm_A,
                "tpm_B": tpm_B,
                "mean_tpm": (tpm_A + tpm_B) / 2,
                "expr_lfc": transcript_row.get(
                    "expr_lfc", 0
                ),  # Make sure this matches the column name
                "expr_pvalue": transcript_row.get("expr_pvalue", 1),
                "expr_significant": transcript_row.get("expr_significant", False),
            }
        )
    # Sort by mean expression
    sorted_indices = sorted(
        range(len(transcript_info)), key=lambda i: -transcript_info[i]["mean_tpm"]
    )

    transcript_info_sorted = [transcript_info[i] for i in sorted_indices]
    heatmap_array = np.array([heatmap_data[i] for i in sorted_indices])
    significance_array = np.array([significance_data[i] for i in sorted_indices])
    counts_array_A = np.array([counts_data_A[i] for i in sorted_indices])
    counts_array_B = np.array([counts_data_B[i] for i in sorted_indices])
    jgene_info_array = np.array([jgene_info_data[i] for i in sorted_indices])
    n_jgenes_array = np.array([n_jgenes_data[i] for i in sorted_indices])

    # Add gene leve data for context
    gene_stats = agged_dfs["aggregated_agg"]

    # Create gene total info
    gene_total_info = {
        "tid": "Gene_Total",
        "tpm_A": gene_stats.loc["A", ("tpm", "mean")],
        "tpm_B": gene_stats.loc["B", ("tpm", "mean")],
        "expr_lfc": agged_dfs["gene_stats"]["expr_lfc"],
        "expr_pvalue": agged_dfs["gene_stats"]["expr_pvalue"],
        "expr_significant": agged_dfs["gene_stats"]["expr_significant"],
    }

    j_number = len(sorted_junctions)
    # Add NA values for junctions in gene total row
    gene_heatmap_row = [np.nan] * j_number
    gene_significance_row = [False] * j_number
    gene_counts_row_A = [np.nan] * j_number
    gene_counts_row_B = [np.nan] * j_number
    jgene_info_row = [""] * j_number
    n_jgene_row = [np.nan] * j_number

    # Append gene row to sorted data
    transcript_info_sorted.append(gene_total_info)
    heatmap_array = np.vstack([heatmap_array, gene_heatmap_row])
    significance_array = np.vstack([significance_array, gene_significance_row])
    counts_array_A = np.vstack([counts_array_A, gene_counts_row_A])
    counts_array_B = np.vstack([counts_array_B, gene_counts_row_B])
    jgene_info_array = np.vstack([jgene_info_array, jgene_info_row])
    n_jgenes_array = np.vstack([n_jgenes_array, n_jgene_row])

    # Create subplots with expression bars
    fig = make_subplots(
        rows=1,
        cols=4,
        column_widths=[0.066, 0.066, 0.067, 0.8],
        subplot_titles=[
            "TPM A",
            "TPM B",
            "LFC",
            f" Differential Junction Usage: {gene_id}",
        ],
        horizontal_spacing=0.02,
    )

    # Add expression bars for condition A
    tpm_A_values = [t["tpm_A"] for t in transcript_info_sorted]
    fig.add_trace(
        go.Heatmap(
            z=[[val] for val in tpm_A_values],
            colorscale="Greens",
            showscale=False,
            text=[[f"{val:.1f}"] for val in tpm_A_values],
            texttemplate="%{text}",
            hovertemplate="TPM A: %{z:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add expression bars for condition B
    tpm_B_values = [t["tpm_B"] for t in transcript_info_sorted]
    fig.add_trace(
        go.Heatmap(
            z=[[val] for val in tpm_B_values],
            colorscale="Purples",
            showscale=False,
            text=[[f"{val:.1f}"] for val in tpm_B_values],
            texttemplate="%{text}",
            hovertemplate="TPM B: %{z:.1f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    expr_text = []
    for t in transcript_info_sorted:
        val = t.get("expr_lfc", 0)
        sig = "*" if t.get("expr_significant", False) else ""
        expr_text.append([f"{val:.1f}{sig}"])

    expr_hover = []
    for t in transcript_info_sorted:
        val = t.get("expr_lfc", 0)
        pval = t.get("expr_pvalue", 1)
        sig = t.get("expr_significant", False)
        hover_text = f"Expr LFC: {val:.2f}<br>p-value: {pval:.3e}"
        if sig:
            hover_text += "<br><b>Significant</b>"
        expr_hover.append(hover_text)

    fig.add_trace(
        go.Heatmap(
            z=[[t.get("expr_lfc", 0)] for t in transcript_info_sorted],
            colorscale="PiYG",
            zmid=0,
            zmin=-3,
            zmax=3,
            showscale=False,
            text=expr_text,
            texttemplate="%{text}",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=[[h] for h in expr_hover],
        ),
        row=1,
        col=3,
    )

    # Add main differential heatmap
    customdata_formatted = []
    for i in range(heatmap_array.shape[0]):
        row_custom = []
        for j in range(heatmap_array.shape[1]):
            hover_text = f"A:{counts_array_A[i,j]:.1f} B:{counts_array_B[i,j]:.1f}"
            # Add multi-gene info if applicable
            if n_jgenes_array[i, j] > 1:
                hover_text += (
                    f"<br><b>⚠ Multi-gene junction ({n_jgenes_array[i,j]} genes)</b>"
                )
                hover_text += f"<br>Genes: {jgene_info_array[i,j]}"
            row_custom.append(hover_text)
        customdata_formatted.append(row_custom)

    annotations = []
    for i in range(heatmap_array.shape[0]):
        for j in range(heatmap_array.shape[1]):
            val = heatmap_array[i, j]
            if np.isnan(val):
                continue

            sig = significance_array[i, j]
            is_multi = n_jgenes_array[i, j] > 1

            # Determine text and style
            text = f"{val:.1f}"
            font_dict = {}

            if is_multi:
                font_dict["color"] = "#FF00FF"
            elif sig:
                font_dict["color"] = "#13294A"
            else:
                font_dict["color"] = "#818181"

            if sig:
                text = f"<b>{text}</b>*"  # Bold for significant

            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=text,
                    xref="x4",
                    yref="y",
                    showarrow=False,
                    font=font_dict,
                )
            )

    heatmap_trace = go.Heatmap(
        z=heatmap_array,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-3,
        zmax=3,
        colorbar=dict(title="ΔPSI (log2)<br>B vs A"),
        customdata=customdata_formatted,
        hovertemplate="ΔPSI: %{z:.2f}<br>%{customdata}<extra></extra>",
    )

    fig.add_trace(heatmap_trace, row=1, col=4)
    fig.update_layout(
        annotations=(
            fig.layout.annotations + tuple(annotations)
            if fig.layout.annotations
            else annotations
        )
    )
    # Update y-axis with transcript names
    y_labels = [t["tid_str"] for t in transcript_info_sorted[:-1]] + [
        "<b>Gene Total</b>"
    ]
    for col in [1, 2, 3]:
        fig.update_yaxes(
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels if col == 1 else [""] * len(y_labels),
            row=1,
            col=col,
        )

    # Update x-axis for junction heatmap
    fig.update_xaxes(
        tickvals=list(range(len(sorted_junctions))),
        ticktext=[f"J{i}" for i in range(len(sorted_junctions))],
        title="Junction Position (5' → 3')",
        row=1,
        col=4,
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=3)

    fig.update_layout(
        title=f"{gene_id} - Sorted by mean expression",
        plot_bgcolor="white",
        height=max(400, len(transcript_info_sorted) * 30),
        width=1200,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray", gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", gridwidth=0.5)

    return fig


def select_and_format_single_sample(raw_data, target_sample_id_int):
    """
    Takes the raw, long-format data dictionary, selects the data for a
    single sample, and pivots it into the wide format expected by the
    plotting function.
    """
    # --- 1. Prepare the Transcript DataFrame ---

    # Filter the long-format transcript data for our one sample
    df_transcripts_long_sample = raw_data["transcripts_long"][
        raw_data["transcripts_long"]["sample_id_int"] == target_sample_id_int
    ].copy()

    # Merge with the base annotations to get the full transcript list
    # The result will have data for the target sample and NaNs for non-expressed transcripts
    df_transcripts_wide = pd.merge(
        raw_data["annotations"],
        df_transcripts_long_sample,
        on="transcript_id_int",
        how="left",
    )

    # --- 2. Prepare the Junction DataFrame ---

    # Filter the long-format junction data for our one sample
    df_junctions_long_sample = raw_data["junctions_long"][
        raw_data["junctions_long"]["sample_id_int"] == target_sample_id_int
    ].copy()

    # The junction data is already in a good format, we just need to rename the column
    df_junctions_wide = df_junctions_long_sample

    # Assemble the final dictionary in the exact format the plotting function needs
    formatted_data = {
        "gene_id": raw_data["gene_id"],
        "transcripts": df_transcripts_wide,
        "junctions": df_junctions_wide,
    }

    return formatted_data


def create_junction_penalty_heatmap_plotly(
    gene_data, sample_name="A", output_prefix=None
):
    """
    Create interactive Plotly heatmap showing junction patterns for all transcripts in a single gene.

    Parameters:
    -----------
    gene_data : dict
        Dictionary returned from fetch_gene_data containing:
        - 'gene_id': str
        - 'transcripts': DataFrame with transcript information
        - 'junctions': DataFrame with junction evidence
        - 'subset_links': DataFrame with subset/superset relationships
        - 'territories': DataFrame with territory data
        - 'aggregated': DataFrame with gene-level aggregated data
    sample_name : str
        to indicate which sample being visualized
    output_prefix : str, optional
        Prefix for output files (if saving static versions)
    """

    gene_id = gene_data["gene_id"]
    df_transcripts = gene_data["transcripts"]
    df_junctions = gene_data["junctions"]

    # Step 1: Collect ALL unique junctions across all transcripts
    all_junctions = df_junctions["junction_id_str"].unique()

    # Sort junctions by genomic position
    def parse_junction(junc_key):
        """Parse junction string format: chr:start-end:strand"""
        parts = junc_key.split(":")
        chrom = parts[0]
        coords = parts[1].split("-")
        start = int(coords[0])
        end = int(coords[1])
        strand = parts[2] if len(parts) > 2 else "+"
        return chrom, start, end, strand

    sorted_junctions = sorted(all_junctions, key=lambda j: parse_junction(j)[1:3])

    # Handle strand - reverse if negative strand gene
    if sorted_junctions and parse_junction(sorted_junctions[0])[3] == "-":
        sorted_junctions = sorted_junctions[::-1]

    # Step 2: Build heatmap data matrix
    heatmap_data = []
    transcript_info = []
    hover_texts = []

    for _, transcript_row in df_transcripts.sort_values("tpm").iterrows():
        tid_int = transcript_row["transcript_id_int"]
        tid_str = transcript_row["transcript_id_str"]

        # Get junction data for this transcript
        transcript_junctions = df_junctions[
            df_junctions["transcript_id_int"] == tid_int
        ]
        expected_junctions = set(transcript_junctions["junction_id_str"])

        # Create dictionary of junction read counts
        junction_counts = {}
        read_count_col = "read_count"
        if read_count_col in transcript_junctions.columns:
            junction_counts = dict(
                zip(
                    transcript_junctions["junction_id_str"],
                    transcript_junctions[read_count_col],
                )
            )

        # Build row for heatmap
        row_values = []
        row_hover = []

        for junc in sorted_junctions:
            if junc in expected_junctions:
                count = junction_counts.get(junc, 0)
                row_values.append(count)
                row_hover.append(
                    f"Junction: {junc}<br>Count: {count}<br>Transcript: {tid_str}"
                )
            else:
                row_values.append(np.nan)
                row_hover.append(f"Junction: {junc}<br>Not in transcript: {tid_str}")

        heatmap_data.append(row_values)
        hover_texts.append(row_hover)

        # Store transcript info for sorting and annotation
        transcript_info.append(
            {
                "tid_int": tid_int,
                "tid_str": tid_str,
                "tsl": (
                    int(tsl)
                    if (tsl := transcript_row.get("tsl")) and str(tsl).isdigit()
                    else np.nan
                ),
                "is_subset": transcript_row.get("is_subset", False),
                "completeness_penalty": transcript_row.get("completeness_penalty", 1.0),
                "tsl_penalty": transcript_row.get("tsl_penalty", 1.0),
                "adjusted_subset_penalty": transcript_row.get(
                    "adjusted_subset_penalty", 1.0
                ),
                "initial_count": transcript_row.get("initial_count", 0),
                "final_count": transcript_row.get("count", 0),
                "tpm": transcript_row.get("tpm", 0),
                "n_observed": transcript_row.get("n_observed_junctions", 0),
                "n_expected": transcript_row.get("n_expected_junctions", 0),
            }
        )

    # Sort by completeness penalty (best to worst)
    sorted_indices = sorted(
        range(len(transcript_info)), key=lambda i: (transcript_info[i]["tpm"])
    )

    heatmap_array = np.array([heatmap_data[i] for i in sorted_indices])
    hover_array = [hover_texts[i] for i in sorted_indices]
    transcript_info_sorted = [transcript_info[i] for i in sorted_indices]

    # Apply log transformation for visualization
    log_heatmap = np.where(~np.isnan(heatmap_array), np.log2(heatmap_array + 1), np.nan)

    # Create subplots with annotation columns and count columns
    fig = make_subplots(
        rows=1,
        cols=6,
        column_widths=[0.06, 0.06, 0.06, 0.06, 0.06, 0.70],
        subplot_titles=[
            "TSL",
            "Sub",
            "Comp",
            "Init",
            "Final",
            f"Junction Coverage: {gene_id}",
        ],
        horizontal_spacing=0.02,
    )

    # Prepare y-axis labels
    y_labels = []
    for info in transcript_info_sorted:
        count_str = f"{info['final_count']:.0f}" if info["final_count"] > 0 else "0"
        tpm_str = f"{info['tpm']:.1f}" if info["tpm"] > 0 else "0"
        label = f"{info['tid_str']} (Count:{count_str}, TPM:{tpm_str})"
        y_labels.append(label)

    # 1. TSL Penalty Column
    tsl_penalties = [[info["tsl_penalty"]] for info in transcript_info_sorted]
    tsl_text = [[info["tsl"]] for info in transcript_info_sorted]

    fig.add_trace(
        go.Heatmap(
            z=tsl_penalties,
            text=tsl_text,
            texttemplate="%{text}",
            colorscale="RdYlGn",  # Reversed: low penalty (green) is good
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate="TSL: %{text}<br>Penalty: %{z:.5f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 2. Subset Penalty Column
    subset_penalties = [
        [info["adjusted_subset_penalty"]] for info in transcript_info_sorted
    ]
    subset_text = [
        ["Y" if info["is_subset"] else "N"] for info in transcript_info_sorted
    ]

    fig.add_trace(
        go.Heatmap(
            z=subset_penalties,
            text=subset_text,
            texttemplate="%{text}",
            colorscale="RdYlGn",  # Reversed: low penalty (green) is good
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate="Subset: %{text}<br>Penalty: %{z:.5f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # 3. Completeness Penalty Column
    completeness_penalties = [
        [info["completeness_penalty"]] for info in transcript_info_sorted
    ]
    completeness_text = [
        [f"{info['n_observed']}/{info['n_expected']}"]
        for info in transcript_info_sorted
    ]

    fig.add_trace(
        go.Heatmap(
            z=completeness_penalties,
            text=completeness_text,
            texttemplate="%{text}",
            colorscale="RdYlGn",  # Reversed: low penalty (green) is good
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate="Junctions: %{text}<br>Penalty: %{z:.5f}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    # 4. Initial Count Column (new)
    initial_counts = [[info["initial_count"]] for info in transcript_info_sorted]
    # Use log scale for counts to better show differences
    initial_counts_log = [[np.log10(c[0] + 1)] for c in initial_counts]
    initial_text = [[f"{int(c[0])}" if c[0] > 0 else "0"] for c in initial_counts]

    fig.add_trace(
        go.Heatmap(
            z=initial_counts_log,
            text=initial_text,
            texttemplate="%{text}",
            colorscale="Purples",
            showscale=False,
            hovertemplate="Initial Count: %{text}<extra></extra>",
        ),
        row=1,
        col=4,
    )

    # 5. Final Count Column (new)
    final_counts = [[info["final_count"]] for info in transcript_info_sorted]
    # Use log scale for counts to better show differences
    final_counts_log = [[np.log10(c[0] + 1)] for c in final_counts]
    final_text = [[f"{int(c[0])}" if c[0] > 0 else "0"] for c in final_counts]

    fig.add_trace(
        go.Heatmap(
            z=final_counts_log,
            text=final_text,
            texttemplate="%{text}",
            colorscale="Oranges",
            showscale=False,
            hovertemplate="Final Count: %{text}<extra></extra>",
        ),
        row=1,
        col=5,
    )

    # 6. Main Junction Heatmap
    # Create custom colorscale with white for zero, gray for NA
    colorscale = [
        [0, "white"],  # Zero counts
        [0.001, "#deebf7"],  # Very light blue for small counts
        [0.2, "#9ecae1"],
        [0.4, "#4292c6"],
        [0.6, "#2171b5"],
        [0.8, "#08519c"],
        [1.0, "#08306b"],  # Dark blue for high counts
    ]

    # Create text annotations for the heatmap
    annotations_text = []
    for i, row in enumerate(heatmap_array):
        row_text = []
        for val in row:
            if np.isnan(val):
                row_text.append("")
            elif val == 0:
                row_text.append("0")
            else:
                row_text.append(f"{int(val)}")
        annotations_text.append(row_text)

    fig.add_trace(
        go.Heatmap(
            z=log_heatmap,
            text=annotations_text,
            texttemplate="%{text}",
            customdata=heatmap_array,
            hovertext=hover_array,
            hovertemplate="%{hovertext}<br>Raw count: %{customdata}<br>Log2(count+1): %{z:.2f}<extra></extra>",
            colorscale=colorscale,
            zmin=0,
            zmax=np.nanmax(log_heatmap) if np.any(~np.isnan(log_heatmap)) else 3,
            colorbar=dict(title="log2(Count+1)", x=1.02, xpad=10),
        ),
        row=1,
        col=6,
    )

    # Update layout
    fig.update_layout(
        title=f"Gene Junction Analysis: {gene_id} - Sample {sample_name}",
        height=max(400, len(transcript_info_sorted) * 30),
        width=1400,
        showlegend=False,
        hovermode="closest",
    )

    # Update y-axes
    for col in [1, 2, 3, 4, 5]:
        fig.update_yaxes(
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels if col == 1 else [""] * len(y_labels),
            row=1,
            col=col,
        )

    fig.update_yaxes(
        tickvals=list(range(len(y_labels))), ticktext=[""] * len(y_labels), row=1, col=6
    )

    # Update x-axes for main heatmap
    if len(sorted_junctions) <= 30:
        x_ticktext = [f"J{i}" for i in range(len(sorted_junctions))]
    else:
        # Show subset of labels for many junctions
        step = len(sorted_junctions) // 20
        x_ticktext = [
            f"J{i}" if i % step == 0 else "" for i in range(len(sorted_junctions))
        ]

    fig.update_xaxes(
        tickvals=list(range(len(sorted_junctions))),
        ticktext=x_ticktext,
        title="Junction Position (5' → 3')",
        row=1,
        col=6,
    )

    # Hide x-axis for annotation columns
    for col in [1, 2, 3, 4, 5]:
        fig.update_xaxes(showticklabels=False, row=1, col=col)

    # Add shapes to highlight missing junctions (optional)
    shapes = []
    for i, row in enumerate(heatmap_array):
        for j, val in enumerate(row):
            if not np.isnan(val) and val == 0:
                shapes.append(
                    dict(
                        type="rect",
                        xref=f"x6",
                        yref=f"y6",
                        x0=j - 0.45,
                        x1=j + 0.45,
                        y0=i - 0.45,
                        y1=i + 0.45,
                        line=dict(color="red", width=2),
                        fillcolor="rgba(0,0,0,0)",
                    )
                )

    fig.update_layout(shapes=shapes)
    fig.update_yaxes(showline=False, zeroline=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(showline=False, zeroline=False, showgrid=False, row=1, col=2)
    fig.update_yaxes(showline=False, zeroline=False, showgrid=False, row=1, col=3)
    fig.update_yaxes(showline=False, zeroline=False, showgrid=False, row=1, col=4)
    fig.update_yaxes(showline=False, zeroline=False, showgrid=False, row=1, col=5)
    fig.update_xaxes(showline=False, zeroline=False, showgrid=False, row=1, col=1)
    fig.update_xaxes(showline=False, zeroline=False, showgrid=False, row=1, col=2)
    fig.update_xaxes(showline=False, zeroline=False, showgrid=False, row=1, col=3)
    fig.update_xaxes(showline=False, zeroline=False, showgrid=False, row=1, col=4)
    fig.update_xaxes(showline=False, zeroline=False, showgrid=False, row=1, col=5)

    return fig


# Helper function to draw transcript with bilateral junctions
def draw_transcript_bilateral(
    fig,
    all_junctions,
    transcript_id,
    transcript_info,
    subset_tid,
    row,
    junction_color_a="rgba(150, 50, 50, 0.8)",
    junction_color_b="rgba(50, 50, 150, 0.8)",
):
    """
    Shared function to draw bilateral junction arcs for any transcript.

    Parameters:
    -----------
    fig : plotly figure
        The figure to add traces to
    all_junctions : DataFrame
        DataFrame containing all junction data
    transcript_id : int
        ID of transcript to draw
    transcript_info : Series
        Row from transcripts dataframe with metadata
    subset_tid : int
        ID of the subset transcript (for highlighting)
    row : int
        Row number in subplot
    junction_color_a, junction_color_b : str
        Colors for condition A and B junctions

    Returns:
    --------
    None (modifies fig in place)
    """

    # Parse junction coordinates helper
    def parse_junction_coords(junction_str):
        parts = junction_str.split(":")
        coords = parts[1].split("-")
        return int(coords[0]), int(coords[1])

    tx_junctions = all_junctions[
        all_junctions.index.get_level_values("transcript_id_int") == transcript_id
    ]

    if tx_junctions.empty:
        return

    # Get junction boundaries for baseline
    junction_coords = set()
    for idx in tx_junctions.index:
        junc_str = idx[1] if isinstance(idx, tuple) else idx
        start, end = parse_junction_coords(junc_str)
        junction_coords.add(start)
        junction_coords.add(end)

    sorted_junc_coords = sorted(junction_coords)

    # Draw baseline
    is_subset = transcript_id == subset_tid
    baseline_color = "black" if is_subset else "gray"
    baseline_width = 2 if is_subset else 1

    fig.add_trace(
        go.Scatter(
            x=[sorted_junc_coords[0], sorted_junc_coords[-1]],
            y=[0, 0],
            mode="lines",
            line=dict(color=baseline_color, width=baseline_width),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )

    # Add rich hover information on transcript baseline
    hover_text = f'<b>{"SUBSET" if is_subset else "SUPERSET"}: {transcript_info["transcript_id_str"]}</b><br>'
    hover_text += f'TPM A: {transcript_info.get("tpm_mean_A", 0):.1f}±{transcript_info.get("tpm_std_A", 0):.1f}<br>'
    hover_text += f'TPM B: {transcript_info.get("tpm_mean_B", 0):.1f}±{transcript_info.get("tpm_std_B", 0):.1f}<br>'
    hover_text += f'Count A: {transcript_info.get("norm_count_mean_A", 0):.0f}±{transcript_info.get("norm_count_std_A", 0):.0f}<br>'
    hover_text += f'Count B: {transcript_info.get("norm_count_mean_B", 0):.0f}±{transcript_info.get("norm_count_std_B", 0):.0f}'

    if is_subset:
        hover_text += f'<br>Penalty A: {transcript_info.get("adjusted_subset_penalty_mean_A", 1):.4f}'
        hover_text += f'<br>Penalty B: {transcript_info.get("adjusted_subset_penalty_mean_B", 1):.4f}'

    fig.add_trace(
        go.Scatter(
            x=sorted_junc_coords,
            y=[0] * len(sorted_junc_coords),
            mode="markers",
            marker=dict(
                size=8,
                color=baseline_color,
                symbol="line-ns",
                line=dict(width=2, color=baseline_color),
            ),
            showlegend=False,
            hovertext=[hover_text] * len(sorted_junc_coords),
            hoverinfo="text",
            hoverlabel=dict(bgcolor="white", font_size=11),
        ),
        row=row,
        col=1,
    )

    # Draw bilateral junction arcs
    max_arc_height = 0.4
    max_read_count_a = (
        all_junctions["norm_read_count_mean_A"].max()
        if "norm_read_count_mean_A" in all_junctions.columns
        else 1
    )
    max_read_count_b = (
        all_junctions["norm_read_count_mean_B"].max()
        if "norm_read_count_mean_B" in all_junctions.columns
        else 1
    )

    for idx, junc in tx_junctions.iterrows():
        junc_str = idx[1] if isinstance(idx, tuple) else idx
        start, end = parse_junction_coords(junc_str)

        # Enhanced junction hover text
        junction_hover = f"<b>Junction: {junc_str}</b><br>"
        junction_hover += f'Transcript: {transcript_info["transcript_id_str"]}<br>'

        # Sample A arc (above)
        mean_a = junc.get("norm_read_count_mean_A", 0)
        std_a = junc.get("norm_read_count_std_A", 0)
        if not pd.isna(mean_a):
            junction_hover += f"Sample A: {mean_a:.1f}±{std_a:.1f} reads<br>"
            if mean_a > 0:
                arc_height = 0.1 + max_arc_height * (mean_a / max_read_count_a)
                n_points = 30
                x_points = np.linspace(start, end, n_points)
                y_points = arc_height * np.sin(
                    np.pi * (x_points - start) / (end - start)
                )

                fig.add_trace(
                    go.Scatter(
                        x=x_points,
                        y=y_points,
                        mode="lines",
                        line=dict(
                            color=junction_color_a, width=max(1, min(3, mean_a / 10))
                        ),
                        hovertext=junction_hover,
                        hoverinfo="text",
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
            else:
                x_points = np.linspace(start, end, 20)
                y_points = 0.1 * np.sin(np.pi * (x_points - start) / (end - start))

                fig.add_trace(
                    go.Scatter(
                        x=x_points,
                        y=y_points,
                        mode="lines",
                        line=dict(color="lightgray", width=1, dash="dot"),
                        hovertext=junction_hover + "Missing in A",
                        hoverinfo="text",
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )

        # Sample B arc (below)
        mean_b = junc.get("norm_read_count_mean_B", 0)
        std_b = junc.get("norm_read_count_std_B", 0)
        if not pd.isna(mean_b):
            junction_hover += f"Sample B: {mean_b:.1f}±{std_b:.1f} reads"
            if mean_b > 0:
                arc_height = 0.1 + max_arc_height * (mean_b / max_read_count_b)
                n_points = 30
                x_points = np.linspace(start, end, n_points)
                y_points = -arc_height * np.sin(
                    np.pi * (x_points - start) / (end - start)
                )

                fig.add_trace(
                    go.Scatter(
                        x=x_points,
                        y=y_points,
                        mode="lines",
                        line=dict(
                            color=junction_color_b, width=max(1, min(3, mean_b / 10))
                        ),
                        hovertext=junction_hover,
                        hoverinfo="text",
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
            else:
                x_points = np.linspace(start, end, 20)
                y_points = -0.1 * np.sin(np.pi * (x_points - start) / (end - start))

                fig.add_trace(
                    go.Scatter(
                        x=x_points,
                        y=y_points,
                        mode="lines",
                        line=dict(color="lightgray", width=1, dash="dot"),
                        hovertext=junction_hover + "<br>Missing in B",
                        hoverinfo="text",
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )


def visualize_subset_focused_territories(
    aggregated_data: Dict,
    norm_data: Dict,
    subset_tid: int,
    show_individual_samples: bool = True,
):
    """
    Create a comprehensive visualization showing a subset transcript with ALL its superset partners.
    Features bilateral junction arcs, territory coverage, and metrics panel.

    Parameters:
    -----------
    aggregated_data : dict
        Dictionary from aggregate_replicate_data with aggregated statistics
    norm_data : dict
        Dictionary from fetch_gene_data_raw after size factor count normalization (long-format data)
    subset_tid : int
        Transcript ID (int) of the subset transcript
    show_individual_samples : bool
        Whether to show individual sample distributions
    """

    # Extract relevant data
    df_transcripts = aggregated_data["transcripts_agg"]
    df_junctions = aggregated_data["junctions_agg"]
    df_territories = aggregated_data["territories_agg"]
    df_subset_links = norm_data["subset_links"]

    # Get subset info
    subset_info = df_transcripts[
        df_transcripts["transcript_id_int"] == subset_tid
    ].iloc[0]

    # Check if this subset has significant penalties
    penalty_a = subset_info.get("adjusted_subset_penalty_mean_A", 1.0)
    penalty_b = subset_info.get("adjusted_subset_penalty_mean_B", 1.0)

    if (
        pd.isna(penalty_a)
        or pd.isna(penalty_b)
        or (penalty_a > 0.99 and penalty_b > 0.99)
    ):
        print(f"Skipping {subset_info['transcript_id_str']} - no significant penalties")
        return None

    # Check if this subset has territory data
    subset_territories = df_territories[
        (df_territories.index.get_level_values("transcript_id_int") == subset_tid)
        & (df_territories.index.get_level_values("territory_role") == "unique")
    ]

    if subset_territories.empty:
        print(f"Skipping {subset_info['transcript_id_str']} - no territory data")
        return None

    # Get all superset partners for this subset
    superset_ids = df_subset_links[
        df_subset_links["subset_transcript_id_int"] == subset_tid
    ]["superset_transcript_id_int"].tolist()

    if len(superset_ids) == 0:
        print(f"No supersets found for subset {subset_info['transcript_id_str']}")
        return None

    # Get superset info
    superset_infos = df_transcripts[
        df_transcripts["transcript_id_int"].isin(superset_ids)
    ].sort_values("tpm_mean_A", ascending=False)

    # Get all relevant junctions and territories
    all_transcript_ids = [subset_tid] + superset_ids
    all_junctions = df_junctions[
        df_junctions.index.get_level_values("transcript_id_int").isin(
            all_transcript_ids
        )
    ].copy()

    # Parse junction coordinates
    def parse_junction_coords(junction_str):
        parts = junction_str.split(":")
        coords = parts[1].split("-")
        return int(coords[0]), int(coords[1])

    # Get genomic range
    all_coords = []
    if not all_junctions.empty:
        for junc_str in all_junctions.index.get_level_values(
            "junction_id_str"
        ).unique():
            start, end = parse_junction_coords(junc_str)
            all_coords.extend([start, end])

    # Get territory coordinates
    subset_territories = df_territories[
        (df_territories.index.get_level_values("transcript_id_int") == subset_tid)
        & (df_territories.index.get_level_values("territory_role") == "unique")
    ].copy()

    comparator_territories = df_territories[
        (df_territories.index.get_level_values("transcript_id_int") == subset_tid)
        & (df_territories.index.get_level_values("territory_role") == "comparator")
    ].copy()

    if not subset_territories.empty:
        all_coords.extend(
            subset_territories.index.get_level_values("start").tolist()
            + subset_territories.index.get_level_values("end").tolist()
        )
    if not comparator_territories.empty:
        all_coords.extend(
            comparator_territories.index.get_level_values("start").tolist()
            + comparator_territories.index.get_level_values("end").tolist()
        )

    if not all_coords:
        print("No coordinate data found!")
        return None

    min_coord = min(all_coords) - 500
    max_coord = max(all_coords) + 500

    # Calculate number of transcript rows needed
    n_transcripts = 1 + len(superset_ids)  # subset + all supersets

    # Create figure with dynamic row allocation
    # Adjust heights to reduce excessive spacing
    transcript_height = min(
        0.08, 0.4 / n_transcripts
    )  # Cap individual transcript height
    territory_height = 0.4  # Fixed height for territory panel

    row_heights = []
    row_specs = []
    subplot_titles = []

    # Add rows for each transcript (subset + supersets)
    for i in range(n_transcripts):
        row_heights.append(transcript_height)
        if i == 0:
            # First row: spans all transcript rows plus territory row for the metrics panel
            row_specs.append(
                [
                    {"colspan": 2},
                    None,
                    {"rowspan": n_transcripts + 1, "secondary_y": True},
                ]
            )
            subplot_titles.extend(["", "", ""])
        else:
            # Subsequent rows: the third column is occupied by the rowspan from row 1
            row_specs.append([{"colspan": 2}, None, None])
            superset_info = superset_infos.iloc[i - 1]
            subplot_titles.extend(["", None, None])

    # Add territory coverage row
    row_heights.append(territory_height)
    row_specs.append(
        [{"colspan": 2}, None, None]
    )  # Third column occupied by metrics panel
    subplot_titles.extend(["", None, None])

    fig = make_subplots(
        rows=n_transcripts + 1,
        cols=3,
        column_widths=[0.35, 0.35, 0.30],
        row_heights=row_heights,
        specs=row_specs,
        vertical_spacing=0.01,
        horizontal_spacing=0.07,
        subplot_titles=subplot_titles,
    )

    # Color schemes
    # Color schemes (Revised Territory Palette)
    territory_unique_color_a = "#66C2A5"  # Soft, vibrant teal green
    territory_unique_color_b = "#1B9E77"  # Darker, richer teal green
    territory_comp_color_a = "#B3B3B3"  # Light, neutral grey
    territory_comp_color_b = "#666666"  # Darker, solid grey
    junction_color_a = "rgba(150, 50, 50, 0.8)"
    junction_color_b = "rgba(50, 50, 150, 0.8)"

    # Collect all shapes for bulk updating (performance optimization)
    all_shapes = []

    # Draw subset transcript
    draw_transcript_bilateral(
        fig, all_junctions, subset_tid, subset_info, subset_tid, 1
    )

    # Draw each superset
    for i, (_, superset_info) in enumerate(superset_infos.iterrows(), start=2):
        draw_transcript_bilateral(
            fig,
            all_junctions,
            superset_info["transcript_id_int"],
            superset_info,
            subset_tid,
            i,
        )

    # Apply all shapes at once for better performance
    fig.update_layout(shapes=all_shapes)

    # Territory coverage row (bottom panel)
    territory_row = n_transcripts + 1
    max_coverage = 0

    # Plot unique territories (above zero)
    if not subset_territories.empty:
        for idx, terr in subset_territories.iterrows():
            start = idx[2] if isinstance(idx, tuple) else 0
            end = idx[3] if isinstance(idx, tuple) else 0

            cov_a = terr.get("norm_mean_coverage_mean_A", 0)
            cov_b = terr.get("norm_mean_coverage_mean_B", 0)
            std_a = terr.get("norm_mean_coverage_std_A", 0)
            std_b = terr.get("norm_mean_coverage_std_B", 0)

            if not pd.isna(cov_a):
                max_coverage = max(max_coverage, cov_a + std_a)
                log_cov = np.log2(cov_a + 1)

                fig.add_trace(
                    go.Scatter(
                        x=[start, end, end, start, start],
                        y=[0, 0, log_cov, log_cov, 0],
                        fill="toself",
                        fillcolor=territory_unique_color_a,
                        line=dict(color=territory_unique_color_a, width=1),
                        hovertext=f"Unique Territory A<br>Region: {start}-{end}<br>Coverage: {cov_a:.1f}±{std_a:.1f}",
                        hoverinfo="text",
                        showlegend=False,
                        opacity=0.7,
                    ),
                    row=territory_row,
                    col=1,
                )

            if not pd.isna(cov_b):
                max_coverage = max(max_coverage, cov_b + std_b)
                log_cov = np.log2(cov_b + 1)
                offset = (end - start) * 0.05

                fig.add_trace(
                    go.Scatter(
                        x=[
                            start + offset,
                            end - offset,
                            end - offset,
                            start + offset,
                            start + offset,
                        ],
                        y=[0, 0, log_cov, log_cov, 0],
                        fill="toself",
                        fillcolor=territory_unique_color_b,
                        line=dict(color=territory_unique_color_b, width=1, dash="dot"),
                        hovertext=f"Unique Territory B<br>Region: {start}-{end}<br>Coverage: {cov_b:.1f}±{std_b:.1f}",
                        hoverinfo="text",
                        showlegend=False,
                        opacity=0.5,
                    ),
                    row=territory_row,
                    col=1,
                )

    # Plot comparator territories (below zero)
    if not comparator_territories.empty:
        for idx, terr in comparator_territories.iterrows():
            start = idx[2] if isinstance(idx, tuple) else 0
            end = idx[3] if isinstance(idx, tuple) else 0

            cov_a = terr.get("norm_mean_coverage_mean_A", 0)
            cov_b = terr.get("norm_mean_coverage_mean_B", 0)
            std_a = terr.get("norm_mean_coverage_std_A", 0)
            std_b = terr.get("norm_mean_coverage_std_B", 0)

            if not pd.isna(cov_a):
                max_coverage = max(max_coverage, cov_a + std_a)
                log_cov = np.log2(cov_a + 1)

                fig.add_trace(
                    go.Scatter(
                        x=[start, end, end, start, start],
                        y=[0, 0, -log_cov, -log_cov, 0],
                        fill="toself",
                        fillcolor=territory_comp_color_a,
                        line=dict(color=territory_comp_color_a, width=1),
                        hovertext=f"Comparator Territory A<br>Region: {start}-{end}<br>Coverage: {cov_a:.1f}±{std_a:.1f}",
                        hoverinfo="text",
                        showlegend=False,
                        opacity=0.7,
                    ),
                    row=territory_row,
                    col=1,
                )

            if not pd.isna(cov_b):
                max_coverage = max(max_coverage, cov_b + std_b)
                log_cov = np.log2(cov_b + 1)
                offset = (end - start) * 0.05

                fig.add_trace(
                    go.Scatter(
                        x=[
                            start + offset,
                            end - offset,
                            end - offset,
                            start + offset,
                            start + offset,
                        ],
                        y=[0, 0, -log_cov, -log_cov, 0],
                        fill="toself",
                        fillcolor=territory_comp_color_b,
                        line=dict(color=territory_comp_color_b, width=1, dash="dot"),
                        hovertext=f"Comparator Territory B<br>Region: {start}-{end}<br>Coverage: {cov_b:.1f}±{std_b:.1f}",
                        hoverinfo="text",
                        showlegend=False,
                        opacity=0.5,
                    ),
                    row=territory_row,
                    col=1,
                )

    # RIGHT PANEL: Metrics with grouped violin plots
    raw_transcripts = pd.DataFrame()
    if "transcripts_long" in norm_data:
        raw_transcripts = norm_data["transcripts_long"][
            norm_data["transcripts_long"]["transcript_id_int"] == subset_tid
        ]

    metrics = [
        ("adjusted_subset_penalty", "Penalty", True, False),
        ("territory_evidence_ratio", "Evidence", True, False),
        ("tpm", "TPM", False, True),
    ]

    x_positions = [1, 2.5, 4]

    for i, (metric, label, use_log, secondary_y) in enumerate(metrics):
        x_base = x_positions[i]
        plot_created = False

        if not raw_transcripts.empty and f"{metric}" in raw_transcripts.columns:
            for condition, color, x_offset in [
                ("A", junction_color_a, -0.2),
                ("B", junction_color_b, 0.2),
            ]:
                condition_data = raw_transcripts[raw_transcripts["group"] == condition]
                if not condition_data.empty and metric in condition_data.columns:
                    sample_values = condition_data[metric].dropna()

                    if len(sample_values) > 0:
                        display_values = (
                            -np.log10(sample_values + 1e-10)
                            if use_log
                            else sample_values
                        )

                        fig.add_trace(
                            go.Violin(
                                x=[x_base + x_offset] * len(display_values),
                                y=display_values,
                                width=0.3,
                                marker_color=color,
                                opacity=0.6,
                                name=f"{label} {condition}",
                                box_visible=True,
                                meanline_visible=True,
                                points="all",
                                pointpos=-0.8,
                                jitter=0.05,
                                orientation="v",
                                hovertext=[
                                    f"{label} {condition}<br>Sample: {s}<br>Value: {v:.4f}"
                                    + (
                                        f"<br>-log10: {-np.log10(v + 1e-10):.2f}"
                                        if use_log
                                        else ""
                                    )
                                    for s, v in zip(
                                        condition_data["sample_id_x"], sample_values
                                    )
                                ],
                                hoverinfo="text",
                                showlegend=False,
                                scalegroup=f"{metric}",
                                scalemode="width",
                            ),
                            row=1,
                            col=3,
                            secondary_y=secondary_y,
                        )
                        plot_created = True

        if not plot_created:
            for condition, color, x_offset in [
                ("A", junction_color_a, -0.2),
                ("B", junction_color_b, 0.2),
            ]:
                val_mean = subset_info.get(f"{metric}_mean_{condition}", np.nan)
                val_std = subset_info.get(f"{metric}_std_{condition}", 0)

                if not pd.isna(val_mean):
                    y_val = -np.log10(val_mean + 1e-10) if use_log else val_mean
                    y_err = (
                        abs(-np.log10(max(val_mean - val_std, 1e-10)) - y_val)
                        if (use_log and val_std > 0)
                        else val_std
                    )

                    fig.add_trace(
                        go.Bar(
                            x=[x_base + x_offset],
                            y=[y_val],
                            width=0.3,
                            error_y=(
                                dict(type="data", array=[y_err])
                                if val_std > 0
                                else None
                            ),
                            marker_color=color,
                            hovertext=f"{label} {condition}: {val_mean:.3f}±{val_std:.3f}",
                            hoverinfo="text",
                            showlegend=False,
                            opacity=0.7,
                        ),
                        row=1,
                        col=3,
                    )

    # Update layout
    title_text = f'Territory Analysis: {subset_info["transcript_id_str"]} with {len(superset_ids)} Superset Partners'

    # Calculate optimal height based on number of transcripts
    base_height = 400
    transcript_panel_height = max(200, min(400, n_transcripts * 50))
    territory_panel_height = 250
    total_height = base_height + transcript_panel_height + territory_panel_height

    fig.update_layout(
        title=title_text,
        height=total_height,
        width=1600,
        showlegend=True,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Add custom title annotations with proper positioning
    # Calculate y positions for titles based on row structure
    cumulative_heights = np.cumsum([0] + row_heights)
    total_height_units = sum(row_heights)

    # Subset title (row 1)
    fig.add_annotation(
        text=f'<b>Subset: {subset_info["transcript_id_str"]}</b>',
        xref="paper",
        yref="paper",
        x=0.35,  # Center of left panel
        y=1 - (cumulative_heights[0] / total_height_units) + 0.02,  # Above first row
        showarrow=False,
        font=dict(size=11),
        xanchor="center",
    )

    # Territory coverage title
    fig.add_annotation(
        text="<b>Territory Coverage (log2 scale)</b>",
        xref="paper",
        yref="paper",
        x=0.35,
        y=1
        - (cumulative_heights[n_transcripts] / total_height_units)
        - 0.05,  # Above territory row
        showarrow=False,
        font=dict(size=11),
        xanchor="center",
    )

    # Metrics panel title
    fig.add_annotation(
        text="<b>Metrics</b>",
        xref="paper",
        yref="paper",
        x=0.82,  # Right panel position
        y=0.98,  # Near top
        showarrow=False,
        font=dict(size=14),
        xanchor="center",
    )

    # Update x-axes for transcript rows with linked zoom/pan
    for i in range(1, n_transcripts + 2):
        fig.update_xaxes(
            range=[min_coord, max_coord],
            title="Genomic Position" if i == n_transcripts + 1 else "",
            row=i,
            col=1,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            matches="x" if i > 1 else None,
        )

    # Update x-axis for metrics panel
    fig.update_xaxes(
        range=[0.5, 4.5],
        tickmode="array",
        tickvals=[1, 2.5, 4],
        ticktext=["Penalty<br>(-log10)", "Evidence<br>(-log10)", "TPM"],
        showgrid=False,
        row=1,
        col=3,
    )

    # Update y-axes for transcript rows
    for i in range(1, n_transcripts + 1):
        fig.update_yaxes(
            range=[-0.5, 0.5],
            showticklabels=False,
            showgrid=False,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            row=i,
            col=1,
        )

    # Update y-axis for territory coverage
    y_max = np.log2(max_coverage + 1) * 1.2 if max_coverage > 0 else 5
    fig.update_yaxes(
        title="Coverage (log2)",
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=2,
        range=[-y_max, y_max],
        row=territory_row,
        col=1,
    )

    # Calculate dynamic ranges
    penalty_max = max(
        -np.log10(subset_info.get("adjusted_subset_penalty_mean_A", 0.1) + 1e-10),
        -np.log10(subset_info.get("adjusted_subset_penalty_mean_B", 0.1) + 1e-10),
    )
    tpm_max = max(subset_info.get("tpm_mean_A", 1), subset_info.get("tpm_mean_B", 1))

    fig.update_yaxes(
        range=[0, penalty_max * 2], row=1, col=3, secondary_y=False  # Double the range
    )

    fig.update_yaxes(
        range=[0, tpm_max * 2], row=1, col=3, secondary_y=True  # Double the range
    )

    fig.update_yaxes(
        title_text="Penalty / Evidence (-log10)", row=1, col=3, secondary_y=False
    )
    fig.update_yaxes(title_text="TPM", row=1, col=3, secondary_y=True, showgrid=False)

    # Add legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=junction_color_a),
            name="Condition A",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=junction_color_b),
            name="Condition B",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=territory_unique_color_a),
            name="Unique Territory A",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=territory_comp_color_a),
            name="Comparator Territory A",
            showlegend=True,
        )
    )

    return fig


def visualize_subset_junctions_only(
    aggregated_data: Dict,
    norm_data: Dict,
    subset_tid: int,
    show_individual_samples: bool = True,
):
    """
    Create visualization for subsets WITHOUT territory data, focusing on junction evidence.
    Features bilateral junction arcs, paired junction violin plots, and metrics panel.

    Parameters:
    -----------
    aggregated_data : dict
        Dictionary from aggregate_replicate_data with aggregated statistics
    norm_data : dict
        Dictionary from fetch_gene_data_raw after normalization
    subset_tid : int
        Transcript ID (int) of the subset transcript
    show_individual_samples : bool
        Whether to show individual sample distributions
    """

    # Extract relevant data
    df_transcripts = aggregated_data["transcripts_agg"]
    df_junctions = aggregated_data["junctions_agg"]
    df_subset_links = norm_data["subset_links"]

    # Get subset info
    subset_info = df_transcripts[
        df_transcripts["transcript_id_int"] == subset_tid
    ].iloc[0]

    # Check if this subset has significant penalties
    penalty_a = subset_info.get("adjusted_subset_penalty_mean_A", 1.0)
    penalty_b = subset_info.get("adjusted_subset_penalty_mean_B", 1.0)

    if (
        pd.isna(penalty_a)
        or pd.isna(penalty_b)
        or (penalty_a > 0.99 and penalty_b > 0.99)
    ):
        print(f"Skipping {subset_info['transcript_id_str']} - no significant penalties")
        return None

    # Get all superset partners for this subset
    superset_ids = df_subset_links[
        df_subset_links["subset_transcript_id_int"] == subset_tid
    ]["superset_transcript_id_int"].tolist()

    if len(superset_ids) == 0:
        print(f"No supersets found for subset {subset_info['transcript_id_str']}")
        return None

    # Get superset info
    superset_infos = df_transcripts[
        df_transcripts["transcript_id_int"].isin(superset_ids)
    ].sort_values("tpm_mean_A", ascending=False)

    # Get all relevant junctions
    all_transcript_ids = [subset_tid] + superset_ids
    all_junctions = df_junctions[
        df_junctions.index.get_level_values("transcript_id_int").isin(
            all_transcript_ids
        )
    ].copy()

    # Parse junction coordinates
    def parse_junction_coords(junction_str):
        parts = junction_str.split(":")
        coords = parts[1].split("-")
        return int(coords[0]), int(coords[1])

    # Get genomic range
    all_coords = []
    if not all_junctions.empty:
        for junc_str in all_junctions.index.get_level_values(
            "junction_id_str"
        ).unique():
            start, end = parse_junction_coords(junc_str)
            all_coords.extend([start, end])

    if not all_coords:
        print("No coordinate data found!")
        return None

    min_coord = min(all_coords) - 500
    max_coord = max(all_coords) + 500

    # Get unique junctions and classify them
    unique_junctions = all_junctions.index.get_level_values("junction_id_str").unique()
    subset_junctions_set = set(
        all_junctions[
            all_junctions.index.get_level_values("transcript_id_int") == subset_tid
        ].index.get_level_values("junction_id_str")
    )

    # Sort junctions by genomic position
    junction_positions = []
    for junc_str in unique_junctions:
        start, end = parse_junction_coords(junc_str)
        junction_positions.append((junc_str, start))
    junction_positions.sort(key=lambda x: x[1])
    sorted_junction_ids = [j[0] for j in junction_positions]

    # Classify junctions
    junction_categories = {}
    for junc_id in sorted_junction_ids:
        if junc_id in subset_junctions_set:
            junction_categories[junc_id] = "subset"
        else:
            junction_categories[junc_id] = "superset_only"

    # Calculate number of transcript rows needed
    n_transcripts = 1 + len(superset_ids)  # subset + all supersets

    # Create figure with dynamic row allocation
    transcript_height = min(0.08, 0.4 / n_transcripts)
    junction_panel_height = 0.4  # Fixed height for junction violin panel

    row_heights = []
    row_specs = []

    # Add rows for each transcript (subset + supersets)
    for i in range(n_transcripts):
        row_heights.append(transcript_height)
        if i == 0:
            row_specs.append(
                [
                    {"colspan": 2},
                    None,
                    {"rowspan": n_transcripts + 1, "secondary_y": True},
                ]
            )
        else:
            row_specs.append([{"colspan": 2}, None, None])

    # Add junction violin panel row
    row_heights.append(junction_panel_height)
    row_specs.append([{"colspan": 2}, None, None])

    fig = make_subplots(
        rows=n_transcripts + 1,
        cols=3,
        column_widths=[0.35, 0.35, 0.30],
        row_heights=row_heights,
        specs=row_specs,
        vertical_spacing=0.01,
        horizontal_spacing=0.07,
        subplot_titles=[""]
        * ((n_transcripts + 1) * 3),  # Empty titles, we'll add as annotations
    )

    # Color schemes
    junction_color_a = "rgba(150, 50, 50, 0.8)"
    junction_color_b = "rgba(50, 50, 150, 0.8)"
    subset_junction_color = (
        "rgba(50, 150, 50, 0.1)"  # Green background for subset junctions
    )
    superset_only_color = "rgba(150, 50, 150, 0.1)"  # Gray background for superset-only

    # Draw subset transcript
    draw_transcript_bilateral(
        fig, all_junctions, subset_tid, subset_info, subset_tid, 1
    )

    # Draw each superset
    for i, (_, superset_info) in enumerate(superset_infos.iterrows(), start=2):
        draw_transcript_bilateral(
            fig,
            all_junctions,
            superset_info["transcript_id_int"],
            superset_info,
            subset_tid,
            i,
        )

    # JUNCTION VIOLIN PANEL (bottom panel)
    junction_row = n_transcripts + 1

    # Get raw junction data if available for violin plots
    raw_junctions = pd.DataFrame()
    if "junctions_long" in norm_data:
        raw_junctions = norm_data["junctions_long"][
            norm_data["junctions_long"]["junction_id_str"].isin(sorted_junction_ids)
        ]

    # Create x positions for junctions
    x_positions = np.arange(len(sorted_junction_ids))

    # Calculate max junction value first to set background height
    max_junction_value = 0
    for junc_id in sorted_junction_ids:
        junc_data = all_junctions[
            all_junctions.index.get_level_values("junction_id_str") == junc_id
        ]
        if not junc_data.empty:
            mean_a = (
                junc_data["norm_read_count_mean_A"].mean()
                if "norm_read_count_mean_A" in junc_data.columns
                else 0
            )
            mean_b = (
                junc_data["norm_read_count_mean_B"].mean()
                if "norm_read_count_mean_B" in junc_data.columns
                else 0
            )
            std_a = (
                junc_data["norm_read_count_std_A"].mean()
                if "norm_read_count_std_A" in junc_data.columns
                else 0
            )
            std_b = (
                junc_data["norm_read_count_std_B"].mean()
                if "norm_read_count_std_B" in junc_data.columns
                else 0
            )
            max_junction_value = max(max_junction_value, mean_a + std_a, mean_b + std_b)

    # Set background height to 2x max value
    bg_height = max_junction_value * 2 if max_junction_value > 0 else 1000

    # Add background rectangles to distinguish junction categories
    for i, junc_id in enumerate(sorted_junction_ids):
        category = junction_categories[junc_id]

        bg_color = (
            subset_junction_color if category == "subset" else superset_only_color
        )

        fig.add_trace(
            go.Scatter(
                x=[i - 0.45, i - 0.45, i + 0.45, i + 0.45, i - 0.45],
                y=[0, bg_height, bg_height, 0, 0],
                fill="toself",
                fillcolor=bg_color,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                mode="none",  # No markers
            ),
            row=junction_row,
            col=1,
        )

    # Plot junction violins or bars
    for i, junc_id in enumerate(sorted_junction_ids):
        # Get junction data across all transcripts that have it
        junc_data = all_junctions[
            all_junctions.index.get_level_values("junction_id_str") == junc_id
        ]

        if not junc_data.empty:
            # Aggregate across transcripts if multiple have the same junction to collapse into single (should be same)
            mean_a = (
                junc_data["norm_read_count_mean_A"].mean()
                if "norm_read_count_mean_A" in junc_data.columns
                else 0
            )
            mean_b = (
                junc_data["norm_read_count_mean_B"].mean()
                if "norm_read_count_mean_B" in junc_data.columns
                else 0
            )
            std_a = (
                junc_data["norm_read_count_std_A"].mean()
                if "norm_read_count_std_A" in junc_data.columns
                else 0
            )
            std_b = (
                junc_data["norm_read_count_std_B"].mean()
                if "norm_read_count_std_B" in junc_data.columns
                else 0
            )

            # Try to get individual sample values for violin plots
            plot_created_a = False
            plot_created_b = False

            if not raw_junctions.empty and show_individual_samples:
                junc_raw = raw_junctions[raw_junctions["junction_id_str"] == junc_id]

                for condition, color, x_offset in [
                    ("A", junction_color_a, -0.15),
                    ("B", junction_color_b, 0.15),
                ]:
                    condition_data = (
                        junc_raw[junc_raw["group"] == condition]
                        if "group" in junc_raw.columns
                        else pd.DataFrame()
                    )

                    # Using size factor normalzied junction read counts
                    if (
                        not condition_data.empty
                        and "norm_read_count" in condition_data.columns
                    ):
                        values = condition_data["norm_read_count"].dropna()
                        if len(values) > 0:
                            # Hover for violin plots
                            category_text = (
                                "Subset junction"
                                if junction_categories[junc_id] == "subset"
                                else "Superset-only junction"
                            )
                            hover_texts = []
                            for idx, val in enumerate(values.values):
                                sample_id = (
                                    condition_data.iloc[idx]["sample_id_x"]
                                    if "sample_id_x" in condition_data.columns
                                    else f"Sample {idx+1}"
                                )
                                hover_text = f"<b>{junc_id}</b><br>"
                                hover_text += f"{category_text}<br>"
                                hover_text += f"Sample: {sample_id}<br>"
                                hover_text += f"Condition: {condition}<br>"
                                hover_text += f"Normalized reads: {val:.1f}<br>"
                                hover_text += f"Mean across samples: {mean_a if condition == 'A' else mean_b:.1f}"
                                hover_texts.append(hover_text)

                            fig.add_trace(
                                go.Violin(
                                    x=[i + x_offset] * len(values),
                                    y=values,
                                    width=0.25,
                                    marker_color=color,
                                    opacity=0.7,
                                    name=f"{condition}",
                                    box_visible=True,
                                    meanline_visible=True,
                                    points="all",
                                    pointpos=-0.8,
                                    jitter=0.05,
                                    hovertext=hover_texts,
                                    hoverinfo="text",
                                    hoverlabel=dict(bgcolor="white", font_size=10),
                                    showlegend=False,
                                ),
                                row=junction_row,
                                col=1,
                            )
                            if condition == "A":
                                plot_created_a = True
                            else:
                                plot_created_b = True

            # Only use bars if violin wasn't created for that condition
            if not plot_created_a and not pd.isna(mean_a):
                category_text = (
                    "Subset junction"
                    if junction_categories[junc_id] == "subset"
                    else "Superset-only junction"
                )
                hover_text = f"<b>{junc_id}</b><br>"
                hover_text += f"{category_text}<br>"
                hover_text += f"Condition A<br>"
                hover_text += f"Mean: {mean_a:.1f}<br>"
                hover_text += f"Std: {std_a:.1f}<br>"
                hover_text += f"N samples: {junc_data['norm_read_count_count_A'].mean() if 'norm_read_count_count_A' in junc_data.columns else 'N/A'}"

                fig.add_trace(
                    go.Bar(
                        x=[i - 0.15],
                        y=[mean_a],
                        width=0.25,
                        error_y=dict(type="data", array=[std_a]) if std_a > 0 else None,
                        marker_color=junction_color_a,
                        opacity=0.7,
                        hovertext=hover_text,
                        hoverinfo="text",
                        hoverlabel=dict(bgcolor="white", font_size=10),
                        showlegend=False,
                    ),
                    row=junction_row,
                    col=1,
                )

            if not plot_created_b and not pd.isna(mean_b):
                category_text = (
                    "Subset junction"
                    if junction_categories[junc_id] == "subset"
                    else "Superset-only junction"
                )
                hover_text = f"<b>{junc_id}</b><br>"
                hover_text += f"{category_text}<br>"
                hover_text += f"Condition B<br>"
                hover_text += f"Mean: {mean_b:.1f}<br>"
                hover_text += f"Std: {std_b:.1f}<br>"
                hover_text += f"N samples: {junc_data['norm_read_count_count_B'].mean() if 'norm_read_count_count_B' in junc_data.columns else 'N/A'}"

                fig.add_trace(
                    go.Bar(
                        x=[i + 0.15],
                        y=[mean_b],
                        width=0.25,
                        error_y=dict(type="data", array=[std_b]) if std_b > 0 else None,
                        marker_color=junction_color_b,
                        opacity=0.7,
                        hovertext=hover_text,
                        hoverinfo="text",
                        hoverlabel=dict(bgcolor="white", font_size=10),
                        showlegend=False,
                    ),
                    row=junction_row,
                    col=1,
                )

    # RIGHT PANEL: Metrics (same as territory version)
    raw_transcripts = pd.DataFrame()
    if "transcripts_long" in norm_data:
        raw_transcripts = norm_data["transcripts_long"][
            norm_data["transcripts_long"]["transcript_id_int"] == subset_tid
        ]

    metrics = [
        ("adjusted_subset_penalty", "Penalty", True, False),
        ("completeness_penalty", "Completeness", True, False),
        ("tpm", "TPM", False, True),
    ]

    metric_x_positions = [1, 2.5, 4]

    for i, (metric, label, use_log, secondary_y) in enumerate(metrics):
        x_base = metric_x_positions[i]
        plot_created_a = False
        plot_created_b = False

        if not raw_transcripts.empty and f"{metric}" in raw_transcripts.columns:
            for condition, color, x_offset in [
                ("A", junction_color_a, -0.2),
                ("B", junction_color_b, 0.2),
            ]:
                condition_data = raw_transcripts[raw_transcripts["group"] == condition]
                if not condition_data.empty and metric in condition_data.columns:
                    sample_values = condition_data[metric].dropna()

                    if len(sample_values) > 0:
                        display_values = (
                            -np.log10(sample_values + 1e-10)
                            if use_log
                            else sample_values
                        )

                        # Enhanced hover for metrics
                        hover_texts = []
                        for idx, (val, disp_val) in enumerate(
                            zip(sample_values.values, display_values)
                        ):
                            sample_id = (
                                condition_data.iloc[idx]["sample_id_x"]
                                if "sample_id_x" in condition_data.columns
                                else f"Sample {idx+1}"
                            )
                            hover_text = f"<b>{label} - Condition {condition}</b><br>"
                            hover_text += f"Sample: {sample_id}<br>"
                            hover_text += f"Value: {val:.4f}"
                            if use_log:
                                hover_text += f"<br>-log10: {disp_val:.2f}"
                            hover_text += f"<br>Mean: {subset_info.get(f'{metric}_mean_{condition}', 0):.4f}"
                            hover_text += f"<br>Std: {subset_info.get(f'{metric}_std_{condition}', 0):.4f}"
                            hover_texts.append(hover_text)

                        fig.add_trace(
                            go.Violin(
                                x=[x_base + x_offset] * len(display_values),
                                y=display_values,
                                width=0.3,
                                marker_color=color,
                                opacity=0.6,
                                box_visible=True,
                                meanline_visible=True,
                                points="all",
                                pointpos=-0.8,
                                jitter=0.05,
                                hovertext=hover_texts,
                                hoverinfo="text",
                                hoverlabel=dict(bgcolor="white", font_size=10),
                                showlegend=False,
                            ),
                            row=1,
                            col=3,
                            secondary_y=secondary_y,
                        )
                        if condition == "A":
                            plot_created_a = True
                        else:
                            plot_created_b = True

        # Only create bars if violins weren't created
        if not plot_created_a:
            val_mean = subset_info.get(f"{metric}_mean_A", np.nan)
            val_std = subset_info.get(f"{metric}_std_A", 0)
            val_count = subset_info.get(f"{metric}_count_A", 0)

            if not pd.isna(val_mean):
                y_val = -np.log10(val_mean + 1e-10) if use_log else val_mean
                y_err = (
                    abs(-np.log10(max(val_mean - val_std, 1e-10)) - y_val)
                    if (use_log and val_std > 0)
                    else val_std
                )

                hover_text = f"<b>{label} - Condition A</b><br>"
                hover_text += f"Mean: {val_mean:.4f}<br>"
                hover_text += f"Std: {val_std:.4f}<br>"
                if use_log:
                    hover_text += f"-log10: {y_val:.2f}<br>"
                hover_text += f"N samples: {val_count:.0f}"

                fig.add_trace(
                    go.Bar(
                        x=[x_base - 0.2],
                        y=[y_val],
                        width=0.3,
                        error_y=(
                            dict(type="data", array=[y_err]) if val_std > 0 else None
                        ),
                        marker_color=junction_color_a,
                        hovertext=hover_text,
                        hoverinfo="text",
                        hoverlabel=dict(bgcolor="white", font_size=10),
                        showlegend=False,
                        opacity=0.7,
                    ),
                    row=1,
                    col=3,
                    secondary_y=secondary_y,
                )

        if not plot_created_b:
            val_mean = subset_info.get(f"{metric}_mean_B", np.nan)
            val_std = subset_info.get(f"{metric}_std_B", 0)
            val_count = subset_info.get(f"{metric}_count_B", 0)

            if not pd.isna(val_mean):
                y_val = -np.log10(val_mean + 1e-10) if use_log else val_mean
                y_err = (
                    abs(-np.log10(max(val_mean - val_std, 1e-10)) - y_val)
                    if (use_log and val_std > 0)
                    else val_std
                )

                hover_text = f"<b>{label} - Condition B</b><br>"
                hover_text += f"Mean: {val_mean:.4f}<br>"
                hover_text += f"Std: {val_std:.4f}<br>"
                if use_log:
                    hover_text += f"-log10: {y_val:.2f}<br>"
                hover_text += f"N samples: {val_count:.0f}"

                fig.add_trace(
                    go.Bar(
                        x=[x_base + 0.2],
                        y=[y_val],
                        width=0.3,
                        error_y=(
                            dict(type="data", array=[y_err]) if val_std > 0 else None
                        ),
                        marker_color=junction_color_b,
                        hovertext=hover_text,
                        hoverinfo="text",
                        hoverlabel=dict(bgcolor="white", font_size=10),
                        showlegend=False,
                        opacity=0.7,
                    ),
                    row=1,
                    col=3,
                    secondary_y=secondary_y,
                )

    # Update layout
    title_text = f'Junction Analysis (No Territory Data): {subset_info["transcript_id_str"]} with {len(superset_ids)} Superset Partners'

    base_height = 400
    transcript_panel_height = max(200, min(400, n_transcripts * 50))
    junction_panel_height = 250
    total_height = base_height + transcript_panel_height + junction_panel_height

    fig.update_layout(
        title=title_text,
        height=total_height,
        width=1600,
        showlegend=True,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Add custom title annotations
    cumulative_heights = np.cumsum([0] + row_heights)
    total_height_units = sum(row_heights)

    # Subset title
    fig.add_annotation(
        text=f'<b>Subset: {subset_info["transcript_id_str"]}</b>',
        xref="paper",
        yref="paper",
        x=0.35,
        y=1 - (cumulative_heights[0] / total_height_units) + 0.02,
        showarrow=False,
        font=dict(size=11),
        xanchor="center",
    )

    # Junction panel title
    fig.add_annotation(
        text="<b>Junction Read Counts (Normalized)</b>",
        xref="paper",
        yref="paper",
        x=0.35,
        y=1 - (cumulative_heights[n_transcripts] / total_height_units) - 0.05,
        showarrow=False,
        font=dict(size=11),
        xanchor="center",
    )

    # Add legend for junction categories
    fig.add_annotation(
        text='<span style="background-color:rgba(50,150,50,0.2)">■</span> Subset junctions  '
        + '<span style="background-color:rgba(150,150,150,0.1)">■</span> Superset-only',
        xref="paper",
        yref="paper",
        x=0.35,
        y=1 - (cumulative_heights[n_transcripts + 1] / total_height_units) + 0.02,
        showarrow=False,
        font=dict(size=9),
        xanchor="center",
    )

    # Metrics panel title
    fig.add_annotation(
        text="<b>Metrics</b>",
        xref="paper",
        yref="paper",
        x=0.82,
        y=0.98,
        showarrow=False,
        font=dict(size=14),
        xanchor="center",
    )

    # Update x-axes
    for i in range(1, n_transcripts + 1):
        fig.update_xaxes(
            range=[min_coord, max_coord],
            title="",
            row=i,
            col=1,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            matches="x" if i > 1 else None,
        )

    # Update x-axis for junction panel
    fig.update_xaxes(
        range=[-0.5, len(sorted_junction_ids) - 0.5],
        tickmode="array",
        tickvals=list(range(len(sorted_junction_ids))),
        ticktext=[
            j.split(":")[1] if len(j.split(":")[1]) < 20 else "..."
            for j in sorted_junction_ids
        ],
        tickangle=45,
        title="Junction Coordinates",
        row=junction_row,
        col=1,
    )

    # Update x-axis for metrics panel
    fig.update_xaxes(
        range=[0.5, 4.5],
        tickmode="array",
        tickvals=[1, 2.5, 4],
        ticktext=["S. penalty<br>(-log10)", "Comp penalty<br>(-log10)", "TPM"],
        showgrid=False,
        row=1,
        col=3,
    )

    # Update y-axes for transcript rows
    for i in range(1, n_transcripts + 1):
        fig.update_yaxes(
            range=[-0.5, 0.5],
            showticklabels=False,
            showgrid=False,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            row=i,
            col=1,
        )

    # Update y-axis for junction panel
    fig.update_yaxes(
        title="Normalized Junction Read Count",
        range=[0, max_junction_value * 1.2] if max_junction_value > 0 else [0, 10],
        row=junction_row,
        col=1,
    )

    # Update y-axes for metrics panel
    fig.update_yaxes(
        title_text="Sub & Completeness Penalties (-log10)",
        row=1,
        col=3,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Subset TPM", row=1, col=3, secondary_y=True, showgrid=False
    )

    # Add legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=junction_color_a),
            name="Condition A",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=junction_color_b),
            name="Condition B",
            showlegend=True,
        )
    )

    return fig


def plot_all_subsets_junction_focus(
    aggregated_data: Dict,
    raw_data: Dict,
    show_individual_samples: bool = True,
    include_territory_subsets: bool = False,
):
    """
    Create junction-focused visualizations for subsets, optionally including those with territory data.

    Parameters:
    -----------
    aggregated_data : dict
        Dictionary from aggregate_replicate_data
    raw_data : dict
        Dictionary from fetch_gene_data_raw
    show_individual_samples : bool
        Whether to show individual sample distributions
    include_territory_subsets : bool
        If True, create junction plots even for subsets with territory data
        If False, only create plots for subsets WITHOUT territory data

    Returns:
    --------
    list of plotly figures
    """
    df_subset_links = raw_data["subset_links"]
    df_transcripts = aggregated_data["transcripts_agg"]
    df_territories = aggregated_data.get("territories_agg", pd.DataFrame())

    # Get unique subset IDs
    subset_ids = df_subset_links["subset_transcript_id_int"].unique()

    figures = []

    for subset_tid in subset_ids:
        # Check if this subset has significant penalties
        subset_info = df_transcripts[df_transcripts["transcript_id_int"] == subset_tid]
        if subset_info.empty:
            continue

        subset_info = subset_info.iloc[0]
        subset_name = subset_info["transcript_id_str"]

        # Check if subset has territory data
        has_territory = False
        if not df_territories.empty:
            subset_territories = df_territories[
                (
                    df_territories.index.get_level_values("transcript_id_int")
                    == subset_tid
                )
                & (df_territories.index.get_level_values("territory_role") == "unique")
            ]
            has_territory = not subset_territories.empty

        # Skip if has territory data and we're not including those
        if has_territory and not include_territory_subsets:
            print(
                f"Skipping {subset_name} - has territory data (use territory visualization instead)"
            )
            continue

        print(f"Creating junction-focused visualization for subset {subset_name}")

        fig = visualize_subset_junctions_only(
            aggregated_data, raw_data, subset_tid, show_individual_samples
        )

        if fig:
            figures.append(fig)

    return figures


def plot_all_subsets_with_territories(
    aggregated_data: Dict, raw_data: Dict, show_individual_samples: bool = True
):
    """
    Create territory visualizations for all subsets with their superset partners.

    Parameters:
    -----------
    aggregated_data : dict
        Dictionary from aggregate_replicate_data
    raw_data : dict
        Dictionary from fetch_gene_data_raw
    show_individual_samples : bool
        Whether to show individual sample distributions

    Returns:
    --------
    list of plotly figures
    """
    df_subset_links = raw_data["subset_links"]
    df_transcripts = aggregated_data["transcripts_agg"]

    # Get unique subset IDs
    subset_ids = df_subset_links["subset_transcript_id_int"].unique()

    figures = []

    for subset_tid in subset_ids:
        # Check if this subset has significant penalties
        subset_info = df_transcripts[df_transcripts["transcript_id_int"] == subset_tid]
        if subset_info.empty:
            continue

        subset_info = subset_info.iloc[0]
        subset_name = subset_info["transcript_id_str"]

        print(
            f"Creating territory visualization for subset {subset_name} with all its supersets"
        )

        fig = visualize_subset_focused_territories(
            aggregated_data, raw_data, subset_tid, show_individual_samples
        )

        if fig:
            figures.append(fig)

    return figures


def export_multipanel_with_navigation(
    figure_list, sample_names, output_file="output.html"
):
    """
    Export figures using iframes with unique subdirectories to avoid overwrites.
    """

    # Create a unique subdirectory based on the output filename
    base_dir = os.path.dirname(output_file)
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    figures_dir = os.path.join(base_dir, f"{base_name}_figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Save each figure as a complete HTML file
    html_files = []
    for i, (fig, name) in enumerate(zip(figure_list, sample_names)):
        fig_path = os.path.join(figures_dir, f"figure_{i}.html")
        fig.write_html(fig_path)
        # Use relative path for the iframe
        html_files.append(f"{base_name}_figures/figure_{i}.html")

    # Create the main navigation HTML
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
            .navigation {{
                text-align: center;
                padding: 15px;
                background: #f0f0f0;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1000;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .nav-button {{
                padding: 10px 20px;
                margin: 0 10px;
                cursor: pointer;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
            }}
            .nav-button:disabled {{
                background: #cccccc;
                cursor: not-allowed;
            }}
            #sample-name {{
                display: inline-block;
                margin: 0 20px;
                font-size: 18px;
                font-weight: bold;
            }}
            #figure-frame {{
                position: fixed;
                top: 60px;
                left: 0;
                right: 0;
                bottom: 0;
                width: 100%;
                height: calc(100vh - 60px);
                border: none;
            }}
        </style>
    </head>
    <body>
        <div class="navigation">
            <button class="nav-button" id="prev-btn" onclick="navigate(-1)">← Previous</button>
            <span id="sample-name">Loading...</span>
            <button class="nav-button" id="next-btn" onclick="navigate(1)">Next →</button>
            <select id="sample-select" onchange="jumpTo(this.value)" style="margin-left: 20px; padding: 5px;">
                {options}
            </select>
        </div>
        <iframe id="figure-frame" src=""></iframe>
        
        <script>
            var currentIndex = 0;
            var sampleNames = {names_json};
            var htmlFiles = {html_files_json};
            
            function showFigure(index) {{
                currentIndex = index;
                document.getElementById('figure-frame').src = htmlFiles[index];
                
                // Update navigation
                document.getElementById('sample-name').innerText = 
                    sampleNames[index] + ' (' + (index + 1) + ' / ' + sampleNames.length + ')';
                document.getElementById('prev-btn').disabled = (index === 0);
                document.getElementById('next-btn').disabled = (index === sampleNames.length - 1);
                document.getElementById('sample-select').value = index;
            }}
            
            function navigate(direction) {{
                var newIndex = currentIndex + direction;
                if (newIndex >= 0 && newIndex < sampleNames.length) {{
                    showFigure(newIndex);
                }}
            }}
            
            function jumpTo(index) {{
                showFigure(parseInt(index));
            }}
            
            // Keyboard navigation
            document.addEventListener('keydown', function(event) {{
                if (event.key === 'ArrowLeft') navigate(-1);
                if (event.key === 'ArrowRight') navigate(1);
            }});
            
            // Initialize
            showFigure(0);
        </script>
    </body>
    </html>
    """

    # Create options for dropdown
    options = "".join(
        [f'<option value="{i}">{name}</option>' for i, name in enumerate(sample_names)]
    )

    # Fill template
    html_content = html_template.format(
        options=options,
        names_json=json.dumps(sample_names),
        html_files_json=json.dumps(html_files),
    )

    # Write main file
    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"Exported to {output_file} with {len(figure_list)} figures")
    return output_file


def export_figures_as_svg_archive(figures, sample_names, zip_path, gene_id):
    """Export all figures as SVGs in a zip archive."""

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fig, name in zip(figures, sample_names):
            # Write SVG to zip directly without intermediate file
            svg_string = fig.to_image(format="svg", width=1600, height=1000)
            zipf.writestr(f"{name}_{gene_id}_heatmap.svg", svg_string)

        # Add a README
        readme = f"""
        {gene_id} Analysis Figures
        Generated: {pd.Timestamp.now()}
        
        Contents:
        {chr(10).join([f'- {name}_heatmap.svg' for name in sample_names])}
        
        These SVG files can be edited in Adobe Illustrator, Inkscape, or similar vector graphics programs.
        """
        zipf.writestr("README.txt", readme)

    print(f"Created archive: {zip_path}")
    return zip_path


def create_combined_pdf(figures, sample_names, output_file):
    """Create combined PDF without temp files."""
    merger = PdfMerger()

    for fig, name in zip(figures, sample_names):
        # Create PDF in memory
        pdf_bytes = fig.to_image(format="pdf", width=1600, height=1000)
        pdf_buffer = io.BytesIO(pdf_bytes)
        merger.append(pdf_buffer)

    # Write final PDF
    with open(output_file, "wb") as f:
        merger.write(f)
    merger.close()

    return output_file


def export_data_to_excel(aggregated_data, norm_data, output_file="analysis_data.xlsx"):
    """
    Export both data dictionaries to Excel with meaningful tab names.

    Parameters:
    -----------
    aggregated_data : dict
        Dictionary from aggregate_replicate_data
    raw_data : dict
        Dictionary from fetch_gene_data_raw
    output_file : str
        Output Excel filename
    """
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        # Export aggregated data
        for key, df in aggregated_data.items():
            if isinstance(df, pd.DataFrame):
                sheet_name = f"agg_{key}"[:31]  # Excel limit

                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    # Flatten the column names
                    df_copy = df.copy()
                    df_copy.columns = [
                        "_".join(map(str, col)).strip() for col in df.columns.values
                    ]
                    df_reset = df_copy.reset_index() if df_copy.index.name else df_copy
                else:
                    df_reset = (
                        df.reset_index()
                        if isinstance(df.index, pd.MultiIndex)
                        else df.copy()
                    )

                df_reset.to_excel(writer, sheet_name=sheet_name, index=False)

        # Export raw data
        for key, df in norm_data.items():
            if isinstance(df, pd.DataFrame):
                sheet_name = f"raw_{key}"[:31]

                # Handle MultiIndex columns here too
                if isinstance(df.columns, pd.MultiIndex):
                    df_copy = df.copy()
                    df_copy.columns = [
                        "_".join(map(str, col)).strip() for col in df.columns.values
                    ]
                    df_reset = df_copy.reset_index() if df_copy.index.name else df_copy
                else:
                    df_reset = (
                        df.reset_index()
                        if isinstance(df.index, pd.MultiIndex)
                        else df.copy()
                    )

                df_reset.to_excel(writer, sheet_name=sheet_name, index=False)

        # Add a README sheet with metadata
        readme_data = {
            "Sheet Name": [],
            "Description": [],
            "Row Count": [],
            "Column Count": [],
        }

        for key, df in {**aggregated_data, **norm_data}.items():
            if isinstance(df, pd.DataFrame):
                readme_data["Sheet Name"].append(key)
                readme_data["Description"].append(
                    get_description(key)
                )  # Helper function
                readme_data["Row Count"].append(len(df))
                readme_data["Column Count"].append(len(df.columns))

        readme_df = pd.DataFrame(readme_data)
        sum_df = make_summary_data(aggregated_data, norm_data)
        readme_df = pd.concat([sum_df, readme_df])
        readme_df.to_excel(writer, sheet_name="README", index=False)

        # Format the README sheet
        workbook = writer.book
        worksheet = writer.sheets["README"]

        # Add header formatting
        header_format = workbook.add_format(
            {"bold": True, "bg_color": "#D7E4BD", "border": 1}
        )

        for col_num, col_name in enumerate(readme_df.columns):
            worksheet.write(0, col_num, col_name, header_format)
            worksheet.set_column(col_num, col_num, 20)  # Set column width

    print(f"Exported data to {output_file}")
    return output_file


def make_summary_data(aggregated_data, norm_data):
    summary = {
        "Metric": [
            "Gene ID",
            "Total Transcripts",
            "Total Samples",
            "Subsets with Penalties",
            "Total Junctions",
            "Date Generated",
        ],
        "Value": [
            norm_data["gene_id"],
            len(aggregated_data.get("transcripts_agg", [])),
            len(norm_data.get("size_factors", [])),
            sum(
                aggregated_data.get("transcripts_agg", pd.DataFrame())[
                    "adjusted_subset_penalty_mean_A"
                ]
                < 0.99
            ),
            len(aggregated_data.get("junctions_agg", [])),
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        ],
    }
    return pd.DataFrame(summary)


def get_description(key):
    """Helper to provide descriptions for each data type."""
    descriptions = {
        "gene_id": "Gene identifier",
        "annotations": "Base transcript annotations",
        "transcripts_agg": "Aggregated transcript metrics by replicate group",
        "junctions_agg": "Junction read counts aggregated by condition",
        "territories_agg": "Territory coverage statistics",
        "aggregated_agg": "Gene-level metrics by replicate group",
        "transcripts_long": "Raw transcript data in long format",
        "junctions_long": "Raw junction counts per sample",
        "territories_long": "Raw territory coverage per sample",
        "subset_links": "Subset-superset transcript relationships",
        "size_factors": "Sample normalization factors",
        "aggregated_long": "Raw gene-level data in long format",
    }
    return descriptions.get(key, "Data table")


def process_related_genes_subprocess(raw_data, current_args, output_dir):
    """
    Process related genes sequentially via subprocess.
    """
    if not current_args.include_related_genes:
        return

    related_genes = raw_data.get("other_genes_with_shared_junctions", [])
    if not related_genes:
        print("No other genes share junctions with the query gene")
        return

    print(f"\n--- Processing {len(related_genes)} related genes ---")

    # Save summary
    summary = {
        "query_gene": raw_data["gene_id"],
        "related_genes": related_genes,
        "shared_junctions": list(raw_data["multi_gene_junctions"]),
        "n_shared": len(raw_data["multi_gene_junctions"]),
    }

    with open(
        os.path.join(output_dir, f"{raw_data['gene_id']}_related_genes_summary.json"),
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    # Build command
    base_cmd = ["majec_visualize"]
    base_args = []

    # Copy over essential arguments
    for arg in [
        "db",
        "group_A_str",
        "group_A_file",
        "group_B_str",
        "group_B_file",
        "sample_list",
        "delta_psi_bootstrap",
    ]:
        if hasattr(current_args, arg) and getattr(current_args, arg):
            base_args.extend([f"--{arg}", str(getattr(current_args, arg))])

    # Handle boolean flags separately - only add flag if True, no value needed
    for flag in ["save_svg", "save_pdf", "export_excel"]:
        if hasattr(current_args, flag) and getattr(current_args, flag):
            base_args.append(f"--{flag}")

    # Process each gene
    for i, gene_id in enumerate(related_genes, 1):
        print(
            f"  [{i}/{len(related_genes)}] Processing {gene_id}...", end="", flush=True
        )

        gene_cmd = (
            base_cmd + base_args + ["--gene", gene_id, "--output_dir", output_dir]
        )

        try:
            subprocess.run(gene_cmd, capture_output=True, text=True, check=True)
            print(" ✓")
        except subprocess.CalledProcessError as e:
            print(f" ✗ (see {gene_id}_error.txt)")
            with open(os.path.join(output_dir, f"{gene_id}_error.txt"), "w") as f:
                f.write(e.stderr)

    print(f"  Results saved to: {output_dir}")


def normalize_args_paths(args):
    """
    Convert all path arguments to absolute paths at startup.
    """
    # Database path
    if hasattr(args, "database"):
        args.database = os.path.abspath(args.database)
        if not os.path.exists(args.database):
            raise FileNotFoundError(f"Database not found: {args.database}")

    # Sample list
    if hasattr(args, "sample_list") and args.sample_list:
        args.sample_list = os.path.abspath(args.sample_list)
        if not os.path.exists(args.sample_list):
            raise FileNotFoundError(f"Sample list not found: {args.sample_list}")

    # Output directory - create if doesn't exist
    if hasattr(args, "output_dir"):
        args.output_dir = os.path.abspath(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

    return args


# ==============================================================================
# 3. USER INPUT PARSING LAYER
# ==============================================================================


def parse_criteria_string(criteria_string):
    """Parses a key-value string like 'cell_line=U937; treatment=DMSO'"""
    criteria = []
    for part in criteria_string.split(";"):
        if "=" not in part:
            continue
        key, values_str = part.strip().split("=", 1)
        values = [v.strip() for v in values_str.split(",")]
        criteria.append({"column": key, "values": values})
    return criteria


def parse_criteria_file(criteria_file):
    """Parses a simple key = value file for group definitions."""
    try:
        with open(criteria_file, "r") as f:
            return parse_criteria_string(f.read())
    except FileNotFoundError:
        print(f"ERROR: Criteria file not found at: {criteria_file}", file=sys.stderr)
        sys.exit(1)


def get_sample_ids_from_criteria(db_conn, criteria):
    """
    Builds and executes a SQL query to get sample IDs based on a list of criteria.
    """
    if not criteria:
        return []

    # Start building the WHERE clause
    where_clauses = []
    params = []
    for criterion in criteria:
        col = criterion["column"]
        vals = criterion["values"]
        # Create a clause like "cell_line IN (?, ?)"
        placeholders = ",".join(["?"] * len(vals))
        where_clauses.append(f'"{col}" IN ({placeholders})')
        params.extend(vals)

    # Combine all clauses with AND
    where_statement = " AND ".join(where_clauses)

    query = f"SELECT sample_id_int FROM sample_metadata WHERE {where_statement};"

    df_ids = pd.read_sql_query(query, db_conn, params=params)
    return df_ids["sample_id_int"].tolist()


# ==============================================================================
# 4. MAIN APPLICATION ORCHESTRATOR
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive analysis plots from a MAJEC database."
    )
    # --- Inputs ---
    parser.add_argument(
        "--db", required=True, help="Path to the MAJEC SQLite database."
    )
    parser.add_argument(
        "--gene", required=True, help="Gene ID or transcript ID to analyze."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save the output HTML files."
    )
    parser.add_argument(
        "--delta_psi_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for delta PSI confidence intervals.",
    )
    parser.add_argument(
        "--save_svg", action="store_true", help="Also save in SVG format."
    )
    parser.add_argument(
        "--save_pdf", action="store_true", help="Also save in pdf format."
    )
    parser.add_argument(
        "--export_excel",
        action="store_true",
        help="Export data to Excel format (requires openpyxl or xlsxwriter)",
    )
    parser.add_argument(
        "--include_related_genes",
        action="store_true",
        help="Also process all genes that share junctions with the query gene",
    )

    # --- Group A Definition (Mutually Exclusive) ---
    group_A = parser.add_mutually_exclusive_group(required=True)
    group_A.add_argument(
        "--group_A_str",
        help="Define Group A with a key-value string (e.g., 'cell_line=U937;treatment=DMSO').",
    )
    group_A.add_argument(
        "--group_A_file", help="Define Group A with a path to a query file."
    )

    # --- Group B Definition (Mutually Exclusive) ---
    group_B = parser.add_mutually_exclusive_group(required=True)
    group_B.add_argument(
        "--group_B_str", help="Define Group B with a key-value string."
    )
    group_B.add_argument(
        "--group_B_file", help="Define Group B with a path to a query file."
    )

    args = parser.parse_args()
    args = normalize_args_paths(args)

    # 1. Connect to DB
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    # 2. Parse User-Defined Groups to get lists of sample IDs
    print("--- Defining Comparison Groups ---")
    criteria_A = (
        parse_criteria_string(args.group_A_str)
        if args.group_A_str
        else parse_criteria_file(args.group_A_file)
    )
    criteria_B = (
        parse_criteria_string(args.group_B_str)
        if args.group_B_str
        else parse_criteria_file(args.group_B_file)
    )

    group_A_ids = get_sample_ids_from_criteria(con, criteria_A)
    group_B_ids = get_sample_ids_from_criteria(con, criteria_B)

    if not group_A_ids or not group_B_ids:
        print(
            "ERROR: One or both of the defined groups resulted in zero samples. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  -> Group A includes {len(group_A_ids)} samples.")
    print(f"  -> Group B includes {len(group_B_ids)} samples.")

    # 3. Fetch all the raw data for these samples
    raw_data = fetch_gene_data_raw_with_junction_context(
        con, args.gene, group_A_ids + group_B_ids
    )

    if not raw_data:
        con.close()
        print(
            "ERROR: No data found for the specified gene or transcript ID. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 4. Calculate all the aggregate statistics
    aggregated_data, norm_data = aggregate_replicate_data(
        raw_data.copy(), group_A_ids, group_B_ids
    )

    # 5. Add additional computed statistics
    aggregated_data = calculate_junction_confidence_intervals(
        aggregated_data, norm_data, n_boot=args.delta_psi_bootstrap
    )
    aggregated_data = calculate_expression_statistics(aggregated_data, norm_data)

    # Make individual sample penalty plots

    # create output dir if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    sample_int_to_string = dict(
        zip(
            norm_data["transcripts_long"]["sample_id_int"].values,
            norm_data["transcripts_long"]["sample_id_x"].values,
        )
    )
    figures = []
    sample_names = []
    for sid, sname in sample_int_to_string.items():
        figures.append(
            create_junction_penalty_heatmap_plotly(
                select_and_format_single_sample(raw_data, sid),
                sample_name=sname,
                output_prefix=None,
            )
        )
        sample_names.append(sname)

    # Save all individual sample penalty heatmaps to a single HTML with navigation
    penalty_heatmap_save_path = (
        f"{args.output_dir}/{raw_data['gene_id']}_all_samples_penalty_heatmaps.html"
    )
    export_multipanel_with_navigation(figures, sample_names, penalty_heatmap_save_path)

    # Save all individual sample penalty heatmaps to a single zip with SVGs
    if args.save_svg and STATIC_EXPORT_AVAILABLE:
        export_figures_as_svg_archive(
            figures,
            sample_names,
            f"{args.output_dir}/{raw_data['gene_id']}_all_samples_penalty_heatmaps.zip",
            raw_data["gene_id"],
        )
        print(
            f"All individual sample penalty heatmaps saved to: {penalty_heatmap_save_path}"
        )

    if args.save_pdf and HAVE_PYPDF2 and STATIC_EXPORT_AVAILABLE:
        pdf_save_path = (
            f"{args.output_dir}/{raw_data['gene_id']}_all_samples_penalty_heatmaps.pdf"
        )
        create_combined_pdf(figures, sample_names, pdf_save_path)
        print(f"All individual sample penalty heatmaps saved to: {pdf_save_path}")

    # Make aggregate plots
    # Agg junction heatmap
    agg_heatmap = create_differential_junction_heatmap(
        aggregated_data, raw_data["gene_id"]
    )
    agg_heatmap_save_path = (
        f"{args.output_dir}/{raw_data['gene_id']}_aggregate_junction_heatmap.html"
    )
    agg_heatmap.write_html(agg_heatmap_save_path)
    print(f"Aggregate junction heatmap saved to: {agg_heatmap_save_path}")
    if args.save_svg and STATIC_EXPORT_AVAILABLE:
        svg_save_path = (
            f"{args.output_dir}/{raw_data['gene_id']}_aggregate_junction_heatmap.svg"
        )
        agg_heatmap.write_image(svg_save_path, width=1600, height=1000)
        print(f"Aggregate junction heatmap SVG saved to: {svg_save_path}")
    if args.save_pdf and STATIC_EXPORT_AVAILABLE:
        pdf_save_path = (
            f"{args.output_dir}/{raw_data['gene_id']}_aggregate_junction_heatmap.pdf"
        )
        agg_heatmap.write_image(pdf_save_path, width=1600, height=1000)
        print(f"Aggregate junction heatmap PDF saved to: {pdf_save_path}")

    # Agg arc plot
    agg_arc_plot = create_junction_arc_plot_aggregated(
        aggregated_data, norm_data, raw_data["gene_id"], track_spacing=3
    )
    agg_arc_plot_save_path = (
        f"{args.output_dir}/{raw_data['gene_id']}_aggregate_junction_arc_plot.html"
    )
    agg_arc_plot.write_html(agg_arc_plot_save_path)
    print(f"Aggregate junction arc plot saved to: {agg_arc_plot_save_path}")
    if args.save_svg and STATIC_EXPORT_AVAILABLE:
        svg_save_path = (
            f"{args.output_dir}/{raw_data['gene_id']}_aggregate_junction_arc_plot.svg"
        )
        agg_arc_plot.write_image(svg_save_path, width=1600, height=1000)
        print(f"Aggregate junction arc plot SVG saved to: {svg_save_path}")
    if args.save_pdf and STATIC_EXPORT_AVAILABLE:
        pdf_save_path = (
            f"{args.output_dir}/{raw_data['gene_id']}_aggregate_junction_arc_plot.pdf"
        )
        agg_arc_plot.write_image(pdf_save_path, width=1600, height=1000)
        print(f"Aggregate junction arc plot PDF saved to: {pdf_save_path}")

    # Agg subset-focused junction plots
    if len(norm_data["subset_links"]) > 0:
        junction_figures = plot_all_subsets_junction_focus(
            aggregated_data,
            norm_data,
            show_individual_samples=True,
            include_territory_subsets=True,
        )
        junction_save_path = (
            f"{args.output_dir}/{raw_data['gene_id']}_all_subsets_junction_plots.html"
        )
        export_multipanel_with_navigation(
            junction_figures,
            [fig.layout.title.text for fig in junction_figures],
            junction_save_path,
        )
        print(f"All subset-focused junction plots saved to: {junction_save_path}")
        if args.save_svg and STATIC_EXPORT_AVAILABLE:
            export_figures_as_svg_archive(
                junction_figures,
                [fig.layout.title.text for fig in junction_figures],
                f"{args.output_dir}/{raw_data['gene_id']}_all_subsets_junction_plots.zip",
                raw_data["gene_id"],
            )
        if args.save_pdf and HAVE_PYPDF2 and STATIC_EXPORT_AVAILABLE:
            pdf_save_path = f"{args.output_dir}/{raw_data['gene_id']}_all_subsets_junction_plots.pdf"
            create_combined_pdf(
                junction_figures,
                [fig.layout.title.text for fig in junction_figures],
                pdf_save_path,
            )
    else:
        print(
            "No subsets defined for this gene; skipping subset-focused junction plots."
        )

    # Agg subset-focused territory plots
    if len(aggregated_data["territories_agg"]) > 1:
        territory_figures = plot_all_subsets_with_territories(
            aggregated_data, norm_data, show_individual_samples=True
        )
        territory_save_path = (
            f"{args.output_dir}/{raw_data['gene_id']}_all_subsets_territory_plots.html"
        )
        export_multipanel_with_navigation(
            territory_figures,
            [fig.layout.title.text for fig in territory_figures],
            territory_save_path,
        )
        print(f"All subset-focused territory plots saved to: {territory_save_path}")
        if args.save_svg and STATIC_EXPORT_AVAILABLE:
            export_figures_as_svg_archive(
                territory_figures,
                [fig.layout.title.text for fig in territory_figures],
                f"{args.output_dir}/{raw_data['gene_id']}_all_subsets_territory_plots.zip",
                raw_data["gene_id"],
            )
        if args.save_pdf and HAVE_PYPDF2 and STATIC_EXPORT_AVAILABLE:
            pdf_save_path = f"{args.output_dir}/{raw_data['gene_id']}_all_subsets_territory_plots.pdf"
            create_combined_pdf(
                territory_figures,
                [fig.layout.title.text for fig in territory_figures],
                pdf_save_path,
            )
    else:
        print("Not enough subsets with coverage data to create territory plots.")

    # 6. Export all data to Excel
    if args.export_excel:
        excel_save_path = f"{args.output_dir}/{raw_data['gene_id']}_analysis_data.xlsx"
        try:
            export_data_to_excel(aggregated_data, norm_data, excel_save_path)
        except ImportError as e:
            print(f"⚠ Excel export skipped: {e}")
            print(
                " ERROR: Excel export requested but required packages (openpyxl or xlsxwriter) are not installed."
            )
            print("  Install with: pip install openpyxl")

    process_related_genes_subprocess(raw_data, args, args.output_dir)

    con.close()
    print("\nAnalysis complete. Plots saved to:", args.output_dir)


if __name__ == "__main__":
    main()
