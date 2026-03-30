import argparse
import sqlite3
import pandas as pd
import numpy as np
import sys
import os
from textwrap import dedent


# ==============================================================================
# 1. USER INPUT PARSING LAYER (No changes needed here)
# ==============================================================================
def sanitize_for_r(name):
    """Make names R-friendly."""
    import re

    # R variable names: start with letter or dot, contain letters/numbers/dots/underscores
    name = re.sub(r"[^a-zA-Z0-9_.]", "_", name)
    # Can't start with number
    if name[0].isdigit():
        name = "G_" + name
    # Reserved R words
    r_reserved = {
        "if",
        "else",
        "for",
        "while",
        "function",
        "in",
        "next",
        "break",
        "TRUE",
        "FALSE",
        "NULL",
        "NA",
        "NaN",
        "Inf",
    }
    if name in r_reserved:
        name = name + "_group"
    return name


def parse_criteria_string(criteria_string):
    criteria = []
    if not criteria_string:
        return criteria
    for part in criteria_string.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        key, values_str = part.split("=", 1)
        values = [v.strip() for v in values_str.split(",")]
        criteria.append({"column": key.strip(), "values": values})
    return criteria


def parse_criteria_file(criteria_file):
    try:
        with open(criteria_file, "r") as f:
            content = f.read().replace("\n", ";")
            return parse_criteria_string(content)
    except FileNotFoundError:
        print(f"ERROR: Criteria file not found at: {criteria_file}", file=sys.stderr)
        sys.exit(1)


def parse_comparison_definitions(comparison_file):
    """Parse using existing criteria parsing logic."""
    comparisons = []
    group_definitions = {}

    with open(comparison_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Split on semicolon+space to get the three parts
            parts = line.split("; ", 2)  # Limit to 3 splits
            if len(parts) != 3:
                continue

            comp_name, criteria1_str, criteria2_str = parts

            # Sanatize comp names as they will be col headers and might cause issues
            comp_name = sanitize_for_r(comp_name)

            # Use existing parser directly
            group1_name = f"{comp_name}_case"
            group2_name = f"{comp_name}_control"

            group_definitions[group1_name] = parse_criteria_string(criteria1_str)
            group_definitions[group2_name] = parse_criteria_string(criteria2_str)

            comparisons.append((comp_name, group1_name, group2_name))

    return comparisons, group_definitions


# ==============================================================================
# 2. DATABASE AND METADATA LAYER
# ==============================================================================


def get_sample_ids_from_criteria(db_conn, criteria):
    """Builds and executes a SQL query to get sample IDs based on criteria."""
    if not criteria:
        return []
    where_clauses, params = [], []
    for criterion in criteria:
        col, vals = criterion["column"], criterion["values"]
        placeholders = ",".join(["?"] * len(vals))
        where_clauses.append(f'"{col}" IN ({placeholders})')
        params.extend(vals)
    where_statement = " AND ".join(where_clauses)
    query = f"SELECT sample_id_int FROM sample_metadata WHERE {where_statement};"
    return pd.read_sql_query(query, db_conn, params=params)["sample_id_int"].tolist()


def fetch_count_matrix(db_conn, sample_ids_int, level="gene"):
    # Set up name fields for different data tables
    table_map = {
        "gene": {
            "tbl": "aggregated_counts_tpm",
            "feature_col": "group_id",
            "count_col": "count",
        },
        "transcript": {
            "tbl": "final_counts_tpm",
            "feature_col": "transcript_id_int",
            "count_col": "count",
        },
        "junction": {
            "tbl": "junction_counts",
            "feature_col": "junction_id",
            "count_col": "read_count",
        },  # <-- Corrected here
    }
    print(
        f"--- Fetching {level}-level count matrix for {len(sample_ids_int)} total samples ---"
    )

    config = table_map[level]
    table, feature_col, count_col = (
        config["tbl"],
        config["feature_col"],
        config["count_col"],
    )
    placeholders = ",".join(["?"] * len(sample_ids_int))
    query = f"""
    SELECT 
        t."{feature_col}", 
        sm.sample_id, 
        t."{count_col}" AS count
    FROM {table} AS t 
    JOIN sample_metadata sm ON t.sample_id_int = sm.sample_id_int 
    WHERE t.sample_id_int IN ({placeholders});
    """
    # query = f"""SELECT t."{feature_col}", t."{feature_col}", sm.sample_id, t."{count_col}" AS count FROM {table} AS t JOIN sample_metadata sm ON t.sample_id_int = sm.sample_id_int WHERE t.sample_id_int IN ({placeholders});"""
    df_long = pd.read_sql_query(query, db_conn, params=sample_ids_int)
    if df_long.empty:
        print("ERROR: No count data found for the selected samples.", file=sys.stderr)
        sys.exit(1)
    count_matrix = df_long.pivot(
        index=feature_col, columns="sample_id", values="count"
    ).fillna(0)
    if level in ["transcript", "junction"]:
        map_config = {
            "transcript": {
                "tbl": "transcript_id_map",
                "int": "transcript_id_int",
                "str": "transcript_id_str",
            },
            "junction": {
                "tbl": "junction_id_map",
                "int": "junction_id_int",
                "str": "junction_id_str",
            },
        }
        map_info = map_config[level]
        df_map = pd.read_sql_query(
            f"SELECT {map_info['int']}, {map_info['str']} FROM {map_info['tbl']}",
            db_conn,
        )
        count_matrix = (
            count_matrix.reset_index()
            .merge(df_map, left_on=feature_col, right_on=map_info["int"])
            .drop(columns=[feature_col, map_info["int"]])
            .set_index(map_info["str"])
        )
    return count_matrix.astype(int)


def fetch_confidence_metrics(
    db_conn, sample_ids_int, level="gene", filtered_features=None
):
    """Fetch confidence metrics aligned with count data."""

    if level == "gene":
        # Get ALL the group confidence metrics, not just weighted_confidence
        query = """
        SELECT 
            gc.group_id as gene_id,
            sm.sample_id,
            gc.weighted_confidence,
            gc.high_conf_fraction,
            gc.gini_coefficient,
            gc.effective_copies,
            gc.dominant_fraction,
            gc.has_junction_validation,
            gc.intra_group_competition,
            gc.inter_group_competition,
            gc.holistic_group_external_distinguishability,
            gc.strong_evidence_fraction,
            gc.n_expressed_transcripts,
            gc.holistic_group_distinguishability
        FROM group_confidence gc
        JOIN sample_metadata sm ON gc.sample_id_int = sm.sample_id_int
        WHERE gc.sample_id_int IN ({})
        """.format(",".join(["?"] * len(sample_ids_int)))

        # Add feature filter if provided
        if filtered_features:
            query += " AND gc.group_id IN ({})".format(
                ",".join(["?"] * len(filtered_features))
            )
            params = list(sample_ids_int) + list(filtered_features)
        else:
            params = sample_ids_int

    elif level == "transcript":
        # Use counts_with_confidence for transcript level
        print("  -> Fetching confidence and prior penalty metrics...")
        query = """
        SELECT 
            t.transcript_id_str,
            sm.sample_id,
            -- Confidence metrics
            cc.confidence_score,
            cc.shared_read_fraction,
            cc.distinguishability_score,
            cc.competition_type,
            cc.discord_score,
            cc.has_unique_junction_support,
            cc.overall_confidence,
            cc.strong_evidence_fraction,
            cc.ambiguous_fraction_distinguishability,
            -- Prior penalty metrics
            p.completeness_penalty,
            p.adjusted_subset_penalty
        FROM transcript_id_map t
        JOIN sample_metadata sm ON 1=1  -- Cartesian product for all sample/transcript combos
        LEFT JOIN counts_with_confidence cc 
            ON cc.transcript_id_int = t.transcript_id_int 
            AND cc.sample_id_int = sm.sample_id_int
        LEFT JOIN priors p 
            ON p.transcript_id_int = t.transcript_id_int 
            AND p.sample_id_int = sm.sample_id_int
        WHERE sm.sample_id_int IN ({})
        """.format(",".join(["?"] * len(sample_ids_int)))

        if filtered_features:
            query += " AND t.transcript_id_str IN ({})".format(
                ",".join(["?"] * len(filtered_features))
            )
            params = list(sample_ids_int) + list(filtered_features)
        else:
            params = sample_ids_int

    df_conf = pd.read_sql_query(query, db_conn, params=params)

    if df_conf.empty:
        print("  -> Warning: No confidence data found")
        return None, None

    if level == "gene":
        matrices = {}
        for metric in [
            "weighted_confidence",
            "gini_coefficient",
            "effective_copies",
            "holistic_group_external_distinguishability",
            "inter_group_competition",
            "strong_evidence_fraction",
            "holistic_group_distinguishability",
        ]:
            if metric in df_conf.columns:
                matrices[metric] = (
                    df_conf.pivot(index="gene_id", columns="sample_id", values=metric)
                    .apply(pd.to_numeric, errors="coerce")
                    .reindex(index=filtered_features, fill_value=np.nan)
                )
        core_key = "holistic_group_external_distinguishability"  # potential reads confoiunded with other genes is biggest sorce of uncertinty at the gene level
    else:
        matrices = {}
        for metric in [
            "confidence_score",
            "shared_read_fraction",
            "distinguishability_score",
            "discord_score",
            "strong_evidence_fraction",
            "completeness_penalty",
            "adjusted_subset_penalty",
            "ambiguous_fraction_distinguishability",
        ]:
            if metric in df_conf.columns:
                matrices[metric] = (
                    df_conf.pivot(
                        index="transcript_id_str", columns="sample_id", values=metric
                    )
                    .apply(pd.to_numeric, errors="coerce")
                    .reindex(index=filtered_features, fill_value=np.nan)
                )
        # 1. Get the "Certain" part of the confidence. This is the foundation.
        certain_confidence = matrices.get("strong_evidence_fraction").fillna(0.0)

        # 2. Get the "Uncertain" part.
        uncertain_fraction = matrices.get("shared_read_fraction").fillna(1.0)

        # 3. Get the reliability of the uncertain part.
        ambiguous_reliability = matrices.get("ambiguous_fraction_distinguishability")
        ambiguous_reliability.fillna(
            1.0, inplace=True
        )  # A unique-only transcript has perfect ambiguous reliability.

        # 4. The final confidence is the certain part + the uncertain part scaled by its reliability.
        final_confidence_matrix = certain_confidence + (
            uncertain_fraction * ambiguous_reliability
        )

        # Ensure the final score is capped at 1.0 in case of floating point inaccuracies
        final_confidence_matrix = final_confidence_matrix.clip(0, 1)

        # Store results
        matrices["final_confidence"] = final_confidence_matrix
        core_key = "final_confidence"

    return matrices[core_key], matrices  # Return primary + all metrics


def fetch_and_normalize_junctions_for_psi(
    db_conn,
    sample_ids_int,
    gene_ids=None,
    scale_factor=10000,
    min_count_per_sample=10,
    min_samples=3,
):
    """
    Fetches raw junction counts for an entire cohort, normalizes them by their
    parent gene's total junction usage on a per-sample basis, and returns an
    integer matrix ready for DESeq2 along with junction metadata.
    """
    print(f"--- Fetching and normalizing junction counts for Delta PSI analysis ---")
    min_total_count = min_count_per_sample * len(sample_ids_int)
    sample_placeholders = ",".join(["?"] * len(sample_ids_int))

    query_params = []
    query_params.extend(sample_ids_int)  # For CohortJunctionStats
    query_params.append(min_total_count)
    query_params.append(min_samples)

    if gene_ids:
        gene_placeholders = ",".join(["?"] * len(gene_ids))
        gene_filter_sql = f"AND gene_id IN ({gene_placeholders})"
        query_params.extend(gene_ids)
    else:
        gene_filter_sql = ""

    query_params.extend(sample_ids_int)  # For GeneJunctionTotals
    query_params.extend(sample_ids_int)  # For final SELECT

    query = f"""
    WITH CohortJunctionStats AS (
        SELECT 
            junction_id,
            SUM(read_count) as total_reads,
            COUNT(DISTINCT CASE WHEN read_count > 0 THEN sample_id_int END) as n_samples_detected
        FROM junction_counts
        WHERE sample_id_int IN ({sample_placeholders})
        GROUP BY junction_id
        HAVING total_reads >= ? AND n_samples_detected >= ?
    ),
    JunctionGeneContexts AS (
        -- Get junction-gene pairs with transcript counts
        SELECT 
            tj.junction_id_int,
            ta.gene_id,
            COUNT(DISTINCT tj.transcript_id_int) as transcript_count
        FROM transcript_junctions tj
        JOIN transcript_annotations ta ON tj.transcript_id_int = ta.transcript_id_int
        WHERE tj.junction_id_int IN (SELECT junction_id FROM CohortJunctionStats)
        {gene_filter_sql}
        GROUP BY tj.junction_id_int, ta.gene_id
    ),
    GeneJunctionTotals AS (
        SELECT
            jgc.gene_id,
            jc.sample_id_int,
            SUM(jc.read_count) AS total_gene_junction_reads
        FROM junction_counts jc
        JOIN JunctionGeneContexts jgc ON jc.junction_id = jgc.junction_id_int
        WHERE jc.sample_id_int IN ({sample_placeholders})
        GROUP BY jgc.gene_id, jc.sample_id_int
    ),
    SharedJunctionInfo AS (
        -- Get list of all genes each junction appears in
        SELECT 
            junction_id_int,
            GROUP_CONCAT(gene_id, ';') as all_genes,
            COUNT(DISTINCT gene_id) as n_gene_contexts
        FROM JunctionGeneContexts
        GROUP BY junction_id_int
    )
    SELECT
        jgc.gene_id || ':' || ja.junction_id_str AS unique_junction_id,
        jgc.gene_id,
        ja.junction_id_str,
        sm.sample_id,
        jc.read_count,
        gjt.total_gene_junction_reads,
        jgc.transcript_count,
        sji.n_gene_contexts,
        sji.all_genes
    FROM junction_counts jc
    JOIN JunctionGeneContexts jgc ON jc.junction_id = jgc.junction_id_int
    JOIN junction_annotations ja ON jc.junction_id = ja.junction_id_int
    JOIN sample_metadata sm ON jc.sample_id_int = sm.sample_id_int
    JOIN GeneJunctionTotals gjt ON jgc.gene_id = gjt.gene_id AND jc.sample_id_int = gjt.sample_id_int
    JOIN SharedJunctionInfo sji ON jc.junction_id = sji.junction_id_int
    WHERE jc.sample_id_int IN ({sample_placeholders})
    ORDER BY unique_junction_id, sm.sample_id
    """

    print("  -> Executing cohort-wide database query...")
    df_long = pd.read_sql_query(query, db_conn, params=query_params)

    if df_long.empty:
        print(
            "ERROR: No junction data found for the selected samples/genes.",
            file=sys.stderr,
        )
        return pd.DataFrame(), pd.DataFrame()

    # Create metadata before pivoting
    junction_metadata = df_long[
        [
            "unique_junction_id",
            "gene_id",
            "junction_id_str",
            "transcript_count",
            "n_gene_contexts",
            "all_genes",
        ]
    ].drop_duplicates()
    junction_metadata["is_multi_gene"] = junction_metadata["n_gene_contexts"] > 1
    junction_metadata = junction_metadata.set_index("unique_junction_id")

    # Report on multi-gene junctions
    multi_gene_count = junction_metadata["is_multi_gene"].sum()
    if multi_gene_count > 0:
        print(
            f"  -> Found {multi_gene_count} junction:gene contexts from junctions shared between multiple genes"
        )

    # Normalization and Pivoting
    print("  -> Normalizing counts and pivoting to matrix format...")

    df_long["normalized_count"] = (
        (df_long["read_count"] + 1)
        / (df_long["total_gene_junction_reads"] + 10)
        * scale_factor
    )

    psi_counts_matrix = (
        df_long.pivot(
            index="unique_junction_id", columns="sample_id", values="normalized_count"
        )
        .fillna(0)
        .round()
        .astype(int)
    )

    print(
        f"  -> Final matrix: {psi_counts_matrix.shape[0]} junction:gene contexts x {psi_counts_matrix.shape[1]} samples"
    )
    print(f"  -> Metadata saved for downstream filtering and interpretation")

    return psi_counts_matrix.astype(int), junction_metadata


# ==============================================================================
# 3. ANALYSIS & SCRIPTING LAYER
# ==============================================================================


def filter_low_expression(counts_df, min_cpm=1.0, min_samples=3):
    """
    Filters a count matrix to remove low-expression features.
    Keeps features with at least `min_cpm` in at least `min_samples` samples.
    """
    print(f"--- Pre-filtering low-expression features ---")
    print(f"    Original number of features: {counts_df.shape[0]:,}")

    lib_sizes_in_millions = counts_df.sum(axis=0) / 1_000_000
    cpm_df = counts_df.div(lib_sizes_in_millions, axis=1)

    keep_mask = (cpm_df >= min_cpm).sum(axis=1) >= min_samples
    filtered_counts_df = counts_df[keep_mask]

    print(f"    Number of features after filtering: {filtered_counts_df.shape[0]:,}")
    print(
        f"    (Kept features with >= {min_cpm} CPM in at least {min_samples} samples)"
    )
    print(
        f"    Removed {counts_df.shape[0] - filtered_counts_df.shape[0]:,} low-expression features"
    )
    print(f"    Median library size: {counts_df.sum(axis=0).median():,.0f}")
    print(
        f"    Library size range: {counts_df.sum(axis=0).min():,.0f} - {counts_df.sum(axis=0).max():,.0f}"
    )

    return filtered_counts_df


def create_variance_matrix_from_confidence(
    counts_matrix, confidence_matrix, level="gene", variance_base=1
):
    # Align matrices
    common_features = counts_matrix.index.intersection(confidence_matrix.index)
    common_samples = counts_matrix.columns.intersection(confidence_matrix.columns)

    counts_aligned = counts_matrix.loc[common_features, common_samples]
    conf_aligned = confidence_matrix.loc[common_features, common_samples]

    has_expression = counts_aligned > 0

    if level == "gene":

        # Create continuous variance multiplier based on inter-gene competition ratio (more potential for counts from other genes = more implicit variance):
        # A 0 here implies no ability to discriminate from a trancript of another gene. While a 1 means no competition from other genes.
        variance_inflation = variance_base ** (
            1 - conf_aligned
        )  # max 10--fold variance boost.

    else:
        # Use the combined transcript level metric to modify variance. Formula is the same but keep the level logic for potential future refinements.
        variance_inflation = variance_base ** (1 - conf_aligned)

    # Use sqrt of counts as base (Poisson-like)
    # This gives more realistic variance scaling
    base_variance = counts_aligned.copy()
    base_variance.loc[:, :] = 1

    # Final variance
    # Only inflate variance where there's expression AND low confidence
    variance_matrix = np.where(
        has_expression,
        base_variance * variance_inflation,  # Apply confidence adjustment
        base_variance,  # No adjustment for true zeros
    )
    return pd.DataFrame(
        variance_matrix, index=counts_aligned.index, columns=counts_aligned.columns
    )


def generate_deseq2_script_with_options(
    comparisons, output_prefix, confidence_mode="none", level="gene", batch_column=None
):
    """
    Generate R script with confidence integration for gene or transcript level.

    Args:
        comparisons: List of (name, case_group, control_group) tuples
        output_prefix: Prefix for output files
        confidence_mode: 'none', 'append', or 'variance'
        level: 'gene' or 'transcript'
    """

    r_script = f"""
    suppressPackageStartupMessages({{
        library("DESeq2")
        library(genefilter)
    }})

    cat("--- Loading Data ---\\n")
    cts <- as.matrix(read.table("{output_prefix}_counts_matrix.tsv", sep="\\t", 
                                row.names=1, header=TRUE, check.names=FALSE))
    coldata <- read.table("{output_prefix}_coldata.tsv", sep="\\t", 
                         row.names=1, header=TRUE, check.names=FALSE)
    """

    # Load confidence metrics if using them
    if confidence_mode in ["append", "variance"]:
        if level == "gene":
            r_script += f"""
        # Load key MAJEC gene-level metrics
        conf_strong_evidence <- as.matrix(read.table("{output_prefix}_strong_evidence_fraction_matrix.tsv", sep="\\t", row.names=1, header=TRUE, check.names=FALSE))
        conf_inter_comp <- as.matrix(read.table("{output_prefix}_holistic_group_external_distinguishability_matrix.tsv", sep="\\t", row.names=1, header=TRUE, check.names=FALSE))
        conf_gini <- as.matrix(read.table("{output_prefix}_gini_coefficient_matrix.tsv", sep="\\t", row.names=1, header=TRUE, check.names=FALSE))
        """
        elif level == "transcript":
            r_script += f"""
        # Load key MAJEC transcript-level metrics
        conf_strong_evidence <- as.matrix(read.table("{output_prefix}_strong_evidence_fraction_matrix.tsv", sep="\\t", row.names=1, header=TRUE, check.names=FALSE))
        conf_distinguish <- as.matrix(read.table("{output_prefix}_distinguishability_score_matrix.tsv", sep="\\t", row.names=1, header=TRUE, check.names=FALSE))
        conf_discord <- as.matrix(read.table("{output_prefix}_discord_score_matrix.tsv", sep="\\t", row.names=1, header=TRUE, check.names=FALSE))
        conf_final <- as.matrix(read.table("{output_prefix}_final_confidence_matrix.tsv", sep="\\t", row.names=1, header=TRUE, check.names=FALSE))
        conf_completeness <- as.matrix(read.table("{output_prefix}_completeness_penalty_matrix.tsv", sep="\\t", row.names=1, header=TRUE, check.names=FALSE))
        conf_subset <- as.matrix(read.table("{output_prefix}_adjusted_subset_penalty_matrix.tsv", sep="\\t", row.names=1, header=TRUE, check.names=FALSE))
        neg_log10_transform <- function(p, epsilon = 1e-6) {{
                p_safe <- pmax(p, epsilon)
                return(-log10(p_safe))
            }}
        completeness_log <- neg_log10_transform(conf_completeness)
        subset_log <- neg_log10_transform(conf_subset)
        """

    if confidence_mode == "variance":
        r_script += f"""
    # Load variance matrix for confidence weighting
    variance_matrix <- as.matrix(read.table("{output_prefix}_variance_matrix.tsv", 
                                            sep="\\t", row.names=1, header=TRUE, check.names=FALSE))
    """

    # Process each comparison
    for comp_name, _, _ in comparisons:
        col_name = f"comp_{comp_name}"
        if batch_column:
            design_formula = f"~ {batch_column} + {col_name}"
            # Ensure the batch column is also treated as a factor in R
            r_script += f"""
    coldata_subset${batch_column} <- as.factor(coldata_subset${batch_column})
    """
        else:
            design_formula = f"~ {col_name}"

        r_script += f"""
    
    # --- Comparison: {comp_name} ---
    cat("\\n--- Processing {comp_name} ---\\n")
    
    # Subset to case/control samples
    coldata_subset <- coldata[coldata${col_name} != "other", ]
    cts_subset <- cts[, rownames(coldata_subset)]
    coldata_subset${col_name} <- factor(coldata_subset${col_name}, levels=c("control", "case"))
 
    # Standard DESeq2
    dds <- DESeqDataSetFromMatrix(
        countData = round(cts_subset),
        colData = coldata_subset,
        design = {design_formula}
    )
    """
        # Create DESeq2 dataset
        if confidence_mode == "variance":
            r_script += f"""    
        
    cat("  -> Integrating MAJEC confidence by modulating dispersions.\\n")
    
    # 1. Run the standard first steps of DESeq2 to get baseline dispersion estimates
    dds <- estimateSizeFactors(dds)
    dds <- estimateDispersions(dds)
    
    # 2. Get the baseline dispersions calculated by DESeq2
    deseq_dispersions <- dispersions(dds)
    
    variance_subset <- variance_matrix[rownames(cts_subset), colnames(cts_subset), drop=FALSE]
    dispersion_multiplier <- rowMeans(variance_subset, na.rm = TRUE)
    
    # Create a boolean matrix indicating where counts are greater than zero.
    expression_mask <- cts_subset > 0
    
    # Replace multiplier values with NA where the transcript is not expressed.
    variance_masked <- variance_subset
    variance_masked[!expression_mask] <- NA
    
    # Now, calculate the rowMeans. na.rm=TRUE will correctly ignore the non-expressed samples.
    dispersion_multiplier <- rowMeans(variance_masked, na.rm = TRUE)
    
    # Handle cases where a transcript has zero counts across ALL samples in this comparison.
    # Its mean will be NaN. We should assign a neutral multiplier (1.0) to these.
    dispersion_multiplier[is.nan(dispersion_multiplier)] <- 1.0

     # --- SAFETY CHECK: Verify that the vectors have the same length ---
    if (length(deseq_dispersions) != length(dispersion_multiplier)) {{
        
        warning("CRITICAL WARNING: Mismatch in feature counts between DESeq2 dispersions and MAJEC variance multipliers.")
        warning(paste("  - DESeq2 dispersion vector length:", length(deseq_dispersions)))
        warning(paste("  - MAJEC multiplier vector length:", length(dispersion_multiplier)))
        warning("  - This can happen if DESeq2 internally filters some features.")
        warning("  - ABORTING variance modulation for this comparison. Proceeding with standard DESeq2 analysis.")
        
        # If the check fails, we skip modulation and run the standard DESeq test.
        # This prevents a crash or incorrect results.
        dds <- nbinomWaldTest(dds)
        
    }} else {{
        
        # --- If the check passes, proceed with your direct modulation ---
        cat("  -> Vector length check passed. Modulating dispersions.")
        
        # 3. Calculate the new dispersions by direct element-wise multiplication.
        modified_dispersions <- deseq_dispersions * dispersion_multiplier
        
        # 4. Replace the original dispersions.
        dispersions(dds) <- modified_dispersions

        # 5. Run the final statistical test without re-estimating dispersions.
        dds <- nbinomWaldTest(dds)
    }}
    # --- END SAFETY CHECK ---
    """
        else:
            r_script += f"""
    # Run differential expression analysis
    dds <- DESeq(dds)

    resLFC <- lfcShrink(dds, coef="{col_name}_case_vs_control", type="apeglm")
        """

        if confidence_mode in ["append", "variance"]:
            if level == "gene":
                r_script += f"""
        
        # Add key MAJEC gene-level metrics to results
        sample_cols <- rownames(coldata_subset)
        
        # Calculate mean metrics across samples in this comparison
        resLFC$MAJEC_strong_evidence_frac <- rowMeans(conf_strong_evidence[rownames(resLFC), sample_cols], na.rm=TRUE)
        resLFC$MAJEC_inter_gene_competition <- rowMeans(conf_inter_comp[rownames(resLFC), sample_cols], na.rm=TRUE)
        resLFC$MAJEC_gini <- rowMeans(conf_gini[rownames(resLFC), sample_cols], na.rm=TRUE)
        
        all_flags <- vector("list", nrow(resLFC))
    
        low_evidence_indices <- which(resLFC$padj < 0.05 & resLFC$MAJEC_strong_evidence_frac < 0.3)
        for (i in low_evidence_indices) {{ all_flags[[i]] <- c(all_flags[[i]], "LOW_EVIDENCE") }}
        
        high_ambiguity_indices <- which(resLFC$padj < 0.05 & resLFC$MAJEC_inter_gene_competition < 0.3)
        for (i in high_ambiguity_indices) {{ all_flags[[i]] <- c(all_flags[[i]], "HIGH_PARALOG_AMBIGUITY") }}
        
        dominated_indices <- which(resLFC$padj < 0.05 & resLFC$MAJEC_gini > 0.90)
        for (i in dominated_indices) {{ all_flags[[i]] <- c(all_flags[[i]], "DOMINATED_BY_ONE_ISOFORM") }}
        
        resLFC$MAJEC_flags <- sapply(all_flags, function(x) {{
        if (is.null(x)) {{
            return("PASS")
        }} else {{
            return(paste(x, collapse=";"))
        }}
        }})
            """
            elif level == "transcript":
                r_script += f"""
        
        # Add key MAJEC transcript-level metrics to results
        sample_cols <- rownames(coldata_subset)
        case_samples <- rownames(coldata_subset[coldata_subset${col_name} == 'case', ])
        control_samples <- rownames(coldata_subset[coldata_subset${col_name} == 'control', ])
        
        # Calculate mean metrics across samples in this comparison
        resLFC$MAJEC_final_confidence <- rowMeans(conf_final[rownames(resLFC), sample_cols], na.rm=TRUE)
        resLFC$MAJEC_strong_evidence_frac <- rowMeans(conf_strong_evidence[rownames(resLFC), sample_cols], na.rm=TRUE)
        resLFC$MAJEC_distinguish_score <- rowMeans(conf_distinguish[rownames(resLFC), sample_cols], na.rm=TRUE)
        resLFC$MAJEC_discord_score <- rowMeans(conf_discord[rownames(resLFC), sample_cols], na.rm=TRUE)

        group_factor <- factor(coldata_subset${col_name})
        completeness_log_subset <- completeness_log[, rownames(coldata_subset)]
        subset_log_subset <- subset_log[, rownames(coldata_subset)]
        
        ttest_results_completeness <- genefilter::rowttests(completeness_log_subset, group_factor)
        resLFC$MAJEC_completeness_penalty_case_mean <- rowMeans(completeness_log[rownames(resLFC), case_samples, drop=FALSE], na.rm=TRUE)
        resLFC$MAJEC_completeness_penalty_control_mean <- rowMeans(completeness_log[rownames(resLFC), control_samples, drop=FALSE], na.rm=TRUE)
        resLFC$MAJEC_completeness_pval <- ttest_results_completeness$p.value
        
        ttest_results_subset <- genefilter::rowttests(subset_log_subset, group_factor)
        resLFC$MAJEC_subset_penalty_case_mean <- rowMeans(subset_log[rownames(resLFC), case_samples, drop=FALSE], na.rm=TRUE)
        resLFC$MAJEC_subset_penalty_control_mean <- rowMeans(subset_log[rownames(resLFC), control_samples, drop=FALSE], na.rm=TRUE)
        resLFC$MAJEC_subset_pval <- ttest_results_subset$p.value


        # Create a more nuanced, semicolon-separated flagging system
        # Initialize an empty list to hold flags for each gene
        all_flags <- vector("list", nrow(resLFC))
        
        # Define conditions and add flags
        low_conf_indices <- which(resLFC$padj < 0.05 & resLFC$MAJEC_final_confidence < 0.3)
        for (i in low_conf_indices) {{ all_flags[[i]] <- c(all_flags[[i]], "LOW_FINAL_CONFIDENCE") }}
        
        low_evidence_indices <- which(resLFC$padj < 0.05 & resLFC$MAJEC_strong_evidence_frac < 0.3)
        for (i in low_evidence_indices) {{ all_flags[[i]] <- c(all_flags[[i]], "LOW_POSITIVE_EVIDENCE") }}
        
        poor_dist_indices <- which(resLFC$padj < 0.05 & resLFC$MAJEC_distinguish_score < 0.3)
        for (i in poor_dist_indices) {{ all_flags[[i]] <- c(all_flags[[i]], "POOR_DISTINGUISHABILITY") }}
        
        high_discord_indices <- which(resLFC$padj < 0.05 & resLFC$MAJEC_discord_score > 1.5)
        for (i in high_discord_indices) {{ all_flags[[i]] <- c(all_flags[[i]], "HIGH_EM_DISCORDANCE") }}
        
        # Add flags for significant penalty changes
        completeness_changed <- which(resLFC$padj < 0.05 & !is.na(resLFC$MAJEC_completeness_pval) & 
                                     resLFC$MAJEC_completeness_pval < 0.05)
        for (i in completeness_changed) {{ all_flags[[i]] <- c(all_flags[[i]], "COMPLETENESS_PENALTY_CHANGED") }}
        
        subset_changed <- which(resLFC$padj < 0.05 & !is.na(resLFC$MAJEC_subset_pval) & 
                               resLFC$MAJEC_subset_pval < 0.05)
        for (i in subset_changed) {{ all_flags[[i]] <- c(all_flags[[i]], "SUBSET_PENALTY_CHANGED") }}
        
        # Collapse the list of flags into a single character vector
        resLFC$MAJEC_flags <- sapply(all_flags, function(x) {{
            if (length(x) == 0) {{
                return("PASS")
            }} else {{
                return(paste(x, collapse=";"))
            }}
        }})
        """

        r_script += f"""
        
        # Save results with all metrics
        resLFC_df <- as.data.frame(resLFC)

        # Add the feature names (rownames) as the first column
        resLFC_df <- cbind(feature_id = rownames(resLFC_df), resLFC_df)

        # Sort the data frame by the 'pvalue' column
        resLFC_df <- resLFC_df[order(resLFC_df$pvalue), ]
        
        # Write the results to a CSV file
        write.csv(resLFC_df, "{output_prefix}_{comp_name}_DESeq2_results.csv", row.names=FALSE, quote=FALSE)
        
        # Print summary of confidence warnings
        if("MAJEC_warning" %in% colnames(resLFC_df)) {{
            warning_summary <- table(resLFC_df$MAJEC_warning[resLFC_df$padj < 0.05])
            cat("\\nConfidence warnings for significant genes in {comp_name}:\\n")
            print(warning_summary)
        }}
        """

    r_script += """
    
    cat("\\n--- All comparisons complete ---\\n")
    """

    return r_script


def build_comparison_columns(all_metadata, comparisons, group_to_sample_map):
    """
    Build separate metadata columns for each comparison.
    Each column has 'case', 'control', or 'other'.
    """
    for comp_name, case_group, control_group in comparisons:
        col_name = f"comp_{comp_name}"

        # Initialize all samples as 'other'
        all_metadata[col_name] = "other"

        # Mark case samples
        if case_group in group_to_sample_map:
            case_samples = group_to_sample_map[case_group]
            all_metadata.loc[
                all_metadata["sample_id_int"].isin(case_samples), col_name
            ] = "case"

        # Mark control samples
        if control_group in group_to_sample_map:
            control_samples = group_to_sample_map[control_group]
            all_metadata.loc[
                all_metadata["sample_id_int"].isin(control_samples), col_name
            ] = "control"

    return all_metadata


# ==============================================================================
# 4. MAIN EXECUTION ORCHESTRATOR
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a complete DESeq2 analysis package from a MAJEC database.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=dedent("""
        Workflow:
          1. Define a baseline 'cohort' of samples to include (optional).
          2. Provide a 'comparisons_file' to define the specific contrasts names and samples you want to compare.
          
        """),
    )
    parser.add_argument(
        "--db", required=True, help="Path to the MAJEC SQLite database."
    )
    cohort_group = parser.add_mutually_exclusive_group()
    cohort_group.add_argument(
        "--cohort_str",
        help="Define a baseline cohort of samples to include for statistical modeling.",
    )
    cohort_group.add_argument(
        "--cohort_file", help="Define a baseline cohort from a file."
    )
    parser.add_argument(
        "--comparisons_file",
        required=True,
        help="Path to a 3-column file defining contrasts (comparison_name;case_group;control_group).",
    )
    parser.add_argument(
        "--level",
        choices=["gene", "transcript", "junction", "delta_psi"],
        default="gene",
        help="Feature level for DE analysis. 'delta_psi' enables junction normalization.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to write all output files."
    )
    parser.add_argument(
        "--output_prefix",
        help="Optional prefix for all output files. (Default: the analysis level)",
    )
    parser.add_argument(
        "--confidence_mode",
        choices=["none", "append", "variance"],
        default="append",
        help="How to use MAJEC confidence: none, append metrics only, or use for variance. Only functions at gene or transcript level",
    )
    parser.add_argument(
        "--variance_base",
        help="A constant to determine how big an effect confidence has on the adjusted variance used in DESeq2\n"
        "Higher values increase implied variance for low confidence features. The param value is the maximum variance multiplier. Deafult = 5",
        default=5,
    )
    parser.add_argument(
        "--min_cpm", type=float, default=1.0, help="Minimum CPM for pre-filtering."
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=3,
        help="Minimum number of samples for CPM pre-filtering.",
    )
    parser.add_argument(
        "--junction_min_average_count",
        type=float,
        default=10,
        help="Minimum average junction count for a junctions inclusion in delta psi.",
    )
    parser.add_argument(
        "--junction_min_samples",
        type=int,
        default=3,
        help="Minimum number of samples with the junction observed for its inclusion in delta psi.",
    )
    parser.add_argument(
        "--min_gene_cpm",
        type=int,
        default=2,
        help="Minimum CPM for gene-based pre-filtering of junctions.",
    )
    parser.add_argument(
        "--min_gene_samples",
        type=int,
        default=3,
        help="Minimum number of samples above CPM threshold for gene-based pre-filtering of junctions.",
    )
    parser.add_argument(
        "--batch_column",
        help="Column name for batch effects (will add to DESeq2 design)",
    )

    args = parser.parse_args()
    if args.level in ["junction", "delta_psi"] and args.confidence_mode != "none":
        print(
            f"WARNING: Confidence metrics are not available for {args.level} level analysis."
        )
        print("         Setting --confidence_mode to 'none'")
        args.confidence_mode = "none"

    os.makedirs(args.output_dir, exist_ok=True)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    output_prefix = args.output_prefix or args.level

    # --- 1. Get baseline cohort IDs (optional) ---
    cohort_ids = set()
    if args.cohort_str or args.cohort_file:
        print("--- Defining Baseline Cohort ---")
        cohort_criteria = parse_criteria_string(args.cohort_str) or parse_criteria_file(
            args.cohort_file
        )
        cohort_ids.update(get_sample_ids_from_criteria(con, cohort_criteria))
        print(f"  -> Baseline cohort includes {len(cohort_ids)} samples.")

    # --- 2. Resolve Group Definitions to Sample IDs ---
    print("--- Resolving Comparison Groups ---")
    comparisons, group_definitions = parse_comparison_definitions(args.comparisons_file)

    # validate comparisons
    for name, case_group, control_group in comparisons:
        if case_group not in group_definitions:
            print(
                f"ERROR: Case group '{name}' not defined in groups file",
                file=sys.stderr,
            )
            sys.exit(1)
        if control_group not in group_definitions:
            print(
                f"ERROR: Control group '{name}' not defined in groups file",
                file=sys.stderr,
            )
            sys.exit(1)

    group_to_sample_map = {}
    for group_name, criteria in group_definitions.items():
        sample_ids = get_sample_ids_from_criteria(con, criteria)
        group_to_sample_map[group_name] = sample_ids
        print(f"    -> Group '{group_name}': {len(sample_ids)} samples")

    # Get all samples involved in any comparison
    all_comparison_samples = set()
    for _, case_group, control_group in comparisons:
        all_comparison_samples.update(group_to_sample_map.get(case_group, []))
        all_comparison_samples.update(group_to_sample_map.get(control_group, []))
    print(all_comparison_samples)

    # Combine with cohort if specified
    final_sample_ids = sorted(list(cohort_ids.union(all_comparison_samples)))

    # Fetch metadata and add comparison columns
    all_metadata = pd.read_sql_query(
        f"SELECT * FROM sample_metadata WHERE sample_id_int IN ({','.join(['?']*len(final_sample_ids))})",
        con,
        params=final_sample_ids,
    )

    # Build comparison-specific columns
    all_metadata = build_comparison_columns(
        all_metadata, comparisons, group_to_sample_map
    )
    # --- 4. Fetch Counts, Filter, and Generate Script ---
    if args.level == "delta_psi":
        # filter on gene expression
        # Fetch the aggregated gene counts for the entire cohort
        id_placeholders = ",".join(["?"] * len(final_sample_ids))
        gene_counts_query = f"""
            SELECT group_id, sm.sample_id, count
            FROM aggregated_counts_tpm
            JOIN sample_metadata sm ON aggregated_counts_tpm.sample_id_int = sm.sample_id_int
            WHERE aggregated_counts_tpm.sample_id_int IN ({id_placeholders});
        """
        df_gene_counts_long = pd.read_sql_query(
            gene_counts_query, con, params=final_sample_ids
        )
        # Pivot to a wide matrix for filtering
        gene_counts_matrix = df_gene_counts_long.pivot(
            index="group_id", columns="sample_id", values="count"
        ).fillna(0)

        # Use CPM filter function
        filtered_gene_matrix = filter_low_expression(
            gene_counts_matrix,
            min_cpm=args.min_gene_cpm,
            min_samples=args.min_gene_samples,
        )

        # This is our definitive list of genes to analyze
        genes_to_analyze = filtered_gene_matrix.index.tolist()
        print(f"  -> Found {len(genes_to_analyze):,} genes that passed the pre-filter.")
        # Use specialized function for this mode
        counts_matrix, junction_metadata = fetch_and_normalize_junctions_for_psi(
            con,
            final_sample_ids,
            gene_ids=genes_to_analyze,
            min_count_per_sample=args.junction_min_average_count,
            min_samples=args.junction_min_samples,
        )
        # Save junction metadata
        junction_metadata_path = os.path.join(
            args.output_dir, f"{output_prefix}_junction_metadata.tsv"
        )
        junction_metadata.to_csv(junction_metadata_path, sep="\t")

    else:
        # Use the general-purpose function for other levels
        counts_matrix = fetch_count_matrix(con, final_sample_ids, args.level)

    coldata = all_metadata.set_index("sample_id").loc[counts_matrix.columns]

    if args.level == "delta_psi":  # Skip CPM filtering for normalized junction data
        filtered_counts = counts_matrix
    else:
        filtered_counts = filter_low_expression(
            counts_matrix, args.min_cpm, args.min_samples
        )

    if args.confidence_mode != "none" and args.level in ["gene", "transcript"]:
        # Get the features that survived filtering
        filtered_features = filtered_counts.index.tolist()

        # Get the confidence matix
        confidence_matrix, all_metrics = fetch_confidence_metrics(
            con, final_sample_ids, args.level, filtered_features
        )

        # Save confidence matrix for R to use
        for metric_name, metric_matrix in all_metrics.items():
            metric_path = os.path.join(
                args.output_dir, f"{output_prefix}_{metric_name}_matrix.tsv"
            )
            metric_matrix.to_csv(metric_path, sep="\t")

        if args.confidence_mode == "variance":
            # Also save variance matrix
            variance_matrix = create_variance_matrix_from_confidence(
                filtered_counts,
                confidence_matrix,
                level=args.level,
                variance_base=args.variance_base,
            )
            var_path = os.path.join(
                args.output_dir, f"{output_prefix}_variance_matrix.tsv"
            )
            variance_matrix.to_csv(var_path, sep="\t")

    r_script = generate_deseq2_script_with_options(
        comparisons,
        output_prefix,
        confidence_mode=args.confidence_mode,
        level=args.level,
        batch_column=args.batch_column,
    )

    # --- 5. Write Files ---
    print(f"--- Writing analysis files to: {args.output_dir} ---")
    counts_path = os.path.join(args.output_dir, f"{output_prefix}_counts_matrix.tsv")
    coldata_path = os.path.join(args.output_dir, f"{output_prefix}_coldata.tsv")
    r_script_path = os.path.join(args.output_dir, f"{output_prefix}_run_deseq2.R")

    filtered_counts.to_csv(counts_path, sep="\t")
    print(f"    -> Count matrix written to: {os.path.basename(counts_path)}")
    coldata.to_csv(coldata_path, sep="\t")
    print(f"    -> ColData metadata written to: {os.path.basename(coldata_path)}")
    with open(r_script_path, "w") as f:
        f.write(r_script)
    print(f"    -> R script written to: {os.path.basename(r_script_path)}")

    con.close()

    print("\n-------------------------------------------------------------")
    print("DESeq2 analysis package created successfully.")
    print(f"To run: cd {args.output_dir} && Rscript {os.path.basename(r_script_path)}")
    print("-------------------------------------------------------------")


if __name__ == "__main__":
    main()
