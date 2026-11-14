#!/usr/bin/env python

import argparse
import sqlite3
import pandas as pd
import numpy as np
import sys


def calculate_median_of_ratios(db_path):
    """
    Reads aggregated counts from the MAJEC database, calculates size factors
    using the median-of-ratios method, and writes them to a new table.
    """
    print(f"Connecting to database: {db_path}")
    con = sqlite3.connect(db_path)

    try:
        # --- Step 1: Fetch the aggregated counts matrix ---
        print("  -> Fetching aggregated counts matrix...")
        query = """
        SELECT
            act.group_id,
            sm.sample_id,
            act.count
        FROM
            aggregated_counts_tpm AS act
        JOIN
            sample_metadata AS sm ON act.sample_id_int = sm.sample_id_int;
        """
        df_long = pd.read_sql_query(query, con)

        # Pivot to a wide matrix: genes x samples
        counts_matrix = df_long.pivot(
            index="group_id", columns="sample_id", values="count"
        ).fillna(0)

        # Filter out genes that are zero across all samples
        counts_matrix = counts_matrix.loc[counts_matrix.sum(axis=1) > 0]
        print(
            f"  -> Matrix created with {counts_matrix.shape[0]} genes and {counts_matrix.shape[1]} samples."
        )

        # --- Step 2: Calculate the size factors (Median of Ratios) ---
        print("  -> Calculating size factors using median-of-ratios method...")

        # Create the pseudo-reference sample (geometric mean)
        # We add 1 to handle zeros before taking the log
        log_counts = np.log(counts_matrix + 1)
        log_geo_means = log_counts.mean(axis=1)

        # For each sample, calculate the ratio to the pseudo-reference
        ratios = counts_matrix.copy()
        for col in ratios.columns:
            # We must use non-log counts for the ratio
            ratios[col] = counts_matrix[col] / np.exp(log_geo_means)

        # The size factor is the median of these ratios (ignoring zeros)
        size_factors = ratios[ratios > 0].median(axis=0)

        print("  -> Calculated size factors:")
        print(size_factors)

        # --- Step 3: Write the size factors to a new table in the database ---
        print("  -> Writing size factors to the database...")
        cur = con.cursor()

        # Create the new table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sample_normalization (
                sample_id_int INTEGER PRIMARY KEY,
                size_factor REAL,
                FOREIGN KEY(sample_id_int) REFERENCES sample_metadata(sample_id_int)
            )
        """)

        # Prepare the data for insertion
        df_factors = size_factors.reset_index()
        df_factors.columns = ["sample_id", "size_factor"]

        # Get the integer IDs for the samples
        df_samples = pd.read_sql_query(
            "SELECT sample_id, sample_id_int FROM sample_metadata", con
        )
        df_factors = pd.merge(df_factors, df_samples, on="sample_id")

        # Use INSERT OR REPLACE to make the script safely re-runnable
        data_to_insert = [
            (row["sample_id_int"], row["size_factor"])
            for _, row in df_factors.iterrows()
        ]
        cur.executemany(
            "INSERT OR REPLACE INTO sample_normalization (sample_id_int, size_factor) VALUES (?, ?)",
            data_to_insert,
        )

        con.commit()
        print("SUCCESS: Normalization factors have been saved to the database.")

    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and add normalization size factors to a MAJEC database."
    )
    parser.add_argument(
        "--db", required=True, help="Path to the MAJEC SQLite database file."
    )
    args = parser.parse_args()
    calculate_median_of_ratios(args.db)


if __name__ == "__main__":
    main()
