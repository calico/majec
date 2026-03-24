#!/usr/bin/env python


import os
import sqlite3
import pandas as pd
import pickle
import gzip
import sys
import polars as pl
import subprocess
import argparse
import json
from textwrap import dedent


# --- SCHEMA DEFINITIONS (Constants) ---
# Schemas are defined centrally for clarity and easy modification.

PRIORS_SCHEMA = {
    "transcript_id_str": pl.Utf8,         # The first column from the file, which has no header
    "initial_count": pl.Float64,
    "junction_boost": pl.Float64,
    "raw_junction_evidence": pl.Float64,
    "junction_weight": pl.Float64,
    "post_junction_count": pl.Float64,
    "n_expected_junctions": pl.Int64,
    "n_observed_junctions": pl.Int64,
    "junction_observation_rate": pl.Float64,
    "completeness_penalty": pl.Float64,
    "completeness_model": pl.Utf8,
    "terminal_recovery_used": pl.Boolean,
    "RMS_used": pl.Boolean,
    "worst_junction_position": pl.Int64,
    "z_score": pl.Float64,
    "n_worst_junctions": pl.Int64,
    "worst_junction_positions": pl.Utf8,
    "worst_junction_counts": pl.Utf8,
    "worst_junction_z_scores": pl.Utf8,
    "recovery_point": pl.Float64,
    "worst_expected_vs_actual": pl.Utf8,
    "median_coverage": pl.Float64,
    "pure_median_1pct": pl.Boolean,
    "used_splice_competition": pl.Boolean,
    "n_competing_junctions": pl.Int64,
    "post_completeness_count": pl.Float64,
    "is_subset": pl.Boolean,
    "original_subset_penalty": pl.Float64,
    "subset_z_score": pl.Float64,
    "adjusted_subset_penalty": pl.Float64,
    "territory_confidence": pl.Float64,
    "territory_evidence_ratio": pl.Float64,
    "penalty_combination_strategy": pl.Utf8,
    "final_count": pl.Float64,
    "total_penalty": pl.Float64,
    "tsl_penalty": pl.Float64,
    "subset_penalty": pl.Float64,
    "total_prior_adjustment": pl.Float64,
    "had_junction_boost": pl.Boolean,
    "had_penalty": pl.Boolean,
    "penalty_types": pl.Utf8
}
COUNTS_WITH_CONFIDENCE_SCHEMA = {
    "transcript_id_str": pl.Utf8, 
    "count": pl.Float64,
    "entropy": pl.Float64,
    "confidence_score": pl.Float64,
    "n_sources": pl.Int64,
    "unique_fraction": pl.Float64,
    "normalized_entropy": pl.Float64,
    "total_count": pl.Float64,
    "shared_read_fraction": pl.Float64,
    "unique_multimapper_fraction": pl.Float64,
    "ambiguous_fraction_distinguishability": pl.Float64,
    "n_unique_competitors": pl.Float64,  
    "min_distinguishability": pl.Float64, 
    "n_competitive_classes": pl.Float64,     
    "hardest_competitor": pl.Utf8,
    "competition_type": pl.Utf8,
    "hardest_competitor_gene": pl.Utf8,
    'dominant_reads':pl.Float64, 
    "hardest_competitor_weight_frac":pl.Float64,
    "ambiguous_fraction_abs_inter_dist": pl.Float64,
    "intra_gene_competition_frac":pl.Float64,
    "inter_gene_competition_frac":pl.Float64,
    "junction_evidence_fraction": pl.Float64,
    "has_unique_junction_support": pl.Boolean,
    "junction_confidence_category": pl.Utf8,
    "discord_score": pl.Float64,
    "overall_confidence": pl.Utf8,
    "dominant_fraction" : pl.Float64,
    "strong_evidence_fraction" : pl.Float64,
    "distinguishability_score" : pl.Float64,
    "abs_inter_dist" : pl.Float64
}


FINAL_COUNTS_SCHEMA = [
    {'name': 'sample_id_int', 'dtype': 'INTEGER'}, {'name': 'transcript_id_int', 'dtype': 'INTEGER'},
    {'name': 'count', 'dtype': 'REAL'}, {'name': 'tpm', 'dtype': 'REAL'}
]

AGGREGATED_COUNTS_SCHEMA = [
    {'name': 'sample_id_int', 'dtype': 'INTEGER'}, {'name': 'group_id', 'dtype': 'TEXT'},
    {'name': 'count', 'dtype': 'REAL'}, {'name': 'tpm', 'dtype': 'REAL'}
]

DISTINGUISHABILITY_SCHEMA = [
    {'name': 'sample_id_int', 'dtype': 'INTEGER'}, {'name': 'transcript_id_int', 'dtype': 'INTEGER'},
    {'name': 'distinguishability_score', 'dtype': 'REAL'}, {'name': 'min_distinguishability', 'dtype': 'REAL'},
    {'name': 'n_competitive_classes', 'dtype': 'INTEGER'}, {'name': 'n_unique_competitors', 'dtype': 'INTEGER'},
    {'name': 'hardest_competitor_int', 'dtype': 'INTEGER'}, {'name': 'hardest_competitor_gene', 'dtype': 'TEXT'},
    {'name': 'competition_type', 'dtype': 'TEXT'}, {'name': 'my_gene', 'dtype': 'TEXT'},
    {'name': 'transcript_type', 'dtype': 'TEXT'}, {'name': 'count', 'dtype': 'REAL'}
]

GROUP_CONFIDENCE_SCHEMA = [{'name': 'sample_id_int', 'dtype': 'INTEGER'},
    {'name': 'group_id', 'dtype': 'TEXT'},
    {'name': 'aggregated_count', 'dtype': 'REAL'},
    {'name': 'n_expressed_transcripts', 'dtype': 'INTEGER'},
    {'name': 'n_total_transcripts', 'dtype': 'INTEGER'},
    {'name': 'weighted_confidence', 'dtype': 'REAL'},
    {'name': 'weighted_unique_fraction', 'dtype': 'REAL'},
    {'name': 'weighted_dominant_reads_fraction', 'dtype': 'REAL'},
    {'name': 'weighted_shared_fraction', 'dtype': 'REAL'},
    {'name': 'ambiguous_fraction_group_distinguishability', 'dtype': 'REAL'},
    {'name': 'holistic_group_distinguishability', 'dtype': 'REAL'},
    {'name': 'holistic_group_external_distinguishability', 'dtype': 'REAL'},
    {'name': 'ambiguous_fraction_external_distinguishability', 'dtype': 'REAL'},
    {'name': 'high_conf_fraction', 'dtype': 'REAL'},
    {'name': 'gini_coefficient', 'dtype': 'REAL'},
    {'name': 'effective_copies', 'dtype': 'REAL'},
    {'name': 'dominant_transcript', 'dtype': 'TEXT'},
    {'name': 'strong_evidence_fraction', 'dtype': 'REAL'},
    {'name': 'dominant_fraction', 'dtype': 'REAL'},
    {'name': 'has_junction_validation', 'dtype': 'INTEGER'},
    {'name': 'group_type', 'dtype': 'TEXT'},
    {'name': 'intra_group_competition', 'dtype': 'REAL'},
    {'name': 'inter_group_competition', 'dtype': 'REAL'},
    {'name': 'main_external_competitor', 'dtype': 'TEXT'},
    {'name': 'is_dominated', 'dtype': 'TEXT'}]


# --- DATABASE FUNCTIONS ---

def connect_db(db_path, force=False):
    """Connects to the SQLite database, handling overwriting."""
    if os.path.exists(db_path) and not force:
        print(f"WARNNING: Database '{db_path}' already exists. Use --force to overwrite.", file=sys.stderr)
        #sys.exit(1)
    elif os.path.exists(db_path) and force:
        print(f"WARNING: Overwriting existing database '{db_path}'.")
        os.remove(db_path)

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    print("Setting PRAGMA for faster bulk loading...")
    cur.execute("PRAGMA synchronous = OFF")
    cur.execute("PRAGMA journal_mode = MEMORY")
    cur.execute("PRAGMA cache_size = 100000") # Increased cache size
    return con, cur

def create_schema(cur):
    """Creates all tables and schemas in the database."""
    print("Creating database schema...")
    
    # --- Main Data Tables ---
    def create_table_from_schema(table_name, schema):
        columns_sql = [f"{col['name']} {col['dtype']}" for col in schema]
        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns_sql)})"
        cur.execute(create_sql)

    create_table_from_schema('group_confidence', GROUP_CONFIDENCE_SCHEMA)
    
    # --- Metadata and Junction Tables ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sample_metadata (
            sample_id_int INTEGER PRIMARY KEY, sample_id TEXT UNIQUE,
            cell_line TEXT, genotype TEXT, compound TEXT, rep TEXT, sID TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transcript_junction_summary (
            sample_id_int INTEGER, transcript_id_int INTEGER, total_junction_reads REAL,
            unique_junction_reads REAL, shared_junction_reads REAL,
            n_unique_junctions INTEGER, n_shared_junctions INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS junction_counts (
            sample_id_int INTEGER,
            junction_id INTEGER,
            read_count REAL,
            PRIMARY KEY (sample_id_int, junction_id) -- New, simpler key
        )
    """)
    
    # --- Annotation Mapping Tables (for a self-contained DB) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transcript_id_map (
            transcript_id_int INTEGER PRIMARY KEY, transcript_id_str TEXT UNIQUE
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS junction_id_map (
            junction_id_int INTEGER PRIMARY KEY, junction_id_str TEXT UNIQUE
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transcript_annotations (
            transcript_id_int INTEGER PRIMARY KEY,
            transcript_id_str TEXT UNIQUE,
            gene_id TEXT,
            chr TEXT,
            strand TEXT,
            start INTEGER,
            end INTEGER,
            transcript_length INTEGER,
            n_exons INTEGER,
            feature_type TEXT,
            tsl TEXT,
            is_subset INTEGER,
            n_supersets INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS subset_supersets (
            subset_transcript_id_int INTEGER,
            superset_transcript_id_int INTEGER,
            FOREIGN KEY(subset_transcript_id_int) REFERENCES transcript_annotations(transcript_id_int),
            FOREIGN KEY(superset_transcript_id_int) REFERENCES transcript_annotations(transcript_id_int)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transcript_junctions (
            transcript_id_int INTEGER,
            junction_id_int INTEGER,
            junction_order INTEGER,
            PRIMARY KEY (transcript_id_int, junction_id_int)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS junction_annotations (
            junction_id_int INTEGER PRIMARY KEY,
            junction_id_str TEXT UNIQUE,
            n_transcripts_sharing INTEGER,
            is_unique INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS territory_regions (
            region_id INTEGER PRIMARY KEY,
            chr TEXT,
            start INTEGER,
            end INTEGER,
            strand TEXT,
            length INTEGER,
            region_type TEXT  -- e.g., 'unique' or 'comparator'
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transcript_territories (
            transcript_id_int INTEGER,
            region_id INTEGER,
            territory_role TEXT, -- 'unique' or 'comparator' for this transcript
            PRIMARY KEY (transcript_id_int, region_id),
            FOREIGN KEY(transcript_id_int) REFERENCES transcript_annotations(transcript_id_int),
            FOREIGN KEY(region_id) REFERENCES territory_regions(region_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS territory_coverage (
            sample_id_int INTEGER,
            region_id INTEGER,
            mean_coverage REAL,
            PRIMARY KEY (sample_id_int, region_id),
            FOREIGN KEY(sample_id_int) REFERENCES sample_metadata(sample_id_int),
            FOREIGN KEY(region_id) REFERENCES territory_regions(region_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS final_counts_tpm (
            sample_id_int INTEGER,
            transcript_id_int INTEGER,
            count REAL,
            tpm REAL,
            PRIMARY KEY (sample_id_int, transcript_id_int)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS aggregated_counts_tpm (
            sample_id_int INTEGER,
            group_id TEXT,
            count REAL,
            tpm REAL,
            PRIMARY KEY (sample_id_int, group_id)
        )
    """)
    
    print("Schema creation complete.")

def extract_manifest_metadata(con, cur, sample_names_from_manifest):
    """
    falls back to using the sample names directly from the run manifest.
    """
    print("creating minimal sample metadata from manifest. no downstream statistics will be enabled ...")
    df = pd.DataFrame({'sample_id': sample_names_from_manifest})

    # Add integer IDs and insert into the DB
    df['sample_id_int'] = df.index
    for col in ['cell_line', 'genotype', 'compound', 'rep', 'sID']:
        if col not in df.columns:
            df[col] = None # Add placeholder columns if they don't exist
    
    # Useing 'replace' to ensure the table is clean before inserting
    df.to_sql('sample_metadata', con, if_exists='replace', index=False)
    con.commit()
    
    sample_string_to_int = dict(zip(df['sample_id'], df['sample_id_int']))
    print(f"  Ingested metadata for {len(sample_string_to_int)} samples.")
    return sample_string_to_int

    
def merge_manifests(manifest_paths):
    """
    Merge multiple manifests from chunked runs, validating compatibility.
    Returns a unified manifest structure with lists for summary files.
    """
    print(f"Merging {len(manifest_paths)} manifest files...")
    
    merged = {
        'annotation_file': None,
        'cache_files': {},
        'per_sample_files': {},
        'summary_files': {
            'transcript_counts': [],
            'transcript_tpm': [],
            'aggregated_counts': [],
            'aggregated_tpm': [],
            'group_confidence': []
        },
        'source_manifests': manifest_paths  # Track provenance
    }
    
    for idx, manifest_path in enumerate(manifest_paths):
        print(f"  Processing manifest {idx+1}/{len(manifest_paths)}: {manifest_path}")
        
        if not os.path.exists(manifest_path):
            print(f"ERROR: Manifest file not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)
            
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Validate annotation consistency
        if merged['annotation_file'] is None:
            merged['annotation_file'] = manifest['annotation_file']
            print(f"    Using annotation: {merged['annotation_file']}")
        elif merged['annotation_file'] != manifest['annotation_file']:
            print(f"ERROR: Annotation file mismatch!", file=sys.stderr)
            print(f"  Expected: {merged['annotation_file']}", file=sys.stderr)
            print(f"  Found:    {manifest['annotation_file']}", file=sys.stderr)
            sys.exit(1)
        
        # Check for sample overlaps in cache files
        overlap_cache = set(merged['cache_files'].keys()) & set(manifest.get('cache_files', {}).keys())
        if overlap_cache:
            print(f"ERROR: Sample overlap detected in cache files: {overlap_cache}", file=sys.stderr)
            sys.exit(1)
            
        # Check for sample overlaps in per-sample files
        overlap_per_sample = set(merged['per_sample_files'].keys()) & set(manifest.get('per_sample_files', {}).keys())
        if overlap_per_sample:
            print(f"ERROR: Sample overlap detected in per-sample files: {overlap_per_sample}", file=sys.stderr)
            sys.exit(1)
        
        # Merge the dictionaries
        merged['cache_files'].update(manifest.get('cache_files', {}))
        merged['per_sample_files'].update(manifest.get('per_sample_files', {}))
        
        # Collect summary files
        for key in merged['summary_files']:
            if key in manifest.get('summary_files', {}):
                file_path = manifest['summary_files'][key]
                if os.path.exists(file_path):
                    merged['summary_files'][key].append(file_path)
                else:
                    print(f"    WARNING: Summary file not found: {file_path}")
        
        print(f"    Added {len(manifest.get('cache_files', {}))} samples from this manifest")
    
    # Validate we have at least some data
    total_samples = len(merged['cache_files'])
    if total_samples == 0:
        print("ERROR: No samples found across all manifests!", file=sys.stderr)
        sys.exit(1)
        
    print(f"Merge complete: {total_samples} total samples from {len(manifest_paths)} manifests")
    return merged


def parse_and_ingest_metadata(db_path, metadata_file, sample_names_from_manifest):
    """
    Parses the special two-header metadata file, validates it, and ingests
    it into the MAJEC database with an "all text" approach.
    """
    print(f"--- Ingesting metadata from: {metadata_file} ---")

    # --- Step 1: Read the special header lines to get the variable types ---
    experimental_vars = []
    id_vars = []
    try:
        with open(metadata_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#experimental_variables:'):
                    experimental_vars = [v.strip() for v in line.split(':', 1)[1].split(',')]
                elif line.startswith('#id_variables:'):
                    id_vars = [v.strip() for v in line.split(':', 1)[1].split(',')]
                # Stop reading after we've found our headers or hit the data
                if not line.startswith('#'):
                    break
    except FileNotFoundError:
        print(f"ERROR: Metadata file not found at '{metadata_file}'", file=sys.stderr)
        return False

    if not experimental_vars:
        print("WARNING: No '#experimental_variables:' line found in metadata file. Cannot perform statistical comparisons.", file=sys.stderr)
        # We can still proceed, but the user should be warned.

    # --- Step 2: Read the main data table using pandas ---
    try:
        # Use comment='#' to automatically ignore the special headers and any other comments.
        # Use dtype=str to ingest everything as text for robustness.
        df_meta = pd.read_csv(metadata_file, sep='\t', comment='#', dtype=str)
    except Exception as e:
        print(f"ERROR: Failed to parse the data table in '{metadata_file}': {e}", file=sys.stderr)
        return False
        
    # --- Step 3: Validate the parsed data ---

    samples_in_manifest = set(sample_names_from_manifest)
    samples_in_metadata = set(df_meta['sample_id'])

    # Sample check 1: Are there samples in the manifest that are MISSING from the metadata?
    missing_from_metadata = samples_in_manifest - samples_in_metadata
    if missing_from_metadata:
        print("\nFATAL ERROR: The following samples were found in the MAJEC run but are MISSING from your metadata file:", file=sys.stderr)
        for sample in sorted(list(missing_from_metadata)):
            print(f"  - {sample}", file=sys.stderr)
        sys.exit(1)
        
    # Sample check 2: Are there extra samples in the metadata that were NOT in the run?
    extra_in_metadata = samples_in_metadata - samples_in_manifest
    if extra_in_metadata:
        print("\n  WARNING: The following samples are in your metadata file but were NOT found in the MAJEC run manifest. They will be ignored:")
        for sample in sorted(list(extra_in_metadata)):
            print(f"    - {sample}")
        
        # Filter the DataFrame to only include samples that are actually in the run
        df_meta = df_meta[df_meta['sample_id'].isin(samples_in_manifest)]
    
    print("  -> Validation successful: All run samples are present in the metadata file.")

    print(f"  -> Found {len(df_meta)} samples and {len(df_meta.columns)} columns.")
    
    # Check that the first column has the correct name
    if df_meta.columns[0] != 'sample_id':
        print(f"ERROR: The first column in the metadata file must be named 'sample_id'. Found '{df_meta.columns[0]}' instead.", file=sys.stderr)
        return False
        
    # Check that all declared variables actually exist as columns
    all_declared_vars = set(experimental_vars + id_vars)
    all_data_columns = set(df_meta.columns[1:]) # All columns except the first ID column
    if not all_declared_vars.issubset(all_data_columns):
        missing_cols = all_declared_vars - all_data_columns
        print(f"ERROR: The following variables were declared in the header but are missing as columns in the data: {missing_cols}", file=sys.stderr)
        return False

    # --- Step 4: Ingest into the database ---
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    try:
        # Write the main metadata table
        df_meta['sample_id_int'] = df_meta.index # Add the integer ID
        df_meta.to_sql('sample_metadata', con, if_exists='replace', index=False)

        # Create and write the new metadata_schema table
        schema_records = []
        for var in experimental_vars:
            schema_records.append({'field_name': var, 'field_type': 'experimental'})
        for var in id_vars:
            schema_records.append({'field_name': var, 'field_type': 'id'})
        
        if schema_records:
            df_schema = pd.DataFrame(schema_records)
            df_schema.to_sql('metadata_schema', con, if_exists='replace', index=False)
            
        con.commit()
        print("  -> Successfully ingested metadata and schema into the database.")
        
    except Exception as e:
        print(f"ERROR: Failed to write metadata to the database: {e}", file=sys.stderr)
        con.rollback()
        return False
    finally:
        con.close()
    sample_string_to_int = dict(zip(df_meta['sample_id'], df_meta['sample_id_int']))
    return sample_string_to_int

def ingest_annotation_data(con, cur, anno_data):
    """Ingests all annotation context into dedicated tables."""
    print("--- Ingesting annotation context data ---")
    

     # Check if annotation data already loaded
    cur.execute("SELECT COUNT(*) FROM transcript_annotations")
    if cur.fetchone()[0] > 0:
        print("  Annotation data already exists. Skipping.")
        return
   
        
    # 1. Ingest Transcript Annotations
    print("  Processing transcript_info...")
    # Convert the nested dict to a list of dicts for DataFrame creation
    records = []
    for tid_int, info in anno_data['transcript_info'].items():
        record = info.copy()
        record['transcript_id_int'] = tid_int
        record['transcript_id_str'] = anno_data['int_to_string'][tid_int]
        # Add subset info
        subset_info = anno_data['subset_relationships'].get(tid_int, {})
        record['is_subset'] = 1 if tid_int in anno_data['subset_relationships'] else 0
        record['n_supersets'] = len(subset_info.get('supersets', []))
        records.append(record)
    
    df_transcripts = pd.DataFrame(records)
    df_transcripts.to_sql('transcript_annotations', con, if_exists='replace', index=False)
    del df_transcripts

    # 2. Ingest Subset-Superset Relationships
    print("  Processing subset_relationships...")
    subset_links = []
    for subset_id, info in anno_data['subset_relationships'].items():
        for superset_id in info.get('supersets', []):
            subset_links.append((subset_id, superset_id))
    
    if subset_links:
        cur.executemany("INSERT INTO subset_supersets VALUES (?, ?)", subset_links)

    # 3. Ingest Transcript-Junction Mappings
    print("  Processing transcript_junction_order...")
    transcript_junction_links = []
    for tid_int, junction_list in anno_data['transcript_junction_order'].items():
        for i, jid_int in enumerate(junction_list):
            transcript_junction_links.append((tid_int, jid_int, i))
    
    if transcript_junction_links:
        cur.executemany("INSERT INTO transcript_junctions VALUES (?, ?, ?)", transcript_junction_links)

    # 4. Ingest Junction Annotations
    print("  Processing junction_map...")
    junction_records = []
    for jid_int, info in anno_data['junction_map'].items():
        junction_records.append({
            'junction_id_int': jid_int,
            'junction_id_str': anno_data['junction_int_to_string'][jid_int],
            'n_transcripts_sharing': info.get('n_transcripts', 0),
            'is_unique': 1 if info.get('is_unique', False) else 0
        })
    
    if junction_records:
        df_junctions = pd.DataFrame(junction_records)
        df_junctions.to_sql('junction_annotations', con, if_exists='replace', index=False)

    # 5. Ingest Transcript ID Map (String <-> Int)
    string_to_int_map = anno_data.get('string_to_int')
    if string_to_int_map:
        df_transcript_map = pd.DataFrame(
            list(string_to_int_map.items()),
            columns=['transcript_id_str', 'transcript_id_int']
        )
        df_transcript_map.to_sql('transcript_id_map', con, if_exists='replace', index=False)
    
    # 6. Junction Mappings
    junction_string_to_int_map = anno_data.get('junction_string_to_int')
    if junction_string_to_int_map:
        df_junction_map = pd.DataFrame(
            list(junction_string_to_int_map.items()),
            columns=['junction_id_str', 'junction_id_int']
        )
        df_junction_map.to_sql('junction_id_map', con, if_exists='replace', index=False)

    con.commit()
    print("Annotation context ingestion complete.")

def ingest_territory_data(con, cur, anno_data):
    """Ingests subset rescue territory data into dedicated tables."""
    print("--- Ingesting subset territory data ---")
    
     # Check if annotation data already loaded
    cur.execute("SELECT COUNT(*) FROM transcript_territories")
    if cur.fetchone()[0] > 0:
        print("  territories data already exists. Skipping.")
        return
    
    territory_mapping = anno_data.get('subset_coverage_territory_mapping')
    if not territory_mapping or 'regions' not in territory_mapping:
        print("  INFO: No subset territory data found in annotation file. Skipping.")
        return

    # 1. Ingest the master list of regions
    print(f"  Processing {len(territory_mapping['regions'])} territory regions...")
    region_records = []
    for region_id, info in territory_mapping['regions'].items():
        region_records.append({
            'region_id': region_id,
            'chr': info['chr'],
            'start': info['start'],
            'end': info['end'],
            'strand': info['strand'],
            'length': info['length'],
            'region_type': info['type']
        })
    
    if region_records:
        df_regions = pd.DataFrame(region_records)
        df_regions.to_sql('territory_regions', con, if_exists='replace', index=False)

    # 2. Ingest the links between transcripts and regions
    print(f"  Processing links for {len(territory_mapping['subset_territories'])} transcripts...")
    link_records = []
    for transcript_id, territories in territory_mapping['subset_territories'].items():
        tid_int = int(transcript_id)
        
        for region_id in territories.get('unique', []):
            link_records.append((tid_int, int(region_id), 'unique'))
            
        for region_id in territories.get('comparator', []):
            link_records.append((tid_int, int(region_id), 'comparator'))
            
    if link_records:
        cur.executemany("INSERT INTO transcript_territories VALUES (?, ?, ?)", link_records)
        
    con.commit()
    print("Subset territory data ingestion complete.")

def ingest_matrix_data_from_multiple_files(con, cur, table_name, 
                                          counts_files, tpm_files, 
                                          sample_map, transcript_map, 
                                          db_path, is_aggregated=False):
    """
    Process multiple matrix file pairs sequentially.
    Each file pair adds additional columns (samples) to the database.
    """
    if not counts_files or not tpm_files:
        print(f"  INFO: No matrix files for '{table_name}'. Skipping.")
        return
    
    # Check for exisiting data
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    if cur.fetchone()[0] > 0:
        print(f"  Table '{table_name}' already contains data. Skipping ingestion.")
        return
        
    if len(counts_files) != len(tpm_files):
        print(f"ERROR: Mismatch in number of counts ({len(counts_files)}) and TPM ({len(tpm_files)}) files", file=sys.stderr)
        sys.exit(1)
    
    print(f"--- Ingesting {len(counts_files)} matrix file pairs for table: {table_name} ---")
    
    transcript_lookup = None
    if not is_aggregated:
        transcript_lookup = pl.DataFrame({
                'transcript_id_str': list(transcript_map.keys()),
                'transcript_id_int': list(transcript_map.values())
            })
        
    for idx, (counts_file, tpm_file) in enumerate(zip(counts_files, tpm_files)):
        print(f"  Processing matrix pair {idx+1}/{len(counts_files)}:")
        print(f"    Counts: {os.path.basename(counts_file)}")
        print(f"    TPM:    {os.path.basename(tpm_file)}")
        
        ingest_matrix_data_by_column_batches(
            cur, table_name, counts_file, tpm_file, sample_map, 
            db_path, is_aggregated, skip_if_exists=False, transcript_lookup=transcript_lookup
        )

def ingest_matrix_data_by_column_batches(cur, table_name, counts_file, tpm_file, sample_map, db_path, is_aggregated=False, skip_if_exists=True, transcript_lookup=None):
    """
    Robustly ingests wide matrices by independently parsing headers for counts and TPM files,
    validating sample consistency, and standardizing column names before processing.
    """
    print(f"--- Ingesting matrix data for table: {table_name} ---")
    
    if skip_if_exists:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        if cur.fetchone()[0] > 0:
            print(f"  Table '{table_name}' already contains data. Skipping ingestion.")
            return

    # Get headers from both files
    counts_header = pl.read_csv(counts_file, separator='\t', n_rows=0).columns
    tpm_header = pl.read_csv(tpm_file, separator='\t', n_rows=0).columns
    
    counts_id_col = counts_header[0]
    tpm_id_col = tpm_header[0]
    
    # Get the sample columns from each file
    sample_columns_counts = counts_header[1:]
    sample_columns_tpm = tpm_header[1:]

    # CRITICAL VALIDATION: Ensure the sample columns are identical.
    if set(sample_columns_counts) != set(sample_columns_tpm):
        print("FATAL ERROR: Sample columns in counts and TPM files do not match.", file=sys.stderr)
        sys.exit(1)
    
    # Use the counts samples as the authority for order
    sample_columns = sample_columns_counts
    
    column_batch_size = 12


    
    for i in range(0, len(sample_columns), column_batch_size):
        batch_samples = sample_columns[i:i+column_batch_size]
        print(f"  Processing samples {i+1} to {min(i+column_batch_size, len(sample_columns))} of {len(sample_columns)}...")
        
           # Read with correct column names for each file
        counts_cols_to_read = [counts_id_col] + batch_samples
        tpm_cols_to_read = [tpm_id_col] + batch_samples

        df_counts_batch = pl.read_csv(counts_file, separator='\t', columns=counts_cols_to_read)
        df_tpm_batch = pl.read_csv(tpm_file, separator='\t', columns=tpm_cols_to_read)
    
        
        if is_aggregated:
            df_counts_batch = df_counts_batch.select([pl.col(df_counts_batch.columns[0]).alias('group_id'),
            *[pl.col(c) for c in df_counts_batch.columns[1:]]
             ])
            df_tpm_batch = df_tpm_batch.select([pl.col(df_tpm_batch.columns[0]).alias('group_id'),
            *[pl.col(c) for c in df_tpm_batch.columns[1:]]
            ])
            id_col='group_id'
            columns_to_write = ['sample_id_int', 'group_id', 'count', 'tpm']
        else:
            df_counts_batch = df_counts_batch.rename({df_counts_batch.columns[0]: 'transcript_id_str'})
            df_counts_batch = df_counts_batch.join(transcript_lookup, on='transcript_id_str', how='left').drop('transcript_id_str')
            
            df_tpm_batch = df_tpm_batch.rename({df_tpm_batch.columns[0]: 'transcript_id_str'})
            df_tpm_batch = df_tpm_batch.join(transcript_lookup, on='transcript_id_str', how='left').drop('transcript_id_str')

            id_col='transcript_id_int'
            columns_to_write = ['sample_id_int', 'transcript_id_int', 'count', 'tpm']

        # Ensure numeric columns are properly typed    
        for col in batch_samples:
            df_counts_batch = df_counts_batch.with_columns(
                pl.col(col).cast(pl.Float64, strict=False)
            )
            df_tpm_batch = df_tpm_batch.with_columns(
                pl.col(col).cast(pl.Float64, strict=False)
            )


        # Both dataframes now use id_col ('group_id' or 'transcript_id_int')
        df_counts_long = df_counts_batch.melt(id_vars=[id_col], variable_name='sample_id', value_name='count')
        df_tpm_long = df_tpm_batch.melt(id_vars=[id_col], variable_name='sample_id', value_name='tpm')
        
        df_long = df_counts_long.join(df_tpm_long, on=['sample_id', id_col])
        df_long = df_long.with_columns([
            pl.col("count").cast(pl.Float64, strict=False), pl.col("tpm").cast(pl.Float64, strict=False)
        ])

        df_long = df_long.filter((pl.col("count") > 0) | (pl.col("tpm") > 0))
        
        if df_long.is_empty():
            print("    No expressed features in this batch. Skipping.")
            continue
            
        df_long = df_long.with_columns(pl.col('sample_id').replace(sample_map).alias('sample_id_int'))
        
        temp_csv = f"/tmp/temp_matrix_{table_name}_{os.getpid()}_batch{i}.csv"
        df_long.select(columns_to_write).write_csv(temp_csv, include_header=False)
        import_commands = f".mode csv\n.import {temp_csv} {table_name}\n"
        subprocess.run(
            ['sqlite3', db_path],
            input=import_commands,
            text=True,
            check=True,
            capture_output=True  # Prevent output clutter
        )
       
        os.remove(temp_csv)
       
        del df_counts_batch, df_tpm_batch, df_counts_long, df_tpm_long, df_long
        

def ingest_per_sample_data(con, cur, db_path, table_name, per_sample_map, file_key, schema_dict, sample_map, transcript_map):
    """
    The definitive, high-performance ingestion function. It uses a static,
    hardcoded Polars schema to robustly read files and create the database table.
    """
    print(f"--- Ingesting data for table: {table_name} (Static Schema) ---")
   
    # --- Step 1: Create the table with the correct schema, IF it doesn't exist ---
    try:
        # Check if table already has data.
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        if cur.fetchone()[0] > 0:
            print(f"  Table '{table_name}' already contains data. Skipping ingestion.")
            return
    except sqlite3.OperationalError:
        # Table doesn't exist, so we create it.
        print(f"  -> Table '{table_name}' not found. Creating it now...")
        
        # Map Polars dtypes to SQLite dtypes for the CREATE TABLE statement
        sql_dtype_map = {
            pl.Int64: 'INTEGER', pl.Int32: 'INTEGER', pl.Int16: 'INTEGER', pl.Int8: 'INTEGER',
            pl.UInt64: 'INTEGER', pl.UInt32: 'INTEGER', pl.UInt16: 'INTEGER', pl.UInt8: 'INTEGER',
            pl.Float64: 'REAL', pl.Float32: 'REAL',
            pl.Boolean: 'INTEGER', # Booleans are stored as 0 or 1 in SQLite
            pl.Utf8: 'TEXT'
        }
        
        final_table_columns = ["sample_id_int INTEGER", "transcript_id_int INTEGER"]
        
        # Iterate through the static schema, skipping the first key ('transcript_id_str')
        for col_name, dtype in list(schema_dict.items())[1:]:
            sql_type = sql_dtype_map.get(dtype, 'TEXT')
            final_table_columns.append(f'"{col_name}" {sql_type}')
            
        # Join them all into the final CREATE TABLE statement
        create_table_sql = f"CREATE TABLE {table_name} ({', '.join(final_table_columns)})"
        cur.execute(create_table_sql)
        con.commit()
    
    # --- Step 2: Loop through all files and ingest them ---
    column_names_from_schema = list(schema_dict.keys())
    
    # make_lookup df
    transcript_lookup = pl.DataFrame({
    'transcript_id_str': list(transcript_map.keys()),
    'transcript_id_int': list(transcript_map.values())
})
    
    # Use a single transaction for the entire table for maximum speed
    with con:
        for file_idx, (sample_name, paths_dict) in enumerate(per_sample_map.items()):
            file_path = paths_dict.get(file_key)
            if not file_path or not os.path.exists(file_path):
                continue
            print(f"  Processing file {file_idx + 1}/{len(per_sample_map)}: {os.path.basename(file_path)}...")

            # Read the file using our robust, static schema. This is the key.
            df_sample = pl.read_csv(
                file_path,
                separator='\t',
                has_header=False,
                skip_rows=1,
                new_columns=column_names_from_schema, # Assign the correct names
                dtypes=schema_dict,                  # Enforce the correct data types
                null_values=['', 'NA', 'None', 'nan', 'inf', 'true', 'false']
            )
            
            if df_sample.is_empty():
                continue

            df_sample = df_sample.with_columns([
                    pl.lit(sample_map.get(sample_name)).alias('sample_id_int')
                ])
            
            df_sample = df_sample.join(
                    transcript_lookup, 
                    on='transcript_id_str', 
                    how='left'
                )
            
            
            #Convert boolean columns to 0/1 for SQLite
            for col_name, dtype in schema_dict.items():
                if dtype == pl.Boolean:
                    df_sample = df_sample.with_columns(pl.col(col_name).cast(pl.Int64))
            
            

            # Select the final columns for the database, dropping the now-redundant string ID
            final_db_columns_for_select = ["sample_id_int", "transcript_id_int"] + column_names_from_schema[1:]
            df_to_write = df_sample.select(final_db_columns_for_select)
          
            # Use the ultra-fast CSV to subprocess import method
            temp_csv = f"temp_ingest_{table_name}_{os.getpid()}_{file_idx}.csv"
            df_to_write.write_csv(temp_csv, include_header=False)
            
            import_commands = f".mode csv\n.import {temp_csv} {table_name}\n"
            
            subprocess.run(
            ['sqlite3', db_path],
            input=import_commands,
            text=True,
            check=True,
            capture_output=True 
        )
           
            del df_sample, df_to_write
            os.remove(temp_csv)
            
            
            
    print(f"  Ingestion for '{table_name}' complete.")

    con.commit()


def ingest_summary_files_multiple(con, db_path, table_name, file_list, 
                                 schema, sample_map):
    """
    Process multiple summary files sequentially using the same pattern as matrix files.
    """
    if not file_list:
        print(f"  INFO: No files for '{table_name}'. Skipping.")
        return
    
    print(f"--- Ingesting {len(file_list)} files for table: {table_name} ---")
    
    # Check if table already has data ONCE before processing any files
    cur = con.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    if cur.fetchone()[0] > 0:
        print(f"  Table '{table_name}' already contains data. Skipping all ingestion.")
        return
    
    for idx, file_path in enumerate(file_list):
        print(f"  Processing file {idx+1}/{len(file_list)}: {os.path.basename(file_path)}")
        
        # Pass skip_if_exists=False so it doesn't check again
        ingest_summary_file(
            con, db_path, table_name, file_path, schema, sample_map,
            skip_if_exists=False  # Don't check again for subsequent files
        )

def ingest_summary_file(con, db_path, table_name, file_path, schema, sample_map, skip_if_exists=True):
    """
    Generic ingestion for a single, comprehensive summary file (like group_confidence).
    """
    print(f"--- Ingesting data for table: {table_name} ---")
    
    if not file_path or not os.path.exists(file_path):
        print(f"  INFO: File for '{table_name}' not found in manifest or at path. Skipping.")
        return

    cur = con.cursor()
    
    if skip_if_exists:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        if cur.fetchone()[0] > 0:
            print(f"  Table '{table_name}' already contains data. Skipping ingestion.")
            return

    print(f"  Processing file: {os.path.basename(file_path)}...")
    df = pl.read_csv(file_path, separator='\t', has_header=True)
    if df.is_empty():
        return
        
    df = df.with_columns(pl.col('sample').replace(sample_map).alias('sample_id_int'))
    
    column_names = [col['name'] for col in schema]
    temp_csv = f"temp_ingest_{table_name}_{os.getpid()}.csv"
    df.select(column_names).write_csv(temp_csv, include_header=False)
    
    subprocess.run([
        'sqlite3', db_path, '.mode csv', f'.import {temp_csv} {table_name}'
    ], check=True)
    
    os.remove(temp_csv)
    con.commit()



def process_cache_files(con, cur, cache_path_map, sample_map):
    """
    Loops through the specific cache files listed in the manifest ONCE.
    """
    print("--- Processing cache files specified in manifest ---")
    if not cache_path_map:
        print("  INFO: No cache files listed in the manifest. Skipping.")
        return

    # Check if tables are already populated
    cur.execute("SELECT COUNT(*) FROM transcript_junction_summary")
    if cur.fetchone()[0] > 0:
        print("  Junction/coverage tables already contain data. Skipping ingestion.")
        return

    file_count = len(cache_path_map)
    for i, (sample_name, cache_file_path) in enumerate(cache_path_map.items()):
        print(f"  Processing cache file {i + 1}/{file_count}: {os.path.basename(cache_file_path)} for sample {sample_name}...")
        if not os.path.exists(cache_file_path):
            print(f"    WARNING: Cache file for sample '{sample_name}' not found at '{cache_file_path}'. Skipping.")
            continue
            
        if sample_name not in sample_map:
            print(f"    WARNING: Sample '{sample_name}' not in metadata. Skipping.")
            continue
        
        sample_id_int = sample_map[sample_name]
        
    
        # --- SINGLE FILE I/O OPERATION ---
        with open(cache_file_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Extract all necessary data from the loaded pickle
        junction_metrics = cache_data.get('junction_metrics', {})
        coverage_metrics = cache_data.get('special_feature_counts', {})

        # --- Use with-con for automatic transaction per file ---
        with con:
            if junction_metrics:
                _ingest_junction_data_for_sample(cur, sample_id_int, junction_metrics)
            if coverage_metrics:
                _ingest_coverage_data_for_sample(cur, sample_id_int, coverage_metrics)
        
        print(f"    Committed data for sample {sample_name}")

def _ingest_junction_data_for_sample(cur, sample_id_int, junction_metrics):
    """
    Helper to bulk-insert junction data for a single sample.
    Now creates a deduplicated junction_counts table.
    """
    # Part 1: Ingest the transcript_junction_summary data.
    summary_data = []
    for transcript_id, metrics in junction_metrics.items():
        summary_data.append((
            sample_id_int,
            int(transcript_id),
            metrics.get('junction_read_count', 0),
            metrics.get('unique_junction_reads', 0),
            metrics.get('shared_junction_reads', 0),
            metrics.get('n_unique_junctions', 0),
            metrics.get('n_shared_junctions', 0)
        ))
    
    if summary_data:
        cur.executemany("INSERT INTO transcript_junction_summary VALUES (?, ?, ?, ?, ?, ?, ?)", summary_data)
    
    # Create a temporary dictionary to hold the single, authoritative read count
    # for each junction in this sample.
    sample_junction_counts = {}
    
    # Iterate through all transcripts to populate our junction-centric dictionary
    for transcript_id, metrics in junction_metrics.items():
        observed_junctions = metrics.get('observed_junctions', {})
        for junction_id, read_count in observed_junctions.items():
            # Method to effectively deduplicating our data.
            sample_junction_counts[int(junction_id)] = read_count
    
    # Now, convert our clean, deduplicated dictionary to a list of tuples for insertion.
    counts_data_to_insert = [
        (sample_id_int, jid, rcount) for jid, rcount in sample_junction_counts.items()
    ]
    
    if counts_data_to_insert:
        # The INSERT statement now correctly has 3 placeholders for the new, leaner table.
        cur.executemany(
            "INSERT INTO junction_counts VALUES (?, ?, ?)",
            counts_data_to_insert
        )


def _ingest_coverage_data_for_sample(cur, sample_id_int, coverage_metrics):
    """Helper to bulk-insert territory coverage data for a single sample."""
    coverage_data_to_insert = []
    for region_key, mean_coverage in coverage_metrics.items():
        try:
            region_id = int(region_key.split('_')[1])
            coverage_data_to_insert.append((sample_id_int, region_id, mean_coverage))
        except (IndexError, ValueError):
            continue
    
    if coverage_data_to_insert:
        cur.executemany("INSERT INTO territory_coverage VALUES (?, ?, ?)", coverage_data_to_insert)

def create_indexes(cur):
    """Creates all indexes on the tables after data has been loaded."""
    print("--- Creating indexes on all tables (this may take a while)... ---")
    
    # Helper function for cleaner code
    def index_table(table, pk_cols, other_indexes=[]):
        print(f"  Indexing {table}...")
        if pk_cols:
            pk_sql = f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_pk ON {table} ({', '.join(pk_cols)})"
            cur.execute(pk_sql)
        for col in other_indexes:
            idx_sql = f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table} ({col})"
            cur.execute(idx_sql)

    # --- Indexing Quantitative Results Tables ---
    index_table('priors', ['sample_id_int', 'transcript_id_int'], ['transcript_id_int'])
    index_table('counts_with_confidence', ['sample_id_int', 'transcript_id_int'], ['transcript_id_int'])
    index_table('final_counts_tpm', ['sample_id_int', 'transcript_id_int'], ['transcript_id_int'])
    #index_table('distinguishability_metrics', ['sample_id_int', 'transcript_id_int'], ['transcript_id_int'])
    
    # --- Indexing Aggregated/Group Tables ---
    index_table('group_confidence', ['sample_id_int', 'group_id'], ['group_id'])
    index_table('aggregated_counts_tpm', ['sample_id_int', 'group_id'], ['group_id'])

    # --- Indexing Annotation Context Tables ---
    index_table('transcript_annotations', ['transcript_id_int'], ['gene_id', 'transcript_id_str'])
    index_table('subset_supersets', ['subset_transcript_id_int', 'superset_transcript_id_int'], ['superset_transcript_id_int'])
    index_table('junction_annotations', ['junction_id_int'], ['junction_id_str'])
    
    
    # --- Indexing Junction Data Tables ---
    index_table('transcript_junction_summary', ['sample_id_int', 'transcript_id_int'], ['transcript_id_int'])
    index_table('junction_counts', ['sample_id_int', 'junction_id'], ['junction_id'])
    index_table('transcript_junctions', ['transcript_id_int', 'junction_id_int'], ['junction_id_int'])

    # --- Indexing Territory Data Tables ---
    index_table('territory_regions', ['region_id'])
    index_table('transcript_territories', ['transcript_id_int', 'region_id'], ['region_id'])
    index_table('territory_coverage', ['sample_id_int', 'region_id'], ['region_id'])
    
    print("Indexing complete.")

def finalize_db(con, cur):
    """Resets PRAGMA settings and closes the connection."""
    print("Finalizing database: resetting PRAGMA and closing connection.")
    cur.execute("PRAGMA synchronous = NORMAL")
    cur.execute("PRAGMA journal_mode = DELETE")
    con.commit()
    con.close()

# --- MAIN EXECUTION ---

def main():
    parser = argparse.ArgumentParser(
        description="Build a comprehensive SQLite database from MAJEC pipeline outputs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--run_manifests", nargs='+',
                        help="Paths to single manifest or multiple *_run_manifest.json files from chunked runs")
    
    parser.add_argument("--output_db", required=True,
                        help="Path to the output SQLite database file.")
    parser.add_argument("--metadata_file",
                        help="Recommended: A TSV file with sample metadata. Overrides parsing from matrix headers.")
    parser.add_argument("--force", action='store_true',
                        help="Overwrite the database file if it already exists.")
    args = parser.parse_args()

    # Load and process manifest(s)
    
    print(f" Processing {len(args.run_manifests)} manifest files")
    manifest = merge_manifests(args.run_manifests)
    
    if 'per_sample_files' in manifest and manifest['per_sample_files']:
        sample_names = list(manifest['per_sample_files'].keys())
    elif 'cache_files' in manifest and manifest['cache_files']:
        sample_names = list(manifest['cache_files'].keys())
    else:
        print("ERROR: Cannot derive sample list. Manifest is missing 'per_sample_files' or 'cache_files' keys.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(sample_names)} samples in the manifest.")
    
    # Get the sub-dictionary for per-sample files from the manifest
    per_sample_files_map = manifest.get('per_sample_files', {})
    summary_files_map = manifest.get('summary_files', {})
    annotation_path = manifest.get('annotation_file')
    
    if not annotation_path or not os.path.exists(annotation_path):
        print(f"ERROR: Annotation file specified in manifest not found at: {annotation_path}", file=sys.stderr)
        print("       If you have moved the project directory, please edit the paths in the manifest file.", file=sys.stderr)
        sys.exit(1)

    # --- Execute Pipeline ---
    con, cur = connect_db(args.output_db, args.force)
    
    try:
        create_schema(cur)
        
        if args.metadata_file:
            if not os.path.exists(args.metadata_file):
                print(f"FATAL ERROR: Specified metadata file not found at '{args.metadata_file}'", file=sys.stderr)
                sys.exit(1)
            sample_map = parse_and_ingest_metadata(args.output_db, args.metadata_file, sample_names)
            if not sample_map:
                print("FATAL ERROR: Metadata ingestion failed. Exiting.", file=sys.stderr)
                sys.exit(1)
        else:
            sample_map = extract_manifest_metadata(con, cur,  args.metadata_file,  sample_names)
        
        print("Loading annotation data...")
        with gzip.open(annotation_path, 'rb') as f:
            anno_data = pickle.load(f)
        
        ingest_annotation_data(con, cur, anno_data)
        if anno_data.get('subset_coverage_territory_mapping'):
            ingest_territory_data(con, cur, anno_data)
        
        # Now we can get the transcript map from the annotation data for other ingest functions
        transcript_map = anno_data['string_to_int']
        
         # Ingest main data tables
        ingest_per_sample_data( con, cur, args.output_db, 'counts_with_confidence',
            per_sample_files_map, 'metrics_sparse',  # <-- file_key (key from manifest dictionary)
            COUNTS_WITH_CONFIDENCE_SCHEMA, sample_map, transcript_map)
       
        ingest_per_sample_data(con, cur, args.output_db, 'priors', per_sample_files_map,
            'priors',  PRIORS_SCHEMA, sample_map, transcript_map)
       
        
        # Use wrappers for multiple files
        # Get the counts/TPM data for single run
        ingest_matrix_data_from_multiple_files(
            con, cur, 'aggregated_counts_tpm',
            summary_files_map.get('aggregated_counts', []),
            summary_files_map.get('aggregated_tpm', []),
            sample_map, None, args.output_db, is_aggregated=True
        )
        
        ingest_matrix_data_from_multiple_files(
            con, cur, 'final_counts_tpm',
            summary_files_map.get('transcript_counts', []),
            summary_files_map.get('transcript_tpm', []),
            sample_map, transcript_map, args.output_db, is_aggregated=False
        )
        
        ingest_summary_files_multiple(
            con, args.output_db, 'group_confidence',
            summary_files_map.get('group_confidence', []),
            GROUP_CONFIDENCE_SCHEMA, sample_map
        )       
       
        cache_path_map = manifest.get('cache_files', {})    

        # Ingest junction and coverage data from cache
        process_cache_files(con, cur, cache_path_map, sample_map)
        create_indexes(cur)
        
        #supplemental indexing
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_transcript_id_map_str ON transcript_id_map (transcript_id_str);")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_transcript_id_map_int ON transcript_id_map (transcript_id_int);")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_junction_id_map_str ON junction_id_map (junction_id_str);")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_junction_id_map_int ON junction_id_map (junction_id_int);")
 
    except Exception as e:
        print(f"\nFATAL ERROR occurred during database creation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        con.rollback()
        sys.exit(1)
    finally:
        finalize_db(con, cur)
        
    print("\nDatabase creation complete!")
    print(f"Database file located at: {args.output_db}")

if __name__ == '__main__':
    main()