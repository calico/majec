#!/usr/bin/env python
"""
Pre-compute ALL annotation data for MAJEC pipeline.
Creates a comprehensive annotation file with integer IDs, gene mappings, and junction data.
"""

import argparse
import pickle
import json
import hashlib
import re
import logging
import sys
import os
import gzip
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

try:
    from intervaltree import Interval, IntervalTree

    intervaltree_loaded = True
except ImportError as e:
    logging.error(
        f"Failed to import 'intervaltree'. Can not generate_rescue_features . Error: {e}"
    )
    intervaltree_loaded = False


def setup_logging():
    """Sets up logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def calculate_gtf_checksum(gtf_path):
    """Calculate MD5 checksum of GTF file."""
    hash_md5 = hashlib.md5()
    with open(gtf_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_chrom_order_from_file(file_path):
    """
    Reads a genome file and performs validation checks.
    """
    try:
        chrom_order = []
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # Skip empty lines and comments

                parts = line.split("\t")

                # --- The New Scrutiny ---
                if len(parts) < 1:
                    logging.warning(
                        f"Skipping malformed line {i+1} in {file_path}: line is empty."
                    )
                    continue

                chrom_name = parts[0]

                # Check for weird characters or spaces
                if " " in chrom_name or "\t" in chrom_name:
                    logging.warning(
                        f"Chromosome name '{chrom_name}' on line {i+1} contains whitespace. This may cause issues."
                    )

                chrom_order.append(chrom_name)

        if not chrom_order:
            raise ValueError("File is empty or contains no valid chromosome names.")

        logging.info(
            f"Successfully loaded and validated {len(chrom_order)} chromosomes from {os.path.basename(file_path)}."
        )
        return chrom_order

    except FileNotFoundError:
        logging.error(f"FATAL: Chromosome sort order file not found at: {file_path}")
        raise
    except Exception as e:
        logging.error(f"FATAL: Failed to parse chromosome order file: {e}")
        raise


def parse_complete_gtf(gtf_path, feature_type="gene"):
    """
    Parse GTF file completely, extracting all annotation information including TSL.

    Returns:
        transcript_info: {transcript_id: {gene_id, chr, strand, exons, tsl, ...}}
        gene_to_transcripts: {gene_id: [transcript_ids]}
    """
    logging.info(f"Parsing {feature_type} GTF: {gtf_path}")
    chrom_order = []
    header_done = False
    transcript_info = {}
    gene_to_transcripts = defaultdict(set)
    transcript_to_exons = defaultdict(list)

    with open(gtf_path, "r") as f:
        for line in f:
            if line.startswith("##sequence-region") and not header_done:
                parts = line.strip().split()
                if len(parts) >= 2:
                    chrom_order.append(parts[1])

            if not line.startswith("#"):
                header_done = True  # Stop looking for header lines

            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue

            # Parse basic fields
            chrom = fields[0]
            feature = fields[2]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attributes = fields[8]

            # Extract IDs
            transcript_match = re.search(r'transcript_id "([^"]+)"', attributes)
            gene_match = re.search(r'gene_id "([^"]+)"', attributes)

            if not transcript_match:
                continue

            transcript_id = transcript_match.group(1).strip()
            gene_id = gene_match.group(1).strip() if gene_match else transcript_id

            # For transcript features, extract TSL
            if feature == "transcript":
                # Try different TSL formats
                tsl_match = re.search(r'transcript_support_level "([^"]+)"', attributes)
                if not tsl_match:
                    tsl_match = re.search(r'tsl "([^"]+)"', attributes)

                tsl = tsl_match.group(1) if tsl_match else "NA"

                # Initialize transcript info with TSL
                if transcript_id not in transcript_info:
                    transcript_info[transcript_id] = {
                        "gene_id": gene_id,
                        "chr": chrom,
                        "strand": strand,
                        "exons": [],
                        "start": float("inf"),
                        "end": 0,
                        "feature_type": feature_type,
                        "tsl": tsl,
                    }
                else:
                    # Update TSL if we found it
                    transcript_info[transcript_id]["tsl"] = tsl

            # Initialize transcript info if not seen in transcript feature
            if transcript_id not in transcript_info:
                transcript_info[transcript_id] = {
                    "gene_id": gene_id,
                    "chr": chrom,
                    "strand": strand,
                    "exons": [],
                    "start": float("inf"),
                    "end": 0,
                    "feature_type": feature_type,
                    "tsl": "NA",  # Default if no TSL found
                }

            # Update bounds and collect exons
            if feature == "exon":
                transcript_info[transcript_id]["exons"].append((start, end))
                transcript_info[transcript_id]["start"] = min(
                    transcript_info[transcript_id]["start"], start
                )
                transcript_info[transcript_id]["end"] = max(
                    transcript_info[transcript_id]["end"], end
                )
                transcript_to_exons[transcript_id].append(
                    {"chr": chrom, "start": start, "end": end, "strand": strand}
                )

            gene_to_transcripts[gene_id].add(transcript_id)

    # Sort exons for each transcript
    for tid, info in transcript_info.items():
        info["exons"] = sorted(info["exons"])
        # Calculate mature mRNA length (sum of exon lengths)
        if info["exons"]:
            info["transcript_length"] = sum(end - start for start, end in info["exons"])
        else:
            info["transcript_length"] = 0

    # Log TSL statistics
    tsl_counts = defaultdict(int)
    for info in transcript_info.values():
        tsl_counts[info["tsl"]] += 1

    logging.info(
        f"  Parsed {len(transcript_info)} transcripts in {len(gene_to_transcripts)} genes/families"
    )
    logging.info(f"  TSL distribution: {dict(tsl_counts)}")

    return transcript_info, dict(gene_to_transcripts), transcript_to_exons, chrom_order


def create_integer_mappings(all_feature_ids):
    """Create bidirectional mappings between string IDs and integers."""
    sorted_ids = sorted(list(all_feature_ids))
    string_to_int = {tid: i for i, tid in enumerate(sorted_ids)}
    int_to_string = sorted_ids
    return string_to_int, int_to_string


def compute_junction_data(transcript_to_exons, string_to_int):
    """
    Compute junction uniqueness data with integer IDs.

    Returns:
        junction_map: {junction_key: {'transcripts': [int], 'is_unique': bool, ...}}
        transcript_unique_junctions: {transcript_id_int: [junction_keys]}
    """
    logging.info("Computing junction uniqueness data...")

    transcript_to_junctions = defaultdict(set)
    junction_to_transcripts = defaultdict(set)

    for transcript_id, exons in transcript_to_exons.items():
        transcript_int = string_to_int[transcript_id]
        exons_sorted = sorted(exons, key=lambda x: (x["start"], x["end"]))

        for i in range(len(exons_sorted) - 1):
            exon1 = exons_sorted[i]
            exon2 = exons_sorted[i + 1]

            # Junction key stays as string (coordinates)
            junction_key = (
                f"{exon1['chr']}:{exon1['end']}-{exon2['start']}:{exon1['strand']}"
            )
            transcript_to_junctions[transcript_int].add(junction_key)
            junction_to_transcripts[junction_key].add(transcript_int)

    # Build junction map with integer IDs
    junction_map = {}
    transcript_unique_junctions = defaultdict(list)

    for junction_key, transcript_ints in junction_to_transcripts.items():
        is_unique = len(transcript_ints) == 1

        junction_map[junction_key] = {
            "transcripts": list(
                transcript_ints
            ),  # Note: 'transcripts' not 'transcript_ids'
            "is_unique": is_unique,
            "n_transcripts": len(transcript_ints),
        }

        if is_unique:
            transcript_int = list(transcript_ints)[0]
            transcript_unique_junctions[transcript_int].append(junction_key)

    unique_count = sum(1 for info in junction_map.values() if info["is_unique"])
    logging.info(f"  Found {len(junction_map)} junctions ({unique_count} unique)")

    return junction_map, dict(transcript_unique_junctions)


def create_annotation_maps(gene_transcript_info, te_transcript_info, string_to_int):
    """
    Create gene_map and te_map DataFrames with integer IDs.

    Args:
        gene_transcript_info: Dict of transcript_id -> info including gene_id
        te_transcript_info: Dict of transcript_id -> info including gene_id (TE family)
        string_to_int: Mapping from string transcript IDs to integers

    Returns:
        tuple: (gene_map DataFrame, te_map DataFrame or None)
    """
    import pandas as pd

    # Build gene map
    gene_map_data = []
    for tid, info in gene_transcript_info.items():
        tid_int = string_to_int[tid]
        gene_id = info["gene_id"]
        gene_map_data.append([tid_int, gene_id])

    gene_map = pd.DataFrame(gene_map_data, columns=["LocusID", "AggregateID"])

    # Build TE map if TEs exist
    te_map = None
    if te_transcript_info:
        te_map_data = []
        for tid, info in te_transcript_info.items():
            tid_int = string_to_int[tid]
            te_family = info["gene_id"]  # For TEs, gene_id is actually the TE family
            te_map_data.append([tid_int, te_family])
        te_map = pd.DataFrame(te_map_data, columns=["LocusID", "AggregateID"])

    return gene_map, te_map


def calculate_transcript_expected_junctions(transcript_to_exons, string_to_int):
    """
    For each transcript, identify all its theoretical junctions.

    Returns:
        transcript_expected_junctions: {transcript_id_int: {
            'n_expected': int,
            'junction_keys': [junction_key, ...]
        }}
    """
    logging.info("Calculating expected junctions per transcript...")

    transcript_expected_junctions = {}

    for tid, exons in transcript_to_exons.items():
        tid_int = string_to_int[tid]
        exons_sorted = sorted(exons, key=lambda x: (x["start"], x["end"]))

        # Count expected junctions (n_exons - 1)
        n_expected = len(exons_sorted) - 1

        # Generate junction keys
        expected_junctions = []
        for i in range(len(exons_sorted) - 1):
            exon1 = exons_sorted[i]
            exon2 = exons_sorted[i + 1]
            junction_key = (
                f"{exon1['chr']}:{exon1['end']}-{exon2['start']}:{exon1['strand']}"
            )
            expected_junctions.append(junction_key)

        transcript_expected_junctions[tid_int] = {
            "n_expected": n_expected,
            "junction_keys": expected_junctions,
        }

    # Log statistics
    n_single_exon = sum(
        1 for v in transcript_expected_junctions.values() if v["n_expected"] == 0
    )
    n_multi_exon = len(transcript_expected_junctions) - n_single_exon
    avg_junctions = np.mean(
        [v["n_expected"] for v in transcript_expected_junctions.values()]
    )

    logging.info(f"  Single-exon transcripts: {n_single_exon:,}")
    logging.info(f"  Multi-exon transcripts: {n_multi_exon:,}")
    logging.info(f"  Average junctions per transcript: {avg_junctions:.1f}")

    return transcript_expected_junctions


def create_transcript_junction_order(junction_map):
    """
    Restructures a junction_map to create a sorted list of junctions for each
    transcript.

    Args:
        junction_map (dict): The annotation map of {junction_key: {'transcripts': [...]}}.

    Returns:
        dict: A dictionary mapping {transcript_id: [sorted_junction_keys]}.
    """
    # Step 1: Invert the map to get {transcript_id: [junctions]}
    transcript_to_junctions = defaultdict(list)
    for junction_key, info in junction_map.items():
        for transcript_id in info.get("transcripts", []):
            transcript_to_junctions[transcript_id].append(junction_key)

    # Step 2: Sort the junctions for each transcript
    transcript_junction_order = {}
    for transcript_id, junctions in transcript_to_junctions.items():
        if not junctions:
            continue

        # Determine the strand from the first junction in the list.
        # All junctions for a transcript will be on the same strand.
        strand = junctions[0][-1]
        is_reverse_strand = strand == "-"

        # Sort by the integer start coordinate of the junction.
        # For '-' strand transcripts, reverse the sort to maintain 5'->3' order.
        try:
            sorted_junctions = sorted(
                junctions,
                key=lambda j: int(j.split(":")[1].split("-")[0]),
                reverse=is_reverse_strand,
            )
            transcript_junction_order[transcript_id] = sorted_junctions
        except (IndexError, ValueError):
            # Handle potential malformed junction strings gracefully
            print(
                f"Warning: Could not parse and sort junctions for transcript {transcript_id}"
            )
            continue

    return transcript_junction_order


def build_splice_site_competition_map(all_junctions):
    """Build competition map using pandas groupby"""
    logging.info("Building splice site competition map...")

    if not all_junctions:
        logging.info("  No junctions to process — returning empty competition map")
        return {}

    # Parse junctions into DataFrame
    data = []
    for junction in all_junctions:
        chr_name, coords, strand = junction.split(":")
        start, end = coords.split("-")
        data.append(
            {
                "junction": junction,
                "chr": chr_name,
                "start": int(start),
                "end": int(end),
                "strand": strand,
            }
        )

    df = pd.DataFrame(data)

    # Group by chr+strand+start (same donor)
    donor_groups = df.groupby(["chr", "strand", "start"])["junction"].apply(set)

    # Group by chr+strand+end (same acceptor)
    acceptor_groups = df.groupby(["chr", "strand", "end"])["junction"].apply(set)

    # Build competitor map
    junction_competitors = defaultdict(set)
    for group in donor_groups:
        if len(group) > 1:
            for j1 in group:
                junction_competitors[j1].update(group - {j1})

    for group in acceptor_groups:
        if len(group) > 1:
            for j1 in group:
                junction_competitors[j1].update(group - {j1})

    return dict(junction_competitors)


def process_region_for_subsets(args):
    """
    Process a single genomic region for subset relationships.
    Much smaller N for N² comparison.
    """
    region_key, transcript_junction_sets = args

    # Build local junction index just for this region
    junction_to_transcripts = defaultdict(set)
    for tid, junctions in transcript_junction_sets.items():
        for junction in junctions:
            junction_to_transcripts[junction].add(tid)

    # Group by junction count
    by_junction_count = defaultdict(list)
    for tid, junctions in transcript_junction_sets.items():
        if len(junctions) > 0:
            by_junction_count[len(junctions)].append(tid)

    subset_relationships = {}

    # Now we're only comparing transcripts in the same genomic region!
    # For a typical gene with 5-20 isoforms, this is MUCH faster
    for subset_size in sorted(by_junction_count.keys()):
        for subset_tid in by_junction_count[subset_size]:
            subset_junctions = transcript_junction_sets[subset_tid]

            # Find potential supersets
            potential_supersets = None
            for junction in subset_junctions:
                transcripts_with_junction = junction_to_transcripts[junction]
                if potential_supersets is None:
                    potential_supersets = transcripts_with_junction.copy()
                else:
                    potential_supersets &= transcripts_with_junction

            if not potential_supersets:
                continue

            potential_supersets.discard(subset_tid)

            # Check supersets
            for superset_tid in potential_supersets:
                if len(transcript_junction_sets[superset_tid]) > subset_size:
                    if subset_tid not in subset_relationships:
                        subset_relationships[subset_tid] = {
                            "supersets": [],
                            "shared_junctions": list(subset_junctions),
                        }
                    subset_relationships[subset_tid]["supersets"].append(superset_tid)

    # Calculate coverage fractions
    for subset_tid, info in subset_relationships.items():
        n_subset_junctions = len(info["shared_junctions"])
        max_superset_junctions = max(
            len(transcript_junction_sets[sid]) for sid in info["supersets"]
        )
        info["coverage_fraction"] = n_subset_junctions / max_superset_junctions
        info["unique_junction_deficit"] = max_superset_junctions - n_subset_junctions

    return subset_relationships


def identify_subset_superset_relationships_parallel(
    junction_map, transcript_info, n_cores=6, bin_size=50_000_000, overlap=10_000_000
):
    """
    Partition with overlapping bins to catch boundary cases.
    """
    logging.info("Identifying subset/superset relationships with overlapping bins...")

    # Build transcript junction sets and determine spans
    transcript_junction_sets = {}
    transcript_spans = {}  # transcript -> (chrom, strand, start, end)

    for junction_key, info in junction_map.items():
        chrom, coords, strand = junction_key.split(":")
        j_start, j_end = map(int, coords.split("-"))

        for transcript_int in info["transcripts"]:
            if transcript_int not in transcript_junction_sets:
                transcript_junction_sets[transcript_int] = set()
                # Initialize span
                if transcript_int in transcript_info:
                    t_info = transcript_info[transcript_int]
                    transcript_spans[transcript_int] = (
                        t_info.get("chr", chrom),
                        t_info.get("strand", strand),
                        t_info.get("start", float("inf")),
                        t_info.get("end", 0),
                    )
                else:
                    transcript_spans[transcript_int] = (chrom, strand, float("inf"), 0)

            transcript_junction_sets[transcript_int].add(junction_key)

            # Update span based on junction coordinates
            curr_span = transcript_spans[transcript_int]
            transcript_spans[transcript_int] = (
                curr_span[0],
                curr_span[1],
                min(curr_span[2], j_start),
                max(curr_span[3], j_end),
            )

    # Create overlapping bins
    bins = defaultdict(list)

    for transcript_int, (chrom, strand, start, end) in transcript_spans.items():
        # Assign to all bins this transcript overlaps
        start_bin = start // bin_size
        end_bin = end // bin_size

        # Add to primary bins
        for bin_num in range(start_bin, end_bin + 1):
            bin_key = (chrom, strand, bin_num, "primary")
            bins[bin_key].append(transcript_int)

        # Also add to overlap bins if near boundaries
        if start % bin_size < overlap:  # Near start of bin
            overlap_bin = start_bin - 1
            if overlap_bin >= 0:
                bin_key = (chrom, strand, overlap_bin, "overlap")
                bins[bin_key].append(transcript_int)

        if (bin_size - end % bin_size) < overlap:  # Near end of bin
            overlap_bin = end_bin + 1
            bin_key = (chrom, strand, overlap_bin, "overlap")
            bins[bin_key].append(transcript_int)

    # Prepare work units
    work_units = []
    for bin_key, transcript_ids in bins.items():
        if len(transcript_ids) >= 2:
            region_junction_sets = {
                tid: transcript_junction_sets[tid] for tid in transcript_ids
            }
            work_units.append((bin_key, region_junction_sets))

    logging.info(f"  Created {len(work_units)} work units (with overlaps)")

    # Process in parallel
    from multiprocessing import Pool

    with Pool(n_cores) as pool:
        region_results = pool.map(process_region_for_subsets, work_units)

    # Combine results - deduplication needed!
    all_subset_relationships = {}
    for region_subsets in region_results:
        for subset_tid, relationship in region_subsets.items():
            if subset_tid not in all_subset_relationships:
                all_subset_relationships[subset_tid] = relationship
            else:
                # Merge superset lists (deduped)
                existing_supersets = set(
                    all_subset_relationships[subset_tid]["supersets"]
                )
                new_supersets = set(relationship["supersets"])
                all_subset_relationships[subset_tid]["supersets"] = list(
                    existing_supersets | new_supersets
                )

    # Recalculate coverage fractions after deduplication
    for subset_tid, info in all_subset_relationships.items():
        n_subset_junctions = len(info["shared_junctions"])
        max_superset_junctions = max(
            len(transcript_junction_sets[sid]) for sid in info["supersets"]
        )
        info["coverage_fraction"] = n_subset_junctions / max_superset_junctions
        info["unique_junction_deficit"] = max_superset_junctions - n_subset_junctions

    logging.info(f"  Found {len(all_subset_relationships)} subset relationships")

    return all_subset_relationships


# --- Global variable for worker data ---
_worker_data = {}


def init_worker(subset_rels, t_to_exons, i_to_s, min_size):
    """Initializer for the worker processes."""
    global _worker_data
    _worker_data["subset_relationships"] = subset_rels
    _worker_data["transcript_to_exons"] = t_to_exons
    _worker_data["int_to_string"] = i_to_s
    _worker_data["min_unique_size"] = min_size


def characterize_unique_exonic_territory_parallel(
    subset_relationships,
    transcript_to_exons,
    int_to_string,
    min_unique_size=30,
    num_workers=8,
):
    """
    Manager function that chunks the data and runs the characterization in parallel.
    """
    logging.info(
        f"Characterizing {len(subset_relationships)} subsets in parallel with {num_workers} workers..."
    )

    # --- Chunk the subset IDs into smaller lists ---
    subset_ids = list(subset_relationships.keys())
    # Create more chunks than workers for better load balancing, especially if some genes are complex.
    num_chunks = num_workers * 4
    chunks_of_ids = np.array_split(subset_ids, num_chunks)

    final_map = {}
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(
            subset_relationships,
            transcript_to_exons,
            int_to_string,
            min_unique_size,
        ),
    ) as executor:

        # Map the worker function to the chunks of IDs
        results = executor.map(_worker_characterize_chunk, chunks_of_ids)

        # Combine the results from all the chunks
        for chunk_result in results:
            final_map.update(chunk_result)

    # --- Final Logging (copied from your original function) ---
    logging.info(
        f"  Found unique territories for {len(final_map)}/{len(subset_relationships)} subsets"
    )
    event_counts = defaultdict(int)
    for territories in final_map.values():
        for territory in territories:
            event_counts[territory["type"]] += 1
    for event_type, count in event_counts.items():
        logging.info(f"    {event_type}: {count} events")

    return final_map


def _worker_characterize_chunk(subset_chunk):
    """
    Worker function that takes a CHUNK of subset_relationships and processes it.
    It uses the global _worker_data initialized for the process.
    This function's logic is a direct copy of the original, working version.
    """
    # Unpack the large, read-only data structures from the global scope
    subset_relationships = _worker_data[
        "subset_relationships"
    ]  # The full map is needed for context
    transcript_to_exons = _worker_data["transcript_to_exons"]
    int_to_string = _worker_data["int_to_string"]
    min_unique_size = _worker_data["min_unique_size"]

    # Import necessary libraries within the worker
    from intervaltree import IntervalTree, Interval
    from collections import defaultdict

    # This is the dictionary for the results from THIS specific chunk
    chunk_map = {}

    # Now, the original function's main loop runs here, on the smaller subset_chunk
    # subset_chunk is just a list of subset_ids to process
    for subset_id in subset_chunk:
        details = subset_relationships[subset_id]

        subset_tid_str = int_to_string[subset_id]
        subset_exons = transcript_to_exons.get(subset_tid_str, [])

        if not subset_exons:
            continue

        # Sort subset exons by position
        subset_exons_sorted = sorted(subset_exons, key=lambda x: x["start"])
        chr_name = subset_exons_sorted[0]["chr"]

        unique_territories = []

        for superset_id in details["supersets"]:
            superset_tid_str = int_to_string[superset_id]
            superset_exons = transcript_to_exons.get(superset_tid_str, [])

            if not superset_exons:
                continue

            # Sort superset exons by position
            superset_exons_sorted = sorted(superset_exons, key=lambda x: x["start"])

            subset_tree = IntervalTree()
            for exon in subset_exons_sorted:
                if exon["start"] != exon["end"]:
                    subset_tree.add(Interval(exon["start"], exon["end"], exon))
            subset_tree.merge_overlaps()

            superset_tree = IntervalTree()
            for exon in superset_exons_sorted:
                if exon["start"] != exon["end"]:
                    superset_tree.add(Interval(exon["start"], exon["end"], exon))
            superset_tree.merge_overlaps()

            # === CASE 1: Alternative Low-Coordinate Terminus (Optimized) ===
            subset_first_exon = subset_exons_sorted[0]
            overlapping_intervals = superset_tree.overlap(
                subset_first_exon["start"], subset_first_exon["end"]
            )

            if overlapping_intervals:
                superset_comparator_interval = min(
                    overlapping_intervals, key=lambda iv: iv.begin
                )
                superset_comparator = superset_comparator_interval.data

                if subset_first_exon["start"] < superset_comparator["start"]:
                    unique_region_size = (
                        superset_comparator["start"] - subset_first_exon["start"]
                    )
                    if unique_region_size >= min_unique_size:
                        unique_territories.append(
                            {
                                "type": "alt_low_coord_terminus",
                                "unique_coords": f"{chr_name}:{subset_first_exon['start']}-{superset_comparator['start']}",
                                "comparator_coords": f"{chr_name}:{superset_comparator['start']}-{superset_comparator['end']}",
                                "size": unique_region_size,
                                "superset_id": superset_id,
                                "description": "Subset extends beyond superset at low-coordinate end",
                            }
                        )

            # === CASE 2: Alternative High-Coordinate Terminus (Optimized) ===
            subset_last_exon = subset_exons_sorted[-1]
            overlapping_intervals = superset_tree.overlap(
                subset_last_exon["start"], subset_last_exon["end"]
            )

            if overlapping_intervals:
                superset_comparator_interval = max(
                    overlapping_intervals, key=lambda iv: iv.end
                )
                superset_comparator = superset_comparator_interval.data

                if subset_last_exon["end"] > superset_comparator["end"]:
                    unique_region_size = (
                        subset_last_exon["end"] - superset_comparator["end"]
                    )
                    if unique_region_size >= min_unique_size:
                        unique_territories.append(
                            {
                                "type": "alt_high_coord_terminus",
                                "unique_coords": f"{chr_name}:{superset_comparator['end']}-{subset_last_exon['end']}",
                                "comparator_coords": f"{chr_name}:{superset_comparator['start']}-{superset_comparator['end']}",
                                "size": unique_region_size,
                                "superset_id": superset_id,
                                "description": "Subset extends beyond superset at high-coordinate end",
                            }
                        )

            # === CASE 3: Internal differences (retained introns) ===
            subset_unique = IntervalTree(subset_tree)
            for interval in superset_tree:
                subset_unique.chop(interval.begin, interval.end)

            for interval in subset_unique:
                if (
                    interval.begin > subset_exons_sorted[0]["start"] + 100
                    and interval.end < subset_exons_sorted[-1]["end"] - 100
                ):

                    upstream_exon, downstream_exon = None, None
                    for sup_exon in superset_exons_sorted:
                        if sup_exon["end"] <= interval.begin:
                            if (
                                not upstream_exon
                                or sup_exon["end"] > upstream_exon["end"]
                            ):
                                upstream_exon = sup_exon
                        elif sup_exon["start"] >= interval.end:
                            if (
                                not downstream_exon
                                or sup_exon["start"] < downstream_exon["start"]
                            ):
                                downstream_exon = sup_exon

                    if (
                        upstream_exon
                        and downstream_exon
                        and interval.length() >= min_unique_size
                    ):
                        comparator_coords = [
                            f"{chr_name}:{upstream_exon['start']}-{upstream_exon['end']}",
                            f"{chr_name}:{downstream_exon['start']}-{downstream_exon['end']}",
                        ]
                        unique_territories.append(
                            {
                                "type": "retained_intron",
                                "unique_coords": f"{chr_name}:{interval.begin}-{interval.end}",
                                "comparator_coords": comparator_coords,
                                "size": interval.length(),
                                "superset_id": superset_id,
                                "description": "Intron retained in subset, spliced in superset",
                            }
                        )

        if unique_territories:
            chunk_map[subset_id] = unique_territories

    return chunk_map


def build_interval_trees(transcript_to_exons, string_to_int):
    """
    Builds a dictionary of interval trees for all exons in the annotation.

    Args:
        transcript_to_exons (dict): The ground-truth exon data {str_tid: [{'chr':c, 'start':s, 'end':e, 'strand':str}]}.
        string_to_int (dict): Mapping from string IDs to integer IDs.

    Returns:
        dict: A nested dictionary of interval trees, keyed by {chrom: {strand: IntervalTree}}.
    """
    logging.info("Building interval trees for all exons...")
    # The structure is: { 'chr1': {'+': IntervalTree, '-': IntervalTree}, 'chr2': ... }
    trees = defaultdict(lambda: defaultdict(IntervalTree))

    # Iterate through the original, string-keyed exon dictionary
    for transcript_id_str, exons in transcript_to_exons.items():
        if not exons:
            continue

        # Get the integer ID for this transcript to use as the data payload
        transcript_id_int = string_to_int.get(transcript_id_str)
        if transcript_id_int is None:
            # Not sure how this would happpen but sadtey check in case
            continue

        # Use the first exon to get chrom and strand, assuming they are consistent for the transcript.
        chrom = exons[0]["chr"]
        strand = exons[0]["strand"]
        # print(chrom, strand)
        for exon in exons:
            if (
                exon["start"] != exon["end"]
            ):  # safty check for zero-length exons could potentially filter GTF vecotwise earlier
                # Create an Interval object for each exon.
                # The interval is [start, end). The library handles the half-open interval correctly.
                # The data payload is the integer transcript ID.
                trees[chrom][strand].add(
                    Interval(exon["start"], exon["end"], transcript_id_int)
                )

    logging.info(
        f"Successfully built and merged interval trees for {len(trees)} chromosomes."
    )
    return trees


def find_unambiguous_territory(candidate_map, interval_trees, transcript_info):
    """
    Takes a map of candidate unique regions and filters it by checking for
    any overlaps with other transcripts. This is the definitive "daylight" check.

    Args:
        candidate_map (dict): The map of {subset_id: [events]} from the characterization step.
        interval_trees (dict): The pre-built, un-merged interval trees for all exons.
        transcript_info (dict): Metadata for all transcripts {int_tid: info}.

    Returns:
        dict: A new, filtered map containing only events with truly unique territory.
    """
    logging.info(
        f"Searching for unambiguous 'daylight' in {len(candidate_map)} candidate subsets..."
    )
    final_rescue_map = defaultdict(list)

    # Iterate through each subset that had a potential unique region
    for subset_id, events in candidate_map.items():

        # This list will store the validated events for this subset
        validated_events = []

        for event in events:
            try:
                # --- Step 1: Get all necessary data for the event ---
                # This is the code that replaces the placeholder comment.
                coords_str = event["unique_coords"]
                chrom, coords = coords_str.split(":")
                start, end = map(int, coords.split("-"))

                # The strand is determined by the subset transcript itself
                strand = transcript_info[subset_id].get("strand", "+")

                # --- Step 2: Query the global tree for ALL overlaps in the candidate region ---
                overlapping_intervals = interval_trees[chrom][strand].overlap(
                    start, end
                )

                # --- Step 3: Keep everything that is NOT the subset itself. ---
                other_overlaps = {
                    iv for iv in overlapping_intervals if iv.data != subset_id
                }

                # --- Step 4: Perform the "Daylight Chop" ---
                # We start with our candidate region as a tree.
                daylight_tree = IntervalTree([Interval(start, end)])

                # Now, subtract every single other overlapping exon from it.
                for overlap in other_overlaps:
                    daylight_tree.chop(overlap.begin, overlap.end)

                # --- Step 5: The Final Decision ---
                # If there is any interval left in our tree, we have found daylight!
                if daylight_tree:
                    # Find the largest remaining chunk of daylight.
                    largest_daylight_interval = max(
                        daylight_tree, key=lambda i: i.length()
                    )

                    # We create a new, validated event with the precise coordinates of the daylight.
                    updated_event = event.copy()
                    updated_event["unique_coords"] = (
                        f"{chrom}:{largest_daylight_interval.begin}-{largest_daylight_interval.end}"
                    )
                    updated_event["size"] = (
                        largest_daylight_interval.length()
                    )  # Update the size

                    validated_events.append(updated_event)

            except Exception as e:
                # This robustly handles any unexpected errors for a single event.
                logging.warning(
                    f"Could not process event for subset {subset_id} ({event.get('unique_coords', 'N/A')}): {e}"
                )
                continue

        # After checking all events for a subset, if any were validated, add them to the final map.
        if validated_events:
            final_rescue_map[subset_id].extend(validated_events)

    logging.info(
        f"Found {len(final_rescue_map)} subsets with validated, unambiguous territory."
    )
    return dict(final_rescue_map)


def write_territories_to_bed_with_mapping(
    final_rescue_map,
    transcript_info,
    output_bed_path,
    chrom_order,
    output_mapping_path=None,
):
    """
    Convert territories to deduplicated BED format with integer IDs and relational mapping.

    Returns:
        territory_mapping: Dict with all relationships for interpreting coverage data
    """
    INTRON_BOUNDARY_TRIM = 3
    TERMINAL_TRIM = 1

    logging.info(f"Generating deduplicated BED file and territory mapping...")

    if chrom_order is None or len(chrom_order) == 0:
        logging.warning(
            "Could not determine chromosome order from GTF header. Falling back to simple sort."
        )
        # Fallback: get all chromosomes from the data and sort them simply
        all_chroms = {info["chr"] for info in transcript_info.values()}
        chrom_order = sorted(list(all_chroms))

    chrom_rank_map = {chrom: i for i, chrom in enumerate(chrom_order)}

    # Step 1: Collect all unique regions (fully deduplicated)
    unique_regions = {}  # {(chr, start, end, strand): region_id}
    region_counter = 0

    # Step 2: Build the relational mapping
    territory_mapping = {
        "regions": {},  # region_id -> {chr, start, end, strand, length}
        "subset_territories": defaultdict(
            lambda: {"unique": [], "comparator": []}
        ),  # subset_id -> {unique: [region_ids], comparator: [region_ids]}
        "relationships": [],  # List of {subset_id, superset_id, unique_regions, comparator_regions}
    }

    # Process all territories to find unique regions
    for subset_id, events in final_rescue_map.items():
        subset_info = transcript_info.get(subset_id)
        if not subset_info:
            continue

        strand = subset_info.get("strand", ".")

        for event in events:
            # Process unique region
            try:
                coords_str = event["unique_coords"]
                chrom, coords = coords_str.split(":")
                start, end = coords.split("-")
                start, end = int(start), int(end)
                if event["type"] == "retained_intron":
                    start = start + INTRON_BOUNDARY_TRIM
                    end = end - INTRON_BOUNDARY_TRIM
                else:
                    start = start + TERMINAL_TRIM
                    end = end - TERMINAL_TRIM
                if end - start < 10:
                    continue

                region_key = (chrom, start, end, strand)

                # Add to unique regions if not seen
                if region_key not in unique_regions:
                    unique_regions[region_key] = region_counter
                    territory_mapping["regions"][region_counter] = {
                        "chr": chrom,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "length": end - start,
                        "type": "unique",
                    }
                    region_counter += 1

                # Add to subset's unique territories
                region_id = unique_regions[region_key]
                if (
                    region_id
                    not in territory_mapping["subset_territories"][subset_id]["unique"]
                ):
                    territory_mapping["subset_territories"][subset_id]["unique"].append(
                        region_id
                    )

            except (ValueError, KeyError) as e:
                logging.warning(f"Skipping malformed unique region: {e}")

            # Process comparator regions
            comparator_list = event.get("comparator_coords", [])
            if isinstance(comparator_list, str):
                comparator_list = [comparator_list]

            comparator_region_ids = []
            for coords_str in comparator_list:
                try:
                    chrom, coords = coords_str.split(":")
                    start, end = coords.split("-")
                    start, end = int(start), int(end)

                    region_key = (chrom, start, end, strand)

                    # Add to unique regions if not seen
                    if region_key not in unique_regions:
                        unique_regions[region_key] = region_counter
                        territory_mapping["regions"][region_counter] = {
                            "chr": chrom,
                            "start": start,
                            "end": end,
                            "strand": strand,
                            "length": end - start,
                            "type": "comparator",
                        }
                        region_counter += 1

                    region_id = unique_regions[region_key]
                    if (
                        region_id
                        not in territory_mapping["subset_territories"][subset_id][
                            "comparator"
                        ]
                    ):
                        territory_mapping["subset_territories"][subset_id][
                            "comparator"
                        ].append(region_id)
                    comparator_region_ids.append(region_id)

                except (ValueError, KeyError) as e:
                    logging.warning(f"Skipping malformed comparator region: {e}")

            # Add relationship record
            if "superset_id" in event:
                # Need to look up the unique region ID correctly
                unique_coords_str = event["unique_coords"]
                unique_chrom, unique_coords = unique_coords_str.split(":")
                unique_start, unique_end = unique_coords.split("-")
                unique_start, unique_end = int(unique_start), int(unique_end)

                unique_region_key = (unique_chrom, unique_start, unique_end, strand)
                unique_region_id = unique_regions.get(unique_region_key)

                territory_mapping["relationships"].append(
                    {
                        "subset_id": subset_id,
                        "superset_id": event["superset_id"],
                        "event_type": event.get("type", "unknown"),
                        "unique_region": unique_region_id,
                        "comparator_regions": comparator_region_ids,
                    }
                )

    # Step 3: Write BED file
    bed_lines = []

    def genome_sort_key(item):
        region_key = item[0]  # The tuple (chrom, start, end, strand)
        chrom = region_key[0]
        start_pos = region_key[1]

        # Use the rank from our learned map. Default to a high number for unknown contigs.
        rank = chrom_rank_map.get(chrom, float("inf"))

        return (rank, start_pos)

    sorted_regions = sorted(unique_regions.items(), key=genome_sort_key)

    for (chrom, start, end, strand), region_id in sorted_regions:
        bed_line = f"{chrom}\t{start}\t{end}\tREGION_{region_id}\t0\t{strand}"
        bed_lines.append(bed_line)

    with open(output_bed_path, "w") as f:
        f.write("\n".join(bed_lines) + "\n")

    logging.info(f"  Wrote {len(bed_lines)} unique regions to GENOME-SORTED BED file")
    with open(output_bed_path, "w") as f:
        for line in bed_lines:
            f.write(line + "\n")

    logging.info(f"  Wrote {len(bed_lines)} unique regions to BED file")

    # Convert defaultdicts to regular dicts for JSON serialization
    territory_mapping["subset_territories"] = dict(
        territory_mapping["subset_territories"]
    )

    # Step 4: Save mapping if requested
    if output_mapping_path:
        with open(output_mapping_path, "w") as f:
            json.dump(territory_mapping, f, indent=2)
        logging.info(f"  Saved territory mapping to {output_mapping_path}")

    # Add summary statistics
    territory_mapping["stats"] = {
        "total_regions": len(unique_regions),
        "total_subsets": len(territory_mapping["subset_territories"]),
        "total_relationships": len(territory_mapping["relationships"]),
    }

    return territory_mapping


def build_comprehensive_annotation(
    gene_gtf, te_gtf, output_prefix, rescue_data=True, genome_file_path=None
):
    """Build complete annotation data structure with TSL and expected junctions."""

    # Parse GTFs (now includes TSL)
    gene_transcript_info, gene_to_transcripts, gene_transcript_exons, gene_chroms = (
        parse_complete_gtf(gene_gtf, "gene")
    )

    te_transcript_info = {}
    te_to_transcripts = {}
    te_transcript_exons = {}

    te_chroms = []
    if te_gtf:
        te_transcript_info, te_to_transcripts, te_transcript_exons, te_chroms = (
            parse_complete_gtf(te_gtf, "te")
        )

    # Check for duplicates
    gene_ids = set(gene_transcript_info.keys())
    te_ids = set(te_transcript_info.keys())
    duplicates = gene_ids.intersection(te_ids)

    if rescue_data:
        # Validate genomefile contains all GTF chroms
        all_gtf_chroms = set(gene_chroms).union(set(te_chroms))
        chrom_order = get_chrom_order_from_file(genome_file_path)
        authoritative_chrom_set = set(chrom_order)
        chroms_in_gtf_but_not_genome = all_gtf_chroms.difference(
            authoritative_chrom_set
        )
        if chroms_in_gtf_but_not_genome:
            # If the difference set is not empty, we have a mismatch.
            logging.warning("=" * 80)
            logging.warning("WARNING: Chromosome Mismatch Detected!")
            logging.warning(
                f"The following {len(chroms_in_gtf_but_not_genome)} chromosome(s) were found in your GTF file(s) but are NOT present"
            )
            logging.warning(
                f"in your authoritative genome file ('{os.path.basename(genome_file_path)}')."
            )
            logging.warning(
                "Any transcripts or features on these chromosomes will be effectively ignored by the aligner"
            )
            logging.warning(
                "and downstream tools like bedtools. This can lead to unexpected results."
            )

            # Print a manageable list of the problematic chromosomes.
            # If there are too many, just show the first 10.
            preview_list = sorted(list(chroms_in_gtf_but_not_genome))
            for chrom in preview_list[:10]:
                logging.warning(f"  - {chrom}")
            if len(preview_list) > 10:
                logging.warning(f"  - ... and {len(preview_list) - 10} more.")

            logging.warning(
                "Please ensure your GTF and genome files are derived from the exact same reference assembly."
            )
            logging.warning("=" * 80)
        else:
            logging.info(
                "✓ Chromosome name consistency check passed. All GTF chromosomes are present in the genome file."
            )

    if duplicates:
        logging.error(f"Found {len(duplicates)} duplicate transcript IDs between GTFs!")
        logging.error(f"Examples: {list(duplicates)[:5]}")
        raise ValueError("Duplicate transcript IDs found")

    # Combine all transcript IDs
    all_transcript_ids = gene_ids.union(te_ids)
    logging.info(f"Total unique transcript IDs: {len(all_transcript_ids)}")

    # Create integer mappings
    string_to_int, int_to_string = create_integer_mappings(all_transcript_ids)

    # Combine transcript info
    all_transcript_info = {}
    all_transcript_info.update(gene_transcript_info)
    all_transcript_info.update(te_transcript_info)

    # Combine exon data
    all_transcript_exons = {}
    all_transcript_exons.update(gene_transcript_exons)
    all_transcript_exons.update(te_transcript_exons)

    # Compute junction data
    junction_map, transcript_unique_junctions = compute_junction_data(
        all_transcript_exons, string_to_int
    )

    # Create transcript junction order
    transcript_junction_order = create_transcript_junction_order(junction_map)

    # Calculate expected junctions for completeness scoring
    transcript_expected_junctions = calculate_transcript_expected_junctions(
        all_transcript_exons, string_to_int
    )

    # Build splice site competition map
    junction_competitors = build_splice_site_competition_map(junction_map.keys())

    # Create annotation maps
    gene_map, te_map = create_annotation_maps(
        gene_transcript_info, te_transcript_info, string_to_int
    )

    # Convert transcript info to use integer IDs and include TSL
    transcript_info_int = {}
    tsl_distribution = defaultdict(int)

    for tid, info in all_transcript_info.items():
        tid_int = string_to_int[tid]
        transcript_info_int[tid_int] = {
            "gene_id": info["gene_id"],
            "chr": info["chr"],
            "strand": info["strand"],
            "start": info["start"],
            "end": info["end"],
            "transcript_length": info["transcript_length"],
            "n_exons": len(info["exons"]),
            "feature_type": info["feature_type"],
            "tsl": info["tsl"],  # Include TSL
        }
        tsl_distribution[info["tsl"]] += 1

    # Create subset/superset relationships
    subset_relationships = identify_subset_superset_relationships_parallel(
        junction_map, transcript_info_int
    )

    logging.info("Collecting all unique junction strings for integer encoding...")
    master_junction_set = set()
    # collect all the junction strings from multiple sources for int mapping
    # 1. from junction_map keys
    master_junction_set.update(junction_map.keys())

    # 2. from transcript_junction_order values
    for junction_list in transcript_junction_order.values():
        master_junction_set.update(junction_list)

    # 3. from transcript_unique_junctions values
    for junction_list in transcript_unique_junctions.values():
        master_junction_set.update(junction_list)

    # 4. from transcript_expected_junctions values
    for data in transcript_expected_junctions.values():
        master_junction_set.update(data["junction_keys"])

    # 5. from subset_relationships values
    for data in subset_relationships.values():
        master_junction_set.update(data["shared_junctions"])

    # 6. from junction_competitors keys and values
    for junction_key, competitor_set in junction_competitors.items():
        master_junction_set.add(junction_key)
        master_junction_set.update(competitor_set)

    junction_string_to_int, junction_int_to_string = create_integer_mappings(
        master_junction_set
    )

    # Convert junction_map
    junction_map_int = {junction_string_to_int[k]: v for k, v in junction_map.items()}

    # Convert transcript_junction_order
    transcript_junction_order_int = {
        tid: [junction_string_to_int[j] for j in junctions]
        for tid, junctions in transcript_junction_order.items()
    }

    # Convert transcript_unique_junctions
    transcript_unique_junctions_int = {
        tid: [junction_string_to_int[j] for j in junctions]
        for tid, junctions in transcript_unique_junctions.items()
    }

    # Convert transcript_expected_junctions (requires a loop for clarity)
    transcript_expected_junctions_int = {}
    for tid, data in transcript_expected_junctions.items():
        transcript_expected_junctions_int[tid] = {
            "n_expected": data["n_expected"],
            "junction_keys": [junction_string_to_int[j] for j in data["junction_keys"]],
        }

    # Convert subset_relationships (requires a loop for clarity)
    subset_relationships_int = {}
    for tid, data in subset_relationships.items():
        subset_relationships_int[tid] = {
            "supersets": data["supersets"],
            "shared_junctions": [
                junction_string_to_int[j] for j in data["shared_junctions"]
            ],
            "coverage_fraction": data["coverage_fraction"],
            "unique_junction_deficit": data["unique_junction_deficit"],
        }

    # Convert junction_competitors
    junction_competitors_int = {
        junction_string_to_int[k]: {
            junction_string_to_int[v] for v in competitor_set
        }  # <-- Changed to square brackets
        for k, competitor_set in junction_competitors.items()
    }
    logging.info("  Conversion complete.")

    logging.info(
        f"  Found {len(master_junction_set):,} unique junction strings to be encoded."
    )
    # Calculate checksums
    gene_checksum = calculate_gtf_checksum(gene_gtf)
    te_checksum = calculate_gtf_checksum(te_gtf) if te_gtf else None

    # block for generating rescue features
    if rescue_data:
        logging.info(
            "Starting process to identify unique exonic territories and comparator regions..."
        )
        candidate_map = characterize_unique_exonic_territory_parallel(
            subset_relationships_int,
            gene_transcript_exons,
            int_to_string,
            min_unique_size=30,
        )
        if candidate_map:
            logging.info(
                f"  Identified {len(candidate_map)} candidate subsets with potential unique territory."
            )
            logging.info("  Building interval trees for definitive overlap checking...")
            # Build interval trees for all exons
            interval_trees = build_interval_trees(gene_transcript_exons, string_to_int)
            logging.info(
                "  Performing definitive overlap checking to confirm unambiguous territory..."
            )
            final_rescue_map = find_unambiguous_territory(
                candidate_map, interval_trees, transcript_info_int
            )
            if final_rescue_map:
                logging.info(
                    f"  Confirmed {len(final_rescue_map)} subsets with unambiguous unique territory after overlap checking."
                )
                # Write to bed
                subset_coverage_bed_path = (
                    f"{output_prefix}_subset_coverage_features.bed"
                )
                subset_coverage_territory_mapping = (
                    write_territories_to_bed_with_mapping(
                        final_rescue_map,
                        transcript_info_int,
                        subset_coverage_bed_path,
                        chrom_order,
                    )
                )
            else:
                logging.info(
                    "  No subsets confirmed with unambiguous unique territory after overlap checking."
                )
        else:
            logging.info(
                "  No candidate subsets identified with potential unique territory."
            )
    else:
        logging.info(
            "Skipping unique exonic territory and comparator region generation as per user request."
        )

    # Build complete annotation object
    annotation_data = {
        # Core mappings
        "string_to_int": string_to_int,
        "int_to_string": int_to_string,
        "junction_string_to_int": junction_string_to_int,
        "junction_int_to_string": junction_int_to_string,
        #'transcript_to_exon_map': dict(all_transcript_exons), #temporary add for debugging
        # Transcript information (now includes TSL)
        "transcript_info": transcript_info_int,
        # Expected junctions for completeness scoring
        "transcript_expected_junctions": transcript_expected_junctions_int,
        # Gene/TE mappings (as numpy arrays for efficiency)
        "gene_map": {
            "LocusID": gene_map["LocusID"].values,
            "AggregateID": gene_map["AggregateID"].values,
        },
        "te_map": (
            {
                "LocusID": te_map["LocusID"].values if te_map is not None else None,
                "AggregateID": (
                    te_map["AggregateID"].values if te_map is not None else None
                ),
            }
            if te_map is not None
            else None
        ),
        # Junction data
        "junction_map": junction_map_int,
        "transcript_junction_order": transcript_junction_order_int,
        "transcript_unique_junctions": transcript_unique_junctions_int,
        "subset_relationships": subset_relationships_int,
        "junction_competitors": junction_competitors_int,
        # subset coverage data
        "subset_coverage_features": (
            final_rescue_map if rescue_data and final_rescue_map else None
        ),
        "subset_coverage_territory_mapping": (
            subset_coverage_territory_mapping
            if rescue_data and final_rescue_map
            else None
        ),
        # Metadata
        "metadata": {
            "gene_gtf": os.path.abspath(gene_gtf),
            "te_gtf": os.path.abspath(te_gtf) if te_gtf else None,
            "subset_coverage_features_bed_path": (
                os.path.abspath(subset_coverage_bed_path)
                if rescue_data and final_rescue_map
                else None
            ),
            "genome_file_path": (
                os.path.abspath(genome_file_path) if genome_file_path else None
            ),
            "gene_checksum": gene_checksum,
            "te_checksum": te_checksum,
            "creation_date": datetime.now().isoformat(),
            "n_transcripts": len(all_transcript_ids),
            "n_genes": len(gene_to_transcripts),
            "n_te_families": len(te_to_transcripts) if te_gtf else 0,
            "n_junctions": len(junction_map),
            "n_unique_junctions": sum(
                1 for j in junction_map.values() if j["is_unique"]
            ),
            "tsl_distribution": dict(tsl_distribution),
            "has_tsl_data": any(tsl != "NA" for tsl in tsl_distribution.keys()),
        },
    }

    # Save compressed pickle
    pickle_file = f"{output_prefix}_annotations.pkl.gz"
    logging.info(f"Saving annotation data to {pickle_file}")

    with gzip.open(pickle_file, "wb", compresslevel=6) as f:
        pickle.dump(annotation_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save JSON summary (updated with new info)
    json_file = f"{output_prefix}_annotations_summary.json"
    summary = {
        "metadata": annotation_data["metadata"],
        "statistics": {
            "total_transcripts": len(all_transcript_ids),
            "gene_transcripts": len(gene_ids),
            "te_transcripts": len(te_ids),
            "tsl_distribution": dict(tsl_distribution),
            "has_tsl_annotations": annotation_data["metadata"]["has_tsl_data"],
            "expected_junctions": {
                "single_exon_transcripts": sum(
                    1
                    for v in transcript_expected_junctions.values()
                    if v["n_expected"] == 0
                ),
                "multi_exon_transcripts": sum(
                    1
                    for v in transcript_expected_junctions.values()
                    if v["n_expected"] > 0
                ),
                "avg_junctions_per_transcript": np.mean(
                    [v["n_expected"] for v in transcript_expected_junctions.values()]
                ),
            },
            "unique_junctions_per_transcript": {
                "mean": (
                    np.mean([len(j) for j in transcript_unique_junctions.values()])
                    if transcript_unique_junctions
                    else 0
                ),
                "max": (
                    max([len(j) for j in transcript_unique_junctions.values()])
                    if transcript_unique_junctions
                    else 0
                ),
                "transcripts_with_unique": len(transcript_unique_junctions),
            },
        },
    }

    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"\nPreprocessing complete!")
    logging.info(f"  Annotation file: {pickle_file}")
    logging.info(f"  Summary file: {json_file}")
    logging.info(f"  File size: {os.path.getsize(pickle_file) / 1024 / 1024:.1f} MB")
    logging.info(f"  TSL data available: {annotation_data['metadata']['has_tsl_data']}")

    return annotation_data


def validate_annotation_file(annotation_file, gene_gtf, te_gtf=None):
    """Validate that annotation file matches current GTFs."""
    logging.info("Validating annotation file...")

    with gzip.open(annotation_file, "rb") as f:
        data = pickle.load(f)

    metadata = data["metadata"]

    # Check checksums
    current_gene_checksum = calculate_gtf_checksum(gene_gtf)
    if metadata["gene_checksum"] != current_gene_checksum:
        logging.error("Gene GTF has changed since annotation was created!")
        return False

    if te_gtf:
        current_te_checksum = calculate_gtf_checksum(te_gtf)
        if metadata["te_checksum"] != current_te_checksum:
            logging.error("TE GTF has changed since annotation was created!")
            return False

    logging.info("  Validation passed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute all annotation data for MAJEC pipeline"
    )

    parser.add_argument("--gene_gtf", required=True, help="Gene annotation GTF file")
    parser.add_argument("--te_gtf", help="TE annotation GTF file (optional)")
    parser.add_argument(
        "--output", default="majec", help="Output prefix for generated files"
    )
    parser.add_argument("--validate", help="Validate existing annotation file")
    parser.add_argument(
        "--generate_rescue_features",
        action="store_true",
        help="Perform advanced analysis to identify unique exon regions for subset rescue/penalty. "
        "Requires the 'intervaltree' library.",
    )
    parser.add_argument(
        "--genome_file_path",
        help="Path to a plain text file listing chromosome/contig names and sizes, one per line, "
        "in the exact sort order used by the BAM files. "
        "(For STAR users, this is the 'chrNameLength.txt' file in your genome index directory)."
        "Required if --generate_rescue_features is set.",
    )

    args = parser.parse_args()
    if args.generate_rescue_features and not args.genome_file_path:
        parser.error(
            "--genome_file_path is required when --generate_rescue_features is set."
        )
        sys.exit(1)

    # Mode 1: Validate existing annotation
    if args.validate:
        if validate_annotation_file(args.validate, args.gene_gtf, args.te_gtf):
            logging.info("Annotation file is valid!")
        else:
            logging.error("Annotation file validation failed!")
            sys.exit(1)
        return
    setup_logging()

    logging.info("building new annotation...")
    # Mode 2: Build new annotation
    annotation_data = build_comprehensive_annotation(
        args.gene_gtf,
        args.te_gtf,
        args.output,
        args.generate_rescue_features,
        args.genome_file_path,
    )

    logging.info("\nUsage with MAJEC pipeline:")
    logging.info(f"  python MAJEC.py --annotation {args.output}_annotations.pkl.gz ...")


if __name__ == "__main__":
    main()
