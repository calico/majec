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
            family_match = re.search(r'family_id "([^"]+)"', attributes)
            class_match = re.search(r'class_id "([^"]+)"', attributes)

            if not transcript_match:
                continue

            transcript_id = transcript_match.group(1).strip()
            gene_id = gene_match.group(1).strip() if gene_match else transcript_id
            family_id = family_match.group(1).strip() if family_match else None
            class_id = class_match.group(1).strip() if class_match else None

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
                        "family_id": family_id,
                        "class_id": class_id,
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
                    "family_id": family_id,
                    "class_id": class_id,
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


def compute_unique_territory(transcript_info):
    """
    Compute genome-wide unique exonic territory per transcript using an event-based
    sweep line algorithm. A base is "unique" to a transcript if no other transcript
    in the annotation has an exon covering that base (strand-aware).

    Args:
        transcript_info: {transcript_id: {chr, strand, exons: [(start, end), ...], ...}}
            Exon coordinates are 1-based inclusive (GTF convention).

    Returns:
        {transcript_id: unique_bp_count}
    """
    logging.info("Computing unique exonic territory per transcript...")

    # Group exon intervals by (chrom, strand)
    groups = defaultdict(list)
    for tid, info in transcript_info.items():
        key = (info["chr"], info["strand"])
        for start, end in info["exons"]:
            # Convert to half-open [start, end+1) for sweep
            groups[key].append((start, end + 1, tid))

    unique_bp = defaultdict(int)

    for (chrom, strand), intervals in groups.items():
        # Build events: (position, event_type, transcript_id)
        # event_type: 0 = start (exon begins), 1 = end (exon ends)
        events = []
        for start, end, tid in intervals:
            events.append((start, 0, tid))
            events.append((end, 1, tid))
        events.sort()

        # Sweep: track active transcripts at each position
        active = defaultdict(int)  # tid -> count of overlapping exons
        n_active = 0  # number of distinct active transcripts
        prev_pos = None

        for pos, event_type, tid in events:
            # Count bases since last event
            if prev_pos is not None and pos > prev_pos:
                if n_active == 1:
                    sole_tid = next(iter(active))
                    unique_bp[sole_tid] += pos - prev_pos

            # Process event
            if event_type == 0:  # exon start
                if active[tid] == 0:
                    n_active += 1
                active[tid] += 1
            else:  # exon end
                active[tid] -= 1
                if active[tid] == 0:
                    del active[tid]
                    n_active -= 1

            prev_pos = pos

    n_with_unique = sum(1 for v in unique_bp.values() if v > 0)
    total_unique_bp = sum(unique_bp.values())
    logging.info(
        f"  {n_with_unique:,} / {len(transcript_info):,} transcripts have unique territory"
    )
    logging.info(f"  Total unique territory: {total_unique_bp:,} bp")

    return dict(unique_bp)


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
            te_name = info["gene_id"]
            family_id = info.get("family_id")
            class_id = info.get("class_id")
            if family_id and class_id:
                te_aggregate_id = f"{te_name}:{family_id}:{class_id}"
            else:
                te_aggregate_id = te_name
            te_map_data.append([tid_int, te_aggregate_id])
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


def build_comprehensive_annotation(gene_gtf, te_gtf, output_prefix):
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

    # Compute unique exonic territory (genome-wide, strand-aware)
    # Protein-coding only for now — TE overlaps handled by EM, not floors
    unique_territory = compute_unique_territory(gene_transcript_info)

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
            "tsl": info["tsl"],
            "unique_territory_bp": unique_territory.get(tid, 0),
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

    # Build EqC builder data structures (ExonIntervalIndex + junction_unstranded)
    from majec.eqc_builder import (
        build_exon_index_from_annotation,
        build_junction_unstranded_lookup,
    )

    logging.info("Building EqC builder data structures for annotation...")
    exon_index = build_exon_index_from_annotation(all_transcript_exons, string_to_int)
    junction_unstranded = build_junction_unstranded_lookup(
        junction_map_int, junction_int_to_string
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
        # EqC builder data (precomputed for runtime)
        "exon_index": exon_index,
        "junction_unstranded": junction_unstranded,
        # Metadata
        "metadata": {
            "gene_gtf": os.path.abspath(gene_gtf),
            "te_gtf": os.path.abspath(te_gtf) if te_gtf else None,
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

    args = parser.parse_args()

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
        args.gene_gtf, args.te_gtf, args.output
    )

    logging.info("\nUsage with MAJEC pipeline:")
    logging.info(
        f"  majec_run_pipeline --annotation {args.output}_annotations.pkl.gz --bams ... --prefix ..."
    )


if __name__ == "__main__":
    main()
