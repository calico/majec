#!/usr/bin/env python
"""
Build the committed EqC-builder test fixture.

Produces a BAM and a matching annotation small enough to live in the repo, from
a pseudogene simulation (24 transcripts, 12 parent/pseudogene pairs, polyester
reads, known ground truth).

The annotation is cut down two ways:

  * only the six keys run_junction_eqc_builder actually reads are kept
    (exon_index, string_to_int, junction_map, junction_int_to_string,
    junction_string_to_int, junction_unstranded), plus transcript_info and
    int_to_string so the fixture can also drive length/TPM assertions later;
  * only transcripts the BAM can actually reach are kept, and their integer
    ids are renumbered densely from zero.

The renumbering is not optional. The builder does
`n_features = len(string_to_int)` and then indexes a Series of that length by
transcript int, so a sparse id set would silently drop every feature whose
original id landed past the new length.

"Reachable" is the union of two sets, matching the two ways the builder can
reach a transcript from a read:

  * exon overlap - any transcript with an exon overlapping an aligned block,
    which is what query_blocks and query_blocks_contained look at;
  * junction membership - every transcript on any junction whose unstranded
    key appears as an N gap in the BAM, which is what junction refinement
    looks at.

Selecting on the reads themselves rather than on whole contigs is what keeps
the fixture small: the simulation puts reads at 24 loci, so whole-chromosome
subsetting would carry tens of thousands of transcripts that no read can touch.

Rerun with:
    python tests/data/make_fixture.py --sim-dir SIM_DIR --annotation ANNOTATION

SIM_DIR holds star_aligned/sample_01/Aligned.sortedByCoord.out.bam,
ground_truth_counts.tsv and pairs_meta.json; ANNOTATION is the gene-only 0.2.0
annotation the fixture is cut from.
"""

import gzip
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pysam

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from majec.eqc_builder import ExonIntervalIndex  # noqa: E402

# Set from the command line in main().
SIM = SRC_BAM = SRC_ANN = None

OUT = Path(__file__).resolve().parent
OUT_BAM = OUT / 'pseudogene_sim_sample01.bam'
OUT_ANN = OUT / 'pseudogene_sim_annotation.pkl.gz'
OUT_TRUTH = OUT / 'pseudogene_sim_truth.json'


def read_blocks_and_junctions(bam_path):
    """Aligned blocks and splice junctions actually present in the BAM.

    Blocks come back as 1-based inclusive to match the annotation's GTF-style
    coordinates; pysam hands them out 0-based half-open.
    """
    blocks = defaultdict(list)
    junctions = set()
    with pysam.AlignmentFile(str(bam_path), 'rb') as bam:
        for rec in bam:
            if rec.is_unmapped:
                continue
            chrom = rec.reference_name
            bl = [(s + 1, e) for s, e in rec.get_blocks()]
            blocks[chrom].extend(bl)
            # N gaps between consecutive blocks are the splice junctions the
            # builder will look up. Same donor/acceptor convention as
            # junction_unstranded's keys: chrom:lastbase_of_exon-firstbase_of_next.
            for (_, e1), (s2, _) in zip(bl, bl[1:]):
                if s2 > e1 + 1:
                    junctions.add(f"{chrom}:{e1}-{s2}")
    return blocks, junctions


def merge(intervals):
    """Merge overlapping 1-based inclusive intervals."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def build_bam():
    """Copy the simulation BAM, adding the one pair STAR will never produce.

    Every builder branch fires on the simulation as-is except low_mapq_unique,
    which needs a read that is uniquely assigned *and* below min_mapq. STAR
    only hands out low MAPQ to multi-locus reads, and those are routed to the
    multimapper accumulator before the MAPQ test is reached, so the branch is
    unreachable with STAR input alone.

    Rather than synthesise coordinates, clone a real NH:i:1 proper pair and
    lower its MAPQ - the clone keeps a valid CIGAR, real exon overlap and
    consistent mate coordinates. Both mates are lowered because the check runs
    on whichever mate completes the pair.
    """
    with pysam.AlignmentFile(str(SRC_BAM), 'rb') as src:
        header = src.header.to_dict()
        recs = list(src)

    by_name = defaultdict(list)
    for r in recs:
        by_name[r.query_name].append(r)

    donor = None
    for name, group in by_name.items():
        if len(group) != 2:
            continue
        a, b = group
        if (a.is_proper_pair and not a.is_secondary and not b.is_secondary
                and a.get_tag('NH') == 1 and b.get_tag('NH') == 1
                and a.mapping_quality >= 30 and b.mapping_quality >= 30):
            donor = (a, b)
            break
    if donor is None:
        raise RuntimeError("no clean NH:i:1 proper pair to clone")

    unsorted = OUT / '_unsorted.bam'
    with pysam.AlignmentFile(str(unsorted), 'wb', header=header) as out:
        for r in recs:
            out.write(r)
        for r in donor:
            clone = r.__copy__()
            clone.query_name = r.query_name + '_lowmapq'
            clone.mapping_quality = 10
            out.write(clone)

    # No .bai: both builders stream the file linearly (the Python one via
    # fetch(until_eof=True)), so an index would be 900 KB of committed weight
    # that nothing reads. Sorting is kept only so the fixture looks like real
    # aligner output.
    pysam.sort('-o', str(OUT_BAM), str(unsorted))
    unsorted.unlink()
    print(f"Wrote {OUT_BAM.name}  ({OUT_BAM.stat().st_size / 1e6:.2f} MB, "
          f"{len(recs) + 2:,} records; cloned {donor[0].query_name} at MAPQ 10)")


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Build the committed EqC-builder test fixture.")
    ap.add_argument('--sim-dir', required=True, type=Path,
                    help="pseudogene simulation directory (star_aligned/sample_01/"
                         "Aligned.sortedByCoord.out.bam, ground_truth_counts.tsv, pairs_meta.json)")
    ap.add_argument('--annotation', required=True, type=Path,
                    help="gene-only 0.2.0 annotation (*_annotations.pkl.gz) the fixture is cut from")
    return ap.parse_args()


def main():
    global SIM, SRC_BAM, SRC_ANN
    args = _parse_args()
    SIM = args.sim_dir
    SRC_BAM = SIM / 'star_aligned/sample_01/Aligned.sortedByCoord.out.bam'
    SRC_ANN = args.annotation
    print(f"Loading {SRC_ANN.name} ...")
    with gzip.open(SRC_ANN, 'rb') as f:
        ann = pickle.load(f)
    print(f"  {len(ann['string_to_int']):,} transcripts, "
          f"{ann['exon_index'].n_exons:,} exons, "
          f"{len(ann['junction_map']):,} junctions")

    print(f"Scanning {SRC_BAM} for reachable loci ...")
    blocks, bam_junctions = read_blocks_and_junctions(SRC_BAM)
    windows = {c: merge(v) for c, v in blocks.items()}
    n_win = sum(len(v) for v in windows.values())
    print(f"  {len(windows)} contigs, {n_win:,} merged windows, "
          f"{len(bam_junctions):,} distinct splice junctions")

    # --- reachable transcripts -------------------------------------------
    keep = set()

    # (a) exon overlap. Walk the source bins once per contig rather than
    # querying per window; the bin index is already keyed by 1kb bucket.
    ei = ann['exon_index']
    for (chrom, strand), bin_map in ei.bins.items():
        if chrom not in windows:
            continue
        wins = windows[chrom]
        want_bins = set()
        for ws, we in wins:
            want_bins.update(range(ws // ei.BIN_SIZE, we // ei.BIN_SIZE + 1))
        for b in want_bins:
            for ex_s, ex_e, tx in bin_map.get(b, ()):
                for ws, we in wins:
                    if ws <= ex_e and we >= ex_s:
                        keep.add(tx)
                        break

    print(f"  {len(keep):,} transcripts by exon overlap")

    # (b) junction membership
    before = len(keep)
    for key in bam_junctions:
        keep.update(ann['junction_unstranded'].get(key, ()))
    print(f"  +{len(keep) - before:,} by junction membership -> {len(keep):,} total")

    # --- dense renumbering ------------------------------------------------
    old_ids = sorted(keep)                       # sorted => deterministic fixture
    remap = {old: new for new, old in enumerate(old_ids)}
    int_to_string = [ann['int_to_string'][o] for o in old_ids]
    string_to_int = {s: i for i, s in enumerate(int_to_string)}

    # --- exon index -------------------------------------------------------
    new_ei = ExonIntervalIndex()
    for (chrom, strand), bin_map in ei.bins.items():
        seen = set()
        for b, intervals in bin_map.items():
            for ex_s, ex_e, tx in intervals:
                if tx not in remap:
                    continue
                # bins duplicate an exon across every bucket it spans
                k = (ex_s, ex_e, tx)
                if k in seen:
                    continue
                seen.add(k)
                new_ei.add_exon(chrom, ex_s, ex_e, strand, remap[tx])
    new_ei.finalize()
    print(f"  exon index: {new_ei.n_exons:,} exons")

    # --- junctions --------------------------------------------------------
    # Keep a junction if any of its transcripts survived, then drop the
    # transcripts that did not so the lists stay consistent with string_to_int.
    junc_old = [j for j, info in ann['junction_map'].items()
                if any(t in remap for t in info['transcripts'])]
    junc_old.sort()
    junc_remap = {o: n for n, o in enumerate(junc_old)}

    junction_int_to_string = [ann['junction_int_to_string'][o] for o in junc_old]
    junction_string_to_int = {s: i for i, s in enumerate(junction_int_to_string)}

    junction_map = {}
    for o in junc_old:
        info = ann['junction_map'][o]
        txs = sorted(remap[t] for t in info['transcripts'] if t in remap)
        junction_map[junc_remap[o]] = {
            'transcripts': txs,
            'is_unique': len(txs) == 1,
            'n_transcripts': len(txs),
        }

    junction_unstranded = {}
    for key, txs in ann['junction_unstranded'].items():
        new = {remap[t] for t in txs if t in remap}
        if new:
            junction_unstranded[key] = new
    print(f"  junctions: {len(junction_map):,} stranded, "
          f"{len(junction_unstranded):,} unstranded")

    transcript_info = {remap[o]: ann['transcript_info'][o]
                       for o in old_ids if o in ann['transcript_info']}

    fixture = {
        'exon_index': new_ei,
        'string_to_int': string_to_int,
        'int_to_string': int_to_string,
        'junction_map': junction_map,
        'junction_int_to_string': junction_int_to_string,
        'junction_string_to_int': junction_string_to_int,
        'junction_unstranded': junction_unstranded,
        'transcript_info': transcript_info,
        'metadata': {
            'derived_from': SRC_ANN.name,
            'source_bam': SRC_BAM.name,
            'source_transcripts': len(ann['string_to_int']),
            'note': 'Test fixture. Transcript ints are renumbered densely and '
                    'do NOT match the source annotation.',
        },
    }

    with gzip.open(OUT_ANN, 'wb', compresslevel=9) as f:
        pickle.dump(fixture, f, protocol=4)
    print(f"Wrote {OUT_ANN.name}  ({OUT_ANN.stat().st_size / 1e6:.2f} MB)")

    # --- ground truth -----------------------------------------------------
    # sim_transcripts.fa was cut from a GENCODE release older than the v44
    # annotation, so 11 of the 12 parent transcripts carry a stale version
    # suffix (ENST00000323345.10 vs .11). The reads are unaffected - they align
    # to the genome and land on the same locus either way - but a literal id
    # lookup misses them, so resolve on the unversioned accession.
    by_base = {}
    for s in string_to_int:
        by_base.setdefault(s.split('.')[0], s)

    def resolve(tx_id):
        if tx_id in string_to_int:
            return tx_id
        return by_base.get(tx_id.split('.')[0])

    truth = {}
    tsv = (SIM / 'ground_truth_counts.tsv').read_text().splitlines()
    for line in tsv[1:]:
        parts = line.split('\t')
        if len(parts) > 1:
            truth[parts[0]] = int(parts[1])       # sample_1 column

    pairs = json.loads((SIM / 'pairs_meta.json').read_text())
    n_resolved = 0
    for p in pairs:
        for tag in ('parent', 'pseudo'):
            got = resolve(p[f'{tag}_tx'])
            p[f'{tag}_tx_resolved'] = got
            n_resolved += got is not None and got != p[f'{tag}_tx']

    usable = sum(1 for p in pairs
                 if p['parent_tx_resolved'] and p['pseudo_tx_resolved'])
    OUT_TRUTH.write_text(json.dumps({
        'sample': 'sample_01',
        'simulated_reads_per_transcript': truth,
        'pairs_meta': pairs,
    }, indent=2))
    print(f"Wrote {OUT_TRUTH.name}  ({len(truth)} transcripts, "
          f"{n_resolved} ids version-corrected, {usable}/{len(pairs)} pairs usable)")

    build_bam()


if __name__ == '__main__':
    main()
