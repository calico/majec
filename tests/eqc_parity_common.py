"""
Shared vocabulary for the two EqC-builder parity tiers.

test_eqc_parity.py runs against the committed 1.3 MB fixture on every
`pytest tests/`; test_eqc_parity_large.py runs against a multi-GB cluster BAM
under `-m slow`. Both compare the same fields the same way, so the field lists
and the comparison itself live here rather than being restated twice.

Not a test module - no importorskip, no collection.
"""

import numpy as np
import pytest

# Counters both builders maintain. Anything Rust-only is checked separately.
SHARED_STATS = [
    'total_reads', 'unique_mapped', 'multimapped', 'unmapped', 'no_feature',
    'low_mapq_unique', 'spliced_reads', 'junction_refined', 'junction_novel',
    'junction_incompatible', 'exon_refined', 'exon_incompatible',
    'unique_eqc_size_1', 'paired_reads_processed', 'paired_intersection_empty',
    'orphaned_mates', 'unique_eqc_count', 'multi_eqc_count',
    'total_unique_reads_in_eqc', 'total_multi_reads_in_eqc',
    'median_read_length',
    # Fragment length, ported to the pysam builder so both parsers measure it
    # in cDNA space during the same scan. Quantiles are exact integers; means
    # and sds are float sums over the same histogram in a different order.
    'fragment_length_n', 'fragment_length_mean', 'fragment_length_sd',
    'fragment_length_median', 'fragment_length_p10', 'fragment_length_p90',
    'fragment_length_p99',
    'fragment_length_capped_n', 'fragment_length_capped_mean',
    'fragment_length_capped_sd', 'fragment_length_capped_median',
    'fragment_length_capped_p10', 'fragment_length_capped_p90',
    'fragment_length_capped_p99',
    'fragment_length_unique_n', 'fragment_length_unique_mean',
    'fragment_length_unique_sd', 'fragment_length_unique_median',
    'fragment_length_unique_p10', 'fragment_length_unique_p90',
    'fragment_length_unique_p99',
    'genomic_span_n', 'genomic_span_mean', 'genomic_span_sd',
    'genomic_span_median', 'genomic_span_p10', 'genomic_span_p90',
    'genomic_span_p99',
    'fragment_span_not_in_exons',
]

# Branches a parity fixture has to reach. Asserted directly so that a future
# fixture edit cannot quietly stop exercising one and leave the parity
# assertions comparing two implementations over an easier BAM.
#
# low_mapq_unique is excluded: STAR only hands out low MAPQ to multi-locus
# reads, which are routed to the multimapper accumulator before the MAPQ test,
# so no unedited STAR BAM reaches it. The committed fixture adds it back with a
# crafted record and asserts it separately.
REQUIRED_BRANCHES = [
    'total_reads', 'unique_mapped', 'multimapped', 'no_feature',
    'spliced_reads', 'junction_refined', 'junction_novel',
    'junction_incompatible', 'exon_refined', 'exon_incompatible',
    'paired_intersection_empty', 'orphaned_mates',
]


def assert_payloads_match(rust, py, context=''):
    """Compare two builder payloads in full, as one call.

    The fast fixture asserts each field in its own test so a failure names the
    field directly. The slow test cannot afford a rescan of a multi-GB BAM just
    for that granularity, so it funnels through here instead.

    Float weights are compared approximately: both sides sum the same values in
    a different order, which lands them ~1e-14 apart under alignment-score
    weighting. Integer read counts are compared exactly.
    """
    where = f' [{context}]' if context else ''

    assert dict(rust['unique_mapper_groups']) == dict(py['unique_mapper_groups']), \
        f'unique_mapper_groups differ{where}'

    r_mm, p_mm = rust['multimapper_classes'], py['multimapper_classes']
    assert set(r_mm) == set(p_mm), f'multimapper class membership differs{where}'
    for key in r_mm:
        assert r_mm[key] == pytest.approx(p_mm[key], rel=1e-9), \
            f'multimapper weight differs for {set(key)}{where}'

    for field in ('initial_unique_counts', 'initial_multimapper_counts'):
        np.testing.assert_allclose(
            rust[field].values, py[field].values, rtol=1e-9, atol=1e-12,
            err_msg=f'{field} differs{where}')

    assert rust['junction_metrics'] == py['junction_metrics'], \
        f'junction_metrics differ{where}'

    r_fh = rust['stats'].get('fragment_length_hist')
    p_fh = py['stats'].get('fragment_length_hist')
    assert (r_fh is None) == (p_fh is None), \
        f'only one builder produced a fragment histogram{where}'
    if r_fh is not None:
        assert set(r_fh) == set(p_fh), \
            f'fragment histogram covers different lengths{where}'
        for length in r_fh:
            assert r_fh[length] == pytest.approx(p_fh[length], rel=1e-9), \
                f'fragment weight differs at length {length}{where}'

    for key in SHARED_STATS:
        r, p = rust['stats'].get(key), py['stats'].get(key)
        if isinstance(r, float) or isinstance(p, float):
            assert r == pytest.approx(p, rel=1e-9), \
                f'stats[{key}] differs{where}: rust={r} py={p}'
        else:
            assert r == p, f'stats[{key}] differs{where}: rust={r} py={p}'
