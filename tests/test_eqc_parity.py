"""
Equivalence of the Rust and pure-Python EqC builders.

MAJEC keeps a pysam implementation alongside the Rust one, and
eqc_builder_rust.run_junction_eqc_builder silently delegates to it whenever
majec_core will not import. That fallback is only worth having if it agrees
with the fast path, and until this module existed nothing checked that -
parity was maintained by hand, commit by commit.

Two committed fixtures, both in tests/data/:

  * pseudogene_sim_sample01.bam - the main one. A polyester simulation of 12
    parent/pseudogene pairs, built by make_fixture.py. Pseudogenes are the
    useful part: they put near-identical sequence at two loci, which is what
    forces multimapper classes and alignment-score weighting to do real work.
    Everything else in a BAM is comparatively easy to agree on. Paired-end.
  * demo_singleend.bam - 19 KB, from the repo's own demo. It reaches only 5 of
    the 13 branches and has no multimappers at all, so it is not a substitute;
    it is here because it is single-end, and the single-end path skips the
    mate buffer and the pair intersection entirely. Nothing else covers it.

Deliberately excluded from the equality checks:

  * exact float equality on accumulated weights. Both sides sum the same
    values in a different order, which lands ~1e-14 apart, so weights go
    through pytest.approx and integer read counts do not.
"""

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / 'data'
BAM = DATA / 'pseudogene_sim_sample01.bam'
ANNOTATION = DATA / 'pseudogene_sim_annotation.pkl.gz'
TRUTH = DATA / 'pseudogene_sim_truth.json'

SE_BAM = DATA / 'demo_singleend.bam'
SE_ANNOTATION = DATA / 'demo_singleend_annotation.pkl.gz'

from tests.eqc_parity_common import (REQUIRED_BRANCHES,  # noqa: E402
                                     SHARED_STATS, assert_payloads_match)

majec_core = pytest.importorskip('majec_core', reason='Rust extension not built')
pytest.importorskip('pysam', reason='pysam not installed; fallback cannot run')

pytestmark = pytest.mark.skipif(
    not BAM.exists(), reason='fixture missing; run tests/data/make_fixture.py')


@pytest.fixture(scope='module')
def annotation():
    with gzip.open(ANNOTATION, 'rb') as f:
        return pickle.load(f)


def _run(impl, ann, **overrides):
    kwargs = dict(strandedness=0, paired_end=True, min_mapq=30,
                  use_alignment_scoring=False, as_temperature=10.0)
    kwargs.update(overrides)
    return impl(
        str(BAM), ann['exon_index'], ann['junction_unstranded'],
        ann['string_to_int'], ann['junction_map'],
        ann['junction_int_to_string'], ann['junction_string_to_int'],
        sample_name='parity', **kwargs)


@pytest.fixture(scope='module')
def results(annotation):
    """Both builders, both scoring modes. Four scans, so build them once."""
    from majec.eqc_builder import run_junction_eqc_builder as py_impl
    from majec.eqc_builder_rust import run_junction_eqc_builder as rust_impl

    out = {}
    for scoring in (False, True):
        out[scoring] = (
            _run(rust_impl, annotation, use_alignment_scoring=scoring),
            _run(py_impl, annotation, use_alignment_scoring=scoring),
        )
    return out


SCORING = pytest.mark.parametrize(
    'scoring', [False, True], ids=['plain', 'as_weighted'])


@SCORING
def test_unique_mapper_groups_identical(results, scoring):
    """Unique classes are integer read counts, so they must match exactly."""
    rust, py = results[scoring]
    assert dict(rust['unique_mapper_groups']) == dict(py['unique_mapper_groups'])


@SCORING
def test_multimapper_classes_identical(results, scoring):
    rust, py = results[scoring]
    r, p = rust['multimapper_classes'], py['multimapper_classes']
    assert set(r) == set(p), 'multimapper class membership differs'
    for key in r:
        assert r[key] == pytest.approx(p[key], rel=1e-9), f'weight differs for {set(key)}'


@SCORING
@pytest.mark.parametrize('field', ['initial_unique_counts',
                                   'initial_multimapper_counts'])
def test_count_vectors_match(results, scoring, field):
    rust, py = results[scoring]
    np.testing.assert_allclose(rust[field].values, py[field].values,
                               rtol=1e-9, atol=1e-12)


@SCORING
def test_junction_metrics_identical(results, scoring):
    rust, py = results[scoring]
    assert rust['junction_metrics'] == py['junction_metrics']


@SCORING
@pytest.mark.parametrize('key', SHARED_STATS)
def test_shared_stats_match(results, scoring, key):
    rust, py = results[scoring]
    r, p = rust['stats'].get(key), py['stats'].get(key)
    if isinstance(r, float) or isinstance(p, float):
        assert r == pytest.approx(p, rel=1e-9)
    else:
        assert r == p


@SCORING
@pytest.mark.parametrize('branch', REQUIRED_BRANCHES)
def test_fixture_exercises_branch(results, scoring, branch):
    """The fixture must keep reaching every branch, or parity means less."""
    rust, _ = results[scoring]
    assert rust['stats'].get(branch, 0) > 0, f'{branch} no longer exercised'


@SCORING
def test_fixture_exercises_low_mapq(results, scoring):
    """The one branch no unedited STAR BAM reaches.

    make_fixture.py clones a real NH:i:1 proper pair down to MAPQ 10 to get
    here; if that clone is ever dropped, this is what notices.
    """
    rust, _ = results[scoring]
    assert rust['stats'].get('low_mapq_unique', 0) > 0, (
        'low_mapq_unique no longer exercised - has the crafted MAPQ-10 pair '
        'been dropped from the fixture?')


def test_both_builders_measure_fragment_lengths(results):
    """The gap that used to be here is closed; this keeps it closed.

    Before the port only the Rust parser measured the distribution, so a
    Rust-less run silently lost empirical effective lengths and TPM fell back
    to annotated transcript lengths. assert_payloads_match now compares the
    histograms bin for bin; this asserts they are non-trivially populated, so
    that agreeing on "no measurement at all" cannot pass for parity.
    """
    for side in results[False]:
        hist = side['stats'].get('fragment_length_hist')
        assert hist, 'no fragment histogram produced'
        assert len(hist) > 100, f'only {len(hist)} distinct lengths measured'
        assert side['stats']['fragment_length_n'] > 1000


def test_fragment_length_is_measured_in_cdna_space(results):
    """The point of the estimator: introns must not be in the length.

    A genomic span includes them, so if the cDNA projection ever regressed to
    a plain end-minus-start the two would converge. On a spliced library they
    should be far apart.
    """
    stats = results[False][0]['stats']
    assert stats['genomic_span_mean'] > 2 * stats['fragment_length_unique_mean'], (
        f"genomic span {stats['genomic_span_mean']:.0f} is not meaningfully "
        f"larger than the spliced length "
        f"{stats['fragment_length_unique_mean']:.0f} - is the projection "
        f"still removing introns?")


def test_parents_outrank_their_pseudogenes(results, annotation):
    """Sanity check against the simulation's ground truth.

    Parity alone would be satisfied by two implementations that agree and are
    both wrong. Each pair was simulated with thousands of parent reads against
    single-digit pseudogene reads, so the parent must come out ahead.
    """
    truth = json.loads(TRUTH.read_text())
    s2i = annotation['string_to_int']
    counts = results[False][0]['initial_unique_counts']

    checked = 0
    for pair in truth['pairs_meta']:
        # _resolved because sim_transcripts.fa predates v44 and most parents
        # carry a stale version suffix; make_fixture.py maps them forward.
        parent, pseudo = pair['parent_tx_resolved'], pair['pseudo_tx_resolved']
        if not parent or not pseudo:
            continue
        assert pair['parent_sim_reads'] > pair['pseudo_sim_reads']
        assert counts.iloc[s2i[parent]] > counts.iloc[s2i[pseudo]], (
            f"{pair['parent_name']} should outrank {pair['pseudo_name']}")
        checked += 1
    assert checked == 12, f'expected all 12 pairs, resolved {checked}'


# --------------------------------------------------------------------------
# Single-end. Everything above runs paired-end, which means the mate buffer and
# the pair intersection are always in play. This is the only coverage of the
# path where they are not.
# --------------------------------------------------------------------------

se_only = pytest.mark.skipif(
    not SE_BAM.exists(), reason='demo single-end fixture missing')


@pytest.fixture(scope='module')
def se_results():
    from majec.eqc_builder import run_junction_eqc_builder as py_impl
    from majec.eqc_builder_rust import run_junction_eqc_builder as rust_impl

    with gzip.open(SE_ANNOTATION, 'rb') as f:
        ann = pickle.load(f)

    def run(impl, scoring):
        return impl(
            str(SE_BAM), ann['exon_index'], ann['junction_unstranded'],
            ann['string_to_int'], ann['junction_map'],
            ann['junction_int_to_string'], ann['junction_string_to_int'],
            sample_name='parity_se', strandedness=0, paired_end=False,
            min_mapq=30, use_alignment_scoring=scoring, as_temperature=10.0)

    return {s: (run(rust_impl, s), run(py_impl, s)) for s in (False, True)}


@se_only
@SCORING
def test_single_end_parity(se_results, scoring):
    rust, py = se_results[scoring]
    assert_payloads_match(rust, py, context='single-end')


@se_only
def test_single_end_skips_the_pairing_machinery(se_results):
    """Confirm this fixture is actually testing the single-end path.

    If it ever starts reporting paired counters, either the fixture was swapped
    for a paired BAM or paired_end=False stopped being honoured - and then the
    module has no single-end coverage while still looking like it does.
    """
    stats = se_results[False][0]['stats']
    assert stats['total_reads'] > 0
    for key in ('paired_reads_processed', 'paired_intersection_empty',
                'orphaned_mates'):
        assert stats.get(key, 0) == 0, (
            f'{key} fired on a single-end run: {stats[key]}')


@se_only
def test_single_end_still_refines_on_junctions(se_results):
    """The interesting half of the single-end path must still do work."""
    stats = se_results[False][0]['stats']
    assert stats['spliced_reads'] > 0
    assert stats['junction_refined'] > 0
    assert stats['exon_refined'] > 0
