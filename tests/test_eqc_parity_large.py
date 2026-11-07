"""
Rust/Python EqC-builder parity at realistic scale.

The committed fixture in test_eqc_parity.py is 1.3 MB and 50k records. It
covers every branch, but it cannot catch anything that only emerges from
volume: accumulation drift over millions of weights, buffer growth in the
orphan-mate path, contigs the fixture has no reads on. This module runs the
same comparison over a real LongBench alignment, ~7.5M pairs against the full
v44 annotation.

It needs a large BAM and takes roughly 15 minutes, almost all of it the
pure-Python scan, so it is opt-in. Point it at a paired-end STAR BAM and a
matching 0.2.0 annotation:

    MAJEC_PARITY_BAM=/path/to/sample.bam \
    MAJEC_PARITY_ANNOTATION=/path/to/genes_annotations.pkl.gz \
    pytest tests/test_eqc_parity_large.py -m slow -s

MAJEC_PARITY_STRANDEDNESS sets the library strandedness (default 2, reverse-
stranded). The reference run used ~7.5M pairs of a reverse-stranded LongBench
library (SRR30947472, subsampled) against GENCODE v44.

(-s because the scans print progress; without it pytest looks hung.)

Replaces the old tests/test_rust_eqc.py, which was a top-level script rather
than a test module. That script loaded a 4.9 GB BAM and a 39 MB annotation at
import time, so `pytest tests/` hung during collection before running
anything, and it compared the Rust output against a *stale pickled cache* of a
Python run under a different annotation - so it logged differences instead of
asserting, and a real regression would have read as one more line of expected
drift. Both builders now run live on the same inputs, and disagreement is a
failure.
"""

import gzip
import os
import pickle
import sys
import time
from pathlib import Path

import pytest

from tests.eqc_parity_common import (REQUIRED_BRANCHES, SHARED_STATS,
                                     assert_payloads_match)

BAM = Path(os.environ['MAJEC_PARITY_BAM']) if os.environ.get('MAJEC_PARITY_BAM') else None
ANNOTATION = (Path(os.environ['MAJEC_PARITY_ANNOTATION'])
              if os.environ.get('MAJEC_PARITY_ANNOTATION') else None)

# MAJEC's convention is 2 for reverse-stranded (dUTP) libraries. The committed
# fixture runs unstranded, so this tier is also the only parity coverage of the
# strand-aware branch.
STRANDEDNESS = int(os.environ.get('MAJEC_PARITY_STRANDEDNESS', 2))

pytest.importorskip('majec_core', reason='Rust extension not built')
pytest.importorskip('pysam', reason='pysam not installed; fallback cannot run')

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (BAM and ANNOTATION and BAM.exists() and ANNOTATION.exists()),
        reason='set MAJEC_PARITY_BAM and MAJEC_PARITY_ANNOTATION to run this tier'),
]


def _log(msg):
    print(f'  {msg}', file=sys.stderr, flush=True)


@pytest.fixture(scope='module')
def results():
    """Both builders, both scoring modes, on the full BAM. Four scans."""
    from majec.eqc_builder import run_junction_eqc_builder as py_impl
    from majec.eqc_builder_rust import run_junction_eqc_builder as rust_impl

    t0 = time.time()
    with gzip.open(ANNOTATION, 'rb') as f:
        ann = pickle.load(f)
    _log(f'annotation: {len(ann["string_to_int"]):,} transcripts, '
         f'{ann["exon_index"].n_exons:,} exons ({time.time() - t0:.1f}s)')

    def run(impl, scoring):
        return impl(
            str(BAM), ann['exon_index'], ann['junction_unstranded'],
            ann['string_to_int'], ann['junction_map'],
            ann['junction_int_to_string'], ann['junction_string_to_int'],
            sample_name='parity_large', strandedness=STRANDEDNESS,
            paired_end=True, min_mapq=30, use_alignment_scoring=scoring,
            as_temperature=10.0)

    out = {}
    for scoring in (False, True):
        mode = 'as_weighted' if scoring else 'plain'
        for name, impl in (('rust', rust_impl), ('python', py_impl)):
            t = time.time()
            res = run(impl, scoring)
            dt = max(time.time() - t, 1e-9)
            n = res['stats']['total_reads']
            _log(f'{mode:12s} {name:6s} {n:,} reads in {dt:.0f}s '
                 f'({n / dt / 1000:.0f}K/s)')
            out[(scoring, name)] = res
    return out


SCORING = pytest.mark.parametrize(
    'scoring', [False, True], ids=['plain', 'as_weighted'])


@SCORING
def test_builders_agree_at_scale(results, scoring):
    mode = 'as_weighted' if scoring else 'plain'
    assert_payloads_match(results[(scoring, 'rust')],
                          results[(scoring, 'python')], context=mode)


@SCORING
def test_scale_exercises_branches(results, scoring):
    """A real BAM should reach at least what the tiny fixture reaches."""
    stats = results[(scoring, 'rust')]['stats']
    missing = [b for b in REQUIRED_BRANCHES if not stats.get(b, 0)]
    assert not missing, f'branches unexercised on the full BAM: {missing}'


def test_scale_is_actually_larger(results):
    """Guard against silently pointing this at a subsampled BAM.

    The whole justification for the slow tier is volume. If the path is ever
    repointed somewhere small, parity would still pass and the tier would stop
    earning its 15 minutes.
    """
    stats = results[(False, 'rust')]['stats']
    assert stats['total_reads'] > 1_000_000, (
        f'only {stats["total_reads"]:,} reads - this tier exists to test '
        'scale the committed fixture cannot reach')
    assert stats['multimapped'] > 100_000, (
        f'only {stats["multimapped"]:,} multimappers - the accumulation paths '
        'are what scale is meant to stress')


def test_both_builders_measure_fragment_lengths(results):
    """Companion to the fast tier's version; see test_eqc_parity.py.

    At this scale the distribution is also worth sanity-checking against the
    published validation: SRR30947472 is a standard Illumina paired library, so
    a mean in the low hundreds is expected. A projection bug that leaked
    introns would land in the thousands.
    """
    for name in ('rust', 'python'):
        stats = results[(False, name)]['stats']
        assert stats.get('fragment_length_hist'), f'{name}: no histogram'
        assert stats['fragment_length_n'] > 1_000_000, (
            f'{name}: only {stats["fragment_length_n"]:,.0f} pairs measured')
        assert 100 < stats['fragment_length_capped_mean'] < 600, (
            f'{name}: capped mean {stats["fragment_length_capped_mean"]:.1f} is '
            'outside the plausible range for an Illumina library')


def test_shared_stats_are_all_present(results):
    """Every field the comparison claims to check must actually exist.

    assert_payloads_match reads stats with .get(), so a key renamed on both
    sides at once would compare None to None and pass. This is what notices.
    """
    for name in ('rust', 'python'):
        stats = results[(False, name)]['stats']
        missing = [k for k in SHARED_STATS if k not in stats]
        assert not missing, f'{name} builder no longer reports: {missing}'
