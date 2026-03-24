"""Tests for refactored pipeline helper functions."""

import pytest
import numpy as np
import pandas as pd
from argparse import Namespace
from unittest.mock import patch

from majec.pipeline import (
    calculate_tpm,
    _combine_penalties,
    _aggregate_by_map,
    _aggregate_genes_and_tes,
    _apply_bulk_updates,
    _apply_junction_boosts,
    _apply_tsl_penalties,
    build_argument_parser,
    validate_args,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_counts():
    """A small counts DataFrame with 2 samples and 5 transcripts."""
    return pd.DataFrame(
        {'sample_A': [100.0, 200.0, 0.0, 50.0, 150.0],
         'sample_B': [80.0, 0.0, 300.0, 60.0, 10.0]},
        index=[0, 1, 2, 3, 4]
    )


@pytest.fixture
def gene_map():
    """Maps transcript integer IDs to gene names."""
    return pd.DataFrame({
        'LocusID': [0, 1, 2, 3, 4],
        'AggregateID': ['GeneA', 'GeneA', 'GeneB', 'GeneB', 'GeneC']
    })


@pytest.fixture
def te_map():
    """Maps transcript integer IDs to TE subfamilies (non-overlapping with gene_map)."""
    return pd.DataFrame({
        'LocusID': [10, 11],
        'AggregateID': ['L1_HS', 'L1_HS']
    })


@pytest.fixture
def length_vectors():
    """Per-sample length vectors aligned to simple_counts index."""
    return {
        'sample_A': pd.Series([1000.0, 2000.0, 1500.0, 500.0, 3000.0], index=[0, 1, 2, 3, 4]),
        'sample_B': pd.Series([1000.0, 2000.0, 1500.0, 500.0, 3000.0], index=[0, 1, 2, 3, 4]),
    }


# =============================================================================
# _combine_penalties
# =============================================================================

class TestCombinePenalties:
    def test_no_penalty(self):
        assert _combine_penalties(1.0, 1.0) == 1.0

    def test_symmetric(self):
        assert _combine_penalties(0.5, 0.8) == _combine_penalties(0.8, 0.5)

    def test_one_zero(self):
        result = _combine_penalties(0.0, 0.5)
        assert result == 0.0

    def test_both_half(self):
        # min(0.5, 0.5) = 0.5, product = 0.25, average = 0.375
        assert _combine_penalties(0.5, 0.5) == pytest.approx(0.375)

    def test_known_value(self):
        # min(0.3, 0.7) = 0.3, product = 0.21, average = 0.255
        assert _combine_penalties(0.3, 0.7) == pytest.approx(0.255)


# =============================================================================
# calculate_tpm
# =============================================================================

class TestCalculateTPM:
    def test_tpm_sums_to_one_million(self, simple_counts, length_vectors):
        tpm = calculate_tpm(simple_counts, length_vectors)
        for col in tpm.columns:
            assert tpm[col].sum() == pytest.approx(1_000_000, rel=1e-6)

    def test_zero_count_gives_zero_tpm(self, simple_counts, length_vectors):
        tpm = calculate_tpm(simple_counts, length_vectors)
        # Transcript 2 has 0 counts in sample_A
        assert tpm.loc[2, 'sample_A'] == 0.0
        # Transcript 1 has 0 counts in sample_B
        assert tpm.loc[1, 'sample_B'] == 0.0

    def test_shape_preserved(self, simple_counts, length_vectors):
        tpm = calculate_tpm(simple_counts, length_vectors)
        assert tpm.shape == simple_counts.shape
        assert list(tpm.columns) == list(simple_counts.columns)
        assert list(tpm.index) == list(simple_counts.index)

    def test_all_zero_sample(self, length_vectors):
        """A sample with all-zero counts should produce all-zero TPM."""
        counts = pd.DataFrame(
            {'sample_A': [0.0, 0.0, 0.0]},
            index=[0, 1, 2]
        )
        lvecs = {'sample_A': pd.Series([1000.0, 2000.0, 1500.0], index=[0, 1, 2])}
        tpm = calculate_tpm(counts, lvecs)
        assert (tpm['sample_A'] == 0).all()

    def test_longer_transcript_gets_lower_tpm(self):
        """Given equal counts, a longer transcript should get lower TPM."""
        counts = pd.DataFrame({'s': [100.0, 100.0]}, index=[0, 1])
        lvecs = {'s': pd.Series([1000.0, 5000.0], index=[0, 1])}
        tpm = calculate_tpm(counts, lvecs)
        assert tpm.loc[0, 's'] > tpm.loc[1, 's']


# =============================================================================
# _aggregate_by_map / _aggregate_genes_and_tes
# =============================================================================

class TestAggregation:
    def test_basic_aggregation(self, simple_counts, gene_map):
        result = _aggregate_by_map(simple_counts, gene_map)
        # GeneA = transcripts 0+1, GeneB = 2+3, GeneC = 4
        assert result.loc['GeneA', 'sample_A'] == pytest.approx(300.0)
        assert result.loc['GeneB', 'sample_A'] == pytest.approx(50.0)
        assert result.loc['GeneC', 'sample_A'] == pytest.approx(150.0)

    def test_no_matching_ids(self, simple_counts):
        """If no IDs overlap, should return empty DataFrame."""
        bad_map = pd.DataFrame({
            'LocusID': [99, 100],
            'AggregateID': ['Fake', 'Fake']
        })
        result = _aggregate_by_map(simple_counts, bad_map)
        assert result.empty

    def test_genes_and_tes_combined(self, simple_counts, gene_map, te_map):
        # te_map IDs (10, 11) don't exist in simple_counts, so only genes aggregate
        result = _aggregate_genes_and_tes(simple_counts, gene_map, te_map)
        assert 'GeneA' in result.index
        assert 'GeneB' in result.index
        assert 'GeneC' in result.index
        # L1_HS should not appear since IDs 10,11 aren't in the counts
        assert 'L1_HS' not in result.index

    def test_genes_and_tes_with_te_data(self, gene_map):
        """When TE IDs are present in counts, they should appear in result."""
        counts = pd.DataFrame(
            {'s': [10.0, 20.0, 30.0, 40.0, 50.0, 5.0, 15.0]},
            index=[0, 1, 2, 3, 4, 10, 11]
        )
        te_map = pd.DataFrame({
            'LocusID': [10, 11],
            'AggregateID': ['L1_HS', 'L1_HS']
        })
        result = _aggregate_genes_and_tes(counts, gene_map, te_map)
        assert 'L1_HS' in result.index
        assert result.loc['L1_HS', 's'] == pytest.approx(20.0)

    def test_te_map_none(self, simple_counts, gene_map):
        result = _aggregate_genes_and_tes(simple_counts, gene_map, None)
        assert 'GeneA' in result.index
        assert len(result) == 3


# =============================================================================
# _apply_bulk_updates
# =============================================================================

class TestApplyBulkUpdates:
    def test_basic_update(self):
        df = pd.DataFrame({'col_a': [1.0, 2.0, 3.0]}, index=[0, 1, 2])
        updates = {
            'new_col': {0: 'x', 2: 'z'},
            'score': {0: 0.5, 1: 0.9}
        }
        _apply_bulk_updates(df, updates)
        assert df.loc[0, 'new_col'] == 'x'
        assert df.loc[2, 'new_col'] == 'z'
        assert df.loc[0, 'score'] == 0.5
        assert df.loc[1, 'score'] == 0.9

    def test_empty_updates_no_op(self):
        df = pd.DataFrame({'a': [1.0]}, index=[0])
        original = df.copy()
        _apply_bulk_updates(df, {})
        pd.testing.assert_frame_equal(df, original)

    def test_all_empty_dicts(self):
        df = pd.DataFrame({'a': [1.0]}, index=[0])
        original = df.copy()
        _apply_bulk_updates(df, {'col1': {}, 'col2': {}})
        pd.testing.assert_frame_equal(df, original)


# =============================================================================
# _apply_junction_boosts
# =============================================================================

class TestApplyJunctionBoosts:
    def test_boosts_are_additive(self):
        counts = pd.Series([10.0, 20.0, 0.0], index=[0, 1, 2])
        tracking = pd.DataFrame(index=[0, 1])
        tracking['initial_count'] = counts
        evidence = {0: 5.0, 1: 3.0}
        args = Namespace(junction_weight=2.0)

        _apply_junction_boosts('test_sample', counts, tracking, evidence, args)

        # count[0] should be 10 + 5*2 = 20
        assert counts[0] == pytest.approx(20.0)
        # count[1] should be 20 + 3*2 = 26
        assert counts[1] == pytest.approx(26.0)
        # count[2] unchanged (no evidence, not in tracking)
        assert counts[2] == pytest.approx(0.0)

    def test_tracking_columns_set(self):
        counts = pd.Series([10.0, 20.0], index=[0, 1])
        tracking = pd.DataFrame(index=[0, 1])
        tracking['initial_count'] = counts.copy()
        evidence = {0: 4.0}
        args = Namespace(junction_weight=1.5)

        _apply_junction_boosts('test_sample', counts, tracking, evidence, args)

        assert tracking.loc[0, 'junction_boost'] == pytest.approx(6.0)
        assert tracking.loc[0, 'raw_junction_evidence'] == pytest.approx(4.0)
        assert tracking.loc[0, 'junction_weight'] == 1.5
        assert tracking.loc[0, 'post_junction_count'] == pytest.approx(16.0)

    def test_empty_evidence(self):
        counts = pd.Series([10.0], index=[0])
        tracking = pd.DataFrame(index=[0])
        tracking['initial_count'] = counts.copy()
        original = counts.copy()
        args = Namespace(junction_weight=2.0)

        _apply_junction_boosts('test_sample', counts, tracking, {}, args)

        assert counts[0] == original[0]


# =============================================================================
# _apply_tsl_penalties
# =============================================================================

class TestApplyTSLPenalties:
    def _make_context(self, tsl_map):
        """Build a minimal pipeline_context with transcript_info."""
        return {
            'transcript_info': {tid: {'tsl': tsl} for tid, tsl in tsl_map.items()}
        }

    def test_tsl1_no_penalty(self):
        counts = pd.Series([100.0], index=[0])
        tracking = pd.DataFrame(index=[0])
        tracking['initial_count'] = counts.copy()
        ctx = self._make_context({0: '1'})
        args = Namespace(use_tsl_penalty=True, tsl_penalty_values=None)

        _apply_tsl_penalties('test', counts, tracking, ctx, args)

        assert counts[0] == pytest.approx(100.0)
        assert tracking.loc[0, 'tsl_penalty'] == 1.0

    def test_tsl5_strong_penalty(self):
        counts = pd.Series([100.0], index=[0])
        tracking = pd.DataFrame(index=[0])
        tracking['initial_count'] = counts.copy()
        ctx = self._make_context({0: '5'})
        args = Namespace(use_tsl_penalty=True, tsl_penalty_values=None)

        _apply_tsl_penalties('test', counts, tracking, ctx, args)

        assert counts[0] == pytest.approx(30.0)  # 100 * 0.3
        assert tracking.loc[0, 'tsl_penalty'] == pytest.approx(0.3)

    def test_custom_tsl_values(self):
        counts = pd.Series([100.0], index=[0])
        tracking = pd.DataFrame(index=[0])
        tracking['initial_count'] = counts.copy()
        ctx = self._make_context({0: '2'})
        args = Namespace(use_tsl_penalty=True, tsl_penalty_values={'2': 0.5})

        _apply_tsl_penalties('test', counts, tracking, ctx, args)

        assert counts[0] == pytest.approx(50.0)  # 100 * 0.5 (custom override)

    def test_zero_count_not_penalized(self):
        counts = pd.Series([0.0], index=[0])
        tracking = pd.DataFrame(index=[0])
        tracking['initial_count'] = counts.copy()
        ctx = self._make_context({0: '5'})
        args = Namespace(use_tsl_penalty=True, tsl_penalty_values=None)

        _apply_tsl_penalties('test', counts, tracking, ctx, args)

        assert counts[0] == 0.0

    def test_missing_transcript_info(self):
        counts = pd.Series([100.0], index=[0])
        tracking = pd.DataFrame(index=[0])
        tracking['initial_count'] = counts.copy()
        ctx = {'transcript_info': {}}  # transcript 0 not in info
        args = Namespace(use_tsl_penalty=True, tsl_penalty_values=None)

        _apply_tsl_penalties('test', counts, tracking, ctx, args)

        # Should remain 'NA' with default penalty 0.8... but the code only
        # applies penalty if transcript_int is IN transcript_info, so no change
        assert counts[0] == pytest.approx(100.0)


# =============================================================================
# validate_args
# =============================================================================

class TestValidateArgs:
    def _base_args(self, tmp_path):
        """Create a minimal valid args namespace with a real BAM path."""
        bam = tmp_path / "test.bam"
        bam.touch()
        return Namespace(
            bams=[str(bam)],
            frag_stats_dir=None,
            mean_fragment_length=None,
            tsl_penalty_values=None,
            terminal_relax=False,
            library_type='dT',
            calculate_group_confidence=False,
            output_confidence=False,
            output_tpm=False,
        )

    def test_valid_args_pass(self, tmp_path):
        args = self._base_args(tmp_path)
        validate_args(args)  # should not raise or exit

    def test_missing_bam_exits(self, tmp_path):
        args = self._base_args(tmp_path)
        args.bams = ['/nonexistent/path/fake.bam']
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_mutual_exclusive_frag_args(self, tmp_path):
        args = self._base_args(tmp_path)
        args.frag_stats_dir = '/some/dir'
        args.mean_fragment_length = 200.0
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_terminal_relax_without_library_type(self, tmp_path):
        args = self._base_args(tmp_path)
        args.terminal_relax = True
        args.library_type = 'WARNING_UNSPECIFIED'
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_group_confidence_without_output_confidence(self, tmp_path):
        args = self._base_args(tmp_path)
        args.calculate_group_confidence = True
        args.output_confidence = False
        validate_args(args)
        assert args.calculate_group_confidence is False

    def test_invalid_tsl_json(self, tmp_path):
        args = self._base_args(tmp_path)
        args.tsl_penalty_values = 'not valid json'
        with pytest.raises(SystemExit):
            validate_args(args)


# =============================================================================
# build_argument_parser
# =============================================================================

class TestBuildArgumentParser:
    def test_parser_builds(self):
        parser = build_argument_parser()
        assert parser is not None

    def test_required_args(self):
        parser = build_argument_parser()
        # Should fail without required args
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self):
        parser = build_argument_parser()
        args = parser.parse_args(['--annotation', 'test.pkl.gz', '--bams', 'a.bam'])
        assert args.threads == 8
        assert args.em_iterations == 150
        assert args.junction_weight == 3.0
        assert args.prefix == 'MAJEC_output'
        assert args.strandedness == 0
        assert args.paired_end is False
