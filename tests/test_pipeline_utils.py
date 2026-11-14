"""Tests for pipeline utility helpers."""

import pytest
import numpy as np
import pandas as pd

from majec.utils.pipeline_utils import (
    calculate_effective_length_distributional,
    calculate_effective_length_empirical,
    fragment_hist_moments,
)


def _eff(lengths, frag_mean, frag_sd):
    """Convenience wrapper returning a Series indexed by transcript name."""
    s = pd.Series(lengths, index=[f"t{i}" for i in range(len(lengths))])
    return calculate_effective_length_distributional(s, frag_mean, frag_sd)


def _conditional_efflen(L, frag_mean, frag_sd):
    """Reference implementation of L - E[d | d <= L] + 1, computed directly.

    Deliberately independent of the code under test: it builds the truncated
    normal from scipy and does the conditioning by hand, so a shared bug in the
    binning or the renormalization cannot hide in both.
    """
    from scipy.stats import truncnorm

    min_len = int(max(1, frag_mean - 4 * frag_sd))
    max_len = int(frag_mean + 4 * frag_sd)
    d = np.arange(min_len, max_len + 1)
    p = truncnorm.pdf(
        d,
        (min_len - frag_mean) / frag_sd,
        (max_len - frag_mean) / frag_sd,
        loc=frag_mean,
        scale=frag_sd,
    )
    p /= p.sum()
    producible = d <= L
    if not producible.any():
        return 1.0
    return max(L - (d[producible] * p[producible]).sum() / p[producible].sum() + 1, 1.0)


# =============================================================================
# calculate_effective_length_distributional
# =============================================================================


class TestEffectiveLengthDistributional:
    """The result must be the CONDITIONAL effective length L - E[d | d <= L] + 1.

    The un-renormalized numerator sum_d P(d) * max(L - d + 1, 0) collapses
    towards zero below the fragment mean; these tests pin the renormalized
    values so that regression is caught.
    """

    def test_short_transcript_known_value(self):
        # At mean=205, sd=85 a 150 bp transcript has P(d <= 150) = 0.2547,
        # so the numerator (12.19) must be divided up to 47.86.
        result = _eff([150], 205, 85)
        assert result.iloc[0] == pytest.approx(47.857, rel=1e-3)
        # Guard against the numerator-only regression.
        assert result.iloc[0] > 40.0

    def test_known_values_across_lengths(self):
        result = _eff([50, 100, 150, 200], 205, 85)
        expected = [20.510, 34.570, 47.857, 63.937]
        for got, want in zip(result, expected):
            assert got == pytest.approx(want, rel=1e-3)

    def test_long_transcript_approaches_naive_length(self):
        """Well above the fragment mean, P(d <= L) -> 1 and eff -> L - E[d] + 1."""
        result = _eff([5000], 205, 85)
        assert result.iloc[0] == pytest.approx(5000 - 205 + 1, abs=10.0)

    @pytest.mark.parametrize("L", [150, 250, 400, 1000, 10000])
    def test_matches_conditional_reference(self, L):
        """Must equal the conditional length, not the bare numerator
        sum_d P(d) * max(L - d + 1, 0)."""
        result = _eff([float(L)], 205, 85)
        assert result.iloc[0] == pytest.approx(_conditional_efflen(L, 205, 85))

    def test_non_decreasing_in_length(self):
        lengths = list(range(1, 3000, 7))
        result = _eff(lengths, 205, 85)
        diffs = np.diff(result.values)
        assert (diffs >= -1e-9).all()

    @pytest.mark.parametrize("frag_sd", [85, 40, 20])
    def test_monotonic_and_bounded_across_sd(self, frag_sd):
        """The three properties must hold at every fragment SD, not just 85."""
        lengths = [1.0, 10, 45, 46, 100, 150, 250, 400, 1000, 10000]
        result = _eff(lengths, 205, frag_sd)
        assert (np.diff(result.values) >= -1e-9).all()
        assert (result >= 1.0).all()
        assert (result.values <= np.array(lengths) + 1e-9).all()

    def test_continuous_at_support_boundary(self):
        """Below the shortest modelled fragment the length is 1, which is the
        limit of the conditional formula, so there is no jump across it."""
        min_len = 205 - 4 * 40
        result = _eff([min_len - 1.0, float(min_len), min_len + 1.0], 205, 40)
        assert result.iloc[0] == pytest.approx(1.0)
        assert result.iloc[1] == pytest.approx(1.0)
        assert 1.0 < result.iloc[2] < 2.0

    def test_long_transcripts_unchanged_by_conditioning(self):
        """Well above the fragment distribution P(d <= L) = 1, so conditioning
        is a no-op and the result must still equal the non-conditional form."""
        from scipy.stats import truncnorm

        d = np.arange(1, int(205 + 4 * 85) + 1)
        p = truncnorm.pdf(d, (1 - 205) / 85, 4.0, loc=205, scale=85)
        p /= p.sum()
        lengths = np.array([2000.0, 5000.0])
        result = _eff(lengths, 205, 85)
        non_conditional = np.maximum(lengths[:, None] - d[None, :] + 1, 0) @ p
        assert result.values == pytest.approx(non_conditional, rel=1e-9)

    def test_never_below_one(self):
        result = _eff(list(range(0, 400)), 205, 85)
        assert (result >= 1.0).all()

    def test_very_short_transcript_floors_at_one(self):
        """Below the shortest modelled fragment there is nothing producible."""
        result = _eff([5], 500, 20)  # min modelled fragment is 420
        assert result.iloc[0] == 1.0

    def test_never_exceeds_transcript_length(self):
        lengths = [10, 50, 100, 200, 500, 1000, 5000]
        result = _eff(lengths, 205, 85)
        assert (result.values <= np.array(lengths, dtype=float)).all()

    def test_index_and_shape_preserved(self):
        s = pd.Series([150, 800, 2400], index=["ENST1", "ENST2", "ENST3"])
        result = calculate_effective_length_distributional(s, 205, 85)
        assert list(result.index) == ["ENST1", "ENST2", "ENST3"]
        assert len(result) == 3
        # The caller's Series must not be mutated in place.
        assert s.tolist() == [150, 800, 2400]

    def test_integer_input_lengths(self):
        """Integer-dtype lengths must not truncate the float division."""
        s = pd.Series(np.array([150, 300], dtype=np.int64), index=[0, 1])
        result = calculate_effective_length_distributional(s, 205, 85)
        assert result.iloc[0] == pytest.approx(47.857, rel=1e-3)
        assert result.dtype == np.float64

    def test_sequins_fragment_distribution(self):
        """Wide fragment SD (sequins run: mean 232.9, sd 159.5) stays sane."""
        result = _eff([150, 500, 2000], 232.863, 159.491)
        for got, want in zip(result, [64.390, 260.814, 1744.421]):
            assert got == pytest.approx(want, rel=1e-3)

    # --- point-mass branch (frag_sd missing) ---

    def test_no_sd_uses_point_mass(self):
        result = _eff([1000], 200, None)
        assert result.iloc[0] == pytest.approx(801.0)

    def test_no_sd_clips_at_one(self):
        result = _eff([150, 100], 200, None)
        assert (result >= 1.0).all()

    def test_zero_sd_uses_point_mass(self):
        result = _eff([1000], 200, 0)
        assert result.iloc[0] == pytest.approx(801.0)


# =============================================================================
# calculate_effective_length_empirical
# =============================================================================


def _eff_emp(lengths, hist):
    s = pd.Series(lengths, index=[f"t{i}" for i in range(len(lengths))])
    return calculate_effective_length_empirical(s, hist)


def _truncnorm_hist(frag_mean, frag_sd):
    """The exact pmf calculate_effective_length_distributional builds, as a
    histogram, so the two implementations can be compared on identical input."""
    from scipy.stats import truncnorm

    min_len = int(max(1, frag_mean - 4 * frag_sd))
    max_len = int(frag_mean + 4 * frag_sd)
    d = np.arange(min_len, max_len + 1)
    p = truncnorm.pdf(
        d,
        (min_len - frag_mean) / frag_sd,
        (max_len - frag_mean) / frag_sd,
        loc=frag_mean,
        scale=frag_sd,
    )
    return {int(k): float(v) for k, v in zip(d, p)}


def _conditional_efflen_hist(L, hist):
    """Reference implementation of L - E[d | d <= L] + 1 over an arbitrary pmf,
    written as an explicit mask-and-average rather than the cumulative-sum
    identity the code under test uses."""
    d = np.array(sorted(hist), dtype=float)
    w = np.array([hist[int(k)] for k in d], dtype=float)
    m = d <= L
    if not m.any():
        return 1.0
    return max(L - (d[m] * w[m]).sum() / w[m].sum() + 1, 1.0)


class TestEffectiveLengthEmpirical:
    """Same conditional quantity as the truncnorm path, but read off the
    measured fragment histogram instead of a fitted normal."""

    # A deliberately skewed histogram: this is the shape the parser actually
    # produces (mean well above the median), and the reason the function exists.
    SKEWED = {
        int(d): float(np.exp(-((np.log(d) - np.log(240)) ** 2) / (2 * 0.45**2)) / d)
        for d in range(40, 1400)
    }

    @pytest.mark.parametrize("L", [39, 40, 41, 150, 300, 1399, 1400, 5000])
    def test_matches_conditional_reference(self, L):
        got = _eff_emp([float(L)], self.SKEWED).iloc[0]
        assert got == pytest.approx(_conditional_efflen_hist(L, self.SKEWED))

    def test_matches_truncnorm_path_on_the_same_pmf(self):
        """Given the truncnorm path's own pmf the two must agree, which pins the
        difference between them to the distribution and not the arithmetic."""
        hist = _truncnorm_hist(205, 85)
        lengths = [150.0, 250.0, 400.0, 1000.0, 10000.0]
        emp = _eff_emp(lengths, hist)
        ref = _eff(lengths, 205, 85)
        assert emp.values == pytest.approx(ref.values, rel=1e-9)

    def test_skew_changes_the_answer_versus_a_normal_fit(self):
        """A truncnorm fitted to the same mean and sd is not the same model; if
        it were, there would be no reason to carry the histogram around."""
        n, mean, sd = fragment_hist_moments(self.SKEWED)
        emp = _eff_emp([300.0], self.SKEWED).iloc[0]
        fitted = _eff([300.0], mean, sd).iloc[0]
        assert abs(emp - fitted) > 1.0

    def test_point_mass_histogram(self):
        result = _eff_emp([1000.0, 250.0, 199.0], {200: 7.5})
        assert result.iloc[0] == pytest.approx(801.0)
        assert result.iloc[1] == pytest.approx(51.0)
        assert result.iloc[2] == pytest.approx(1.0)  # shorter than any fragment

    def test_weights_need_not_be_normalized(self):
        scaled = {k: v * 1234.5 for k, v in self.SKEWED.items()}
        a = _eff_emp([150.0, 900.0], self.SKEWED)
        b = _eff_emp([150.0, 900.0], scaled)
        assert a.values == pytest.approx(b.values, rel=1e-12)

    def test_fractional_weights_accepted(self):
        """A pair is spread 1/k over its k compatible transcripts, so the
        histogram carries fractional counts."""
        result = _eff_emp([1000.0], {150: 0.5, 250: 0.25, 350: 0.25})
        assert result.iloc[0] == pytest.approx(1001 - 225.0)

    def test_non_decreasing_in_length(self):
        result = _eff_emp([float(L) for L in range(1, 3000, 7)], self.SKEWED)
        assert (np.diff(result.values) >= -1e-9).all()

    def test_bounded_by_one_and_transcript_length(self):
        lengths = [1.0, 10, 39, 40, 41, 100, 500, 2000, 50000]
        result = _eff_emp(lengths, self.SKEWED)
        assert (result >= 1.0).all()
        assert (result.values <= np.array(lengths) + 1e-9).all()

    def test_continuous_at_support_boundary(self):
        """Below the shortest observed fragment the length is 1, matching the
        truncnorm path's behaviour below its own lower bound."""
        result = _eff_emp([39.0, 40.0, 41.0], self.SKEWED)
        assert result.iloc[0] == pytest.approx(1.0)
        assert result.iloc[1] == pytest.approx(1.0)
        assert 1.0 < result.iloc[2] < 2.0

    def test_far_tail_barely_moves_long_transcripts(self):
        """The measured histogram has a small tail past 1 kb that Salmon's own
        distribution cannot represent. Including it must not matter much, which
        is why the untruncated histogram is what gets passed through."""
        capped = {k: v for k, v in self.SKEWED.items() if k <= 1000}
        full = _eff_emp([20000.0], self.SKEWED).iloc[0]
        cut = _eff_emp([20000.0], capped).iloc[0]
        assert abs(full - cut) / cut < 0.01

    def test_empty_histogram_falls_back_to_annotated(self):
        s = pd.Series([150.0, 2000.0], index=["a", "b"])
        result = calculate_effective_length_empirical(s, {})
        assert result.tolist() == [150.0, 2000.0]

    def test_zero_weight_histogram_falls_back_to_annotated(self):
        s = pd.Series([150.0, 2000.0], index=["a", "b"])
        result = calculate_effective_length_empirical(s, {200: 0.0, 300: 0.0})
        assert result.tolist() == [150.0, 2000.0]

    def test_index_preserved_and_input_not_mutated(self):
        s = pd.Series([150, 800, 2400], index=["ENST1", "ENST2", "ENST3"])
        result = calculate_effective_length_empirical(s, self.SKEWED)
        assert list(result.index) == ["ENST1", "ENST2", "ENST3"]
        assert s.tolist() == [150, 800, 2400]

    def test_integer_input_lengths(self):
        s = pd.Series(np.array([150, 3000], dtype=np.int64), index=[0, 1])
        result = calculate_effective_length_empirical(s, self.SKEWED)
        assert result.dtype == np.float64
        assert result.iloc[0] == pytest.approx(
            _conditional_efflen_hist(150, self.SKEWED)
        )


class TestFragmentHistMoments:

    def test_matches_numpy_on_a_weighted_histogram(self):
        hist = {100: 1.0, 200: 3.0, 400: 6.0}
        n, mean, sd = fragment_hist_moments(hist)
        d = np.array([100, 200, 400], dtype=float)
        w = np.array([1.0, 3.0, 6.0])
        assert n == pytest.approx(10.0)
        assert mean == pytest.approx((d * w).sum() / 10.0)
        assert sd == pytest.approx(np.sqrt((w * (d - mean) ** 2).sum() / 10.0))

    def test_empty_histogram(self):
        assert fragment_hist_moments({}) == (0.0, 0.0, 0.0)

    def test_zero_weights(self):
        assert fragment_hist_moments({150: 0.0}) == (0.0, 0.0, 0.0)
