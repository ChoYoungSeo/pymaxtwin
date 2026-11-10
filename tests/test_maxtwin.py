"""Tests for pymaxtwin package."""

import numpy as np
import pytest
from pymaxtwin import maxtwin, initialmaxtwin


class TestInputValidation:
    def test_X_must_be_2d(self):
        with pytest.raises(ValueError, match="2D"):
            maxtwin(np.array([1, 2, 3]), np.array([1, 2, 3]), 0.5)

    def test_Y_length_must_match(self):
        X = np.random.randn(20, 2)
        with pytest.raises(ValueError, match="length"):
            maxtwin(X, np.array([1, 2]), 0.5)

    def test_splitratio_must_be_1_over_r(self):
        X = np.random.randn(20, 2)
        Y = np.random.randn(20)
        with pytest.raises(ValueError, match="reciprocal"):
            maxtwin(X, Y, 0.3)
        with pytest.raises(ValueError, match="1/splitratio = 2.5"):
            maxtwin(X, Y, 0.4)

    @pytest.mark.parametrize(
        "bad_splitratio",
        ["0.2", [0.2], None, np.nan, np.inf, 0, -0.2, True],
    )
    def test_splitratio_must_be_single_finite_positive_number(self, bad_splitratio):
        X = np.random.randn(20, 2)
        Y = np.random.randn(20)
        with pytest.raises(ValueError, match="single finite positive"):
            maxtwin(X, Y, bad_splitratio)

    def test_splitratio_r_must_be_at_least_2(self):
        X = np.random.randn(20, 2)
        Y = np.random.randn(20)
        with pytest.raises(ValueError, match="reciprocal"):
            maxtwin(X, Y, 1.0)

    def test_N_too_small(self):
        X = np.random.randn(3, 2)
        Y = np.random.randn(3)
        with pytest.raises(ValueError, match="at least"):
            maxtwin(X, Y, 0.5)

    def test_constant_column_rejected(self):
        X = np.column_stack([np.ones(20), np.random.randn(20)])
        Y = np.random.randn(20)
        with pytest.raises(ValueError, match="constant"):
            maxtwin(X, Y, 0.5)

    def test_constant_Y_rejected(self):
        X = np.random.randn(20, 2)
        Y = np.ones(20)
        with pytest.raises(ValueError, match="constant"):
            maxtwin(X, Y, 0.5)


class TestOutputCorrectness:
    def setup_method(self):
        np.random.seed(42)
        self.X = np.random.rand(100, 3)
        self.Y = np.sin(self.X[:, 0]) + np.cos(self.X[:, 1])

    def test_returns_sorted_indices(self):
        idx = maxtwin(self.X, self.Y, 0.2)
        assert np.all(idx[:-1] <= idx[1:])

    def test_correct_size(self):
        idx = maxtwin(self.X, self.Y, 0.2)
        assert len(idx) == 20  # 100 * 0.2

    def test_indices_in_bounds(self):
        idx = maxtwin(self.X, self.Y, 0.2)
        assert np.all(idx >= 0)
        assert np.all(idx < 100)

    def test_indices_unique(self):
        idx = maxtwin(self.X, self.Y, 0.2)
        assert len(np.unique(idx)) == len(idx)

    def test_splitratio_half(self):
        idx = maxtwin(self.X, self.Y, 0.5)
        assert len(idx) == 50

    def test_splitratio_quarter(self):
        idx = maxtwin(self.X, self.Y, 0.25)
        assert len(idx) == 25

    def test_splitratio_tenth(self):
        idx = maxtwin(self.X, self.Y, 0.1)
        assert len(idx) == 10


class TestInitialMaxtwin:
    def setup_method(self):
        np.random.seed(42)
        self.X = np.random.rand(100, 3)
        self.Y = np.sin(self.X[:, 0]) + np.cos(self.X[:, 1])

    def test_returns_sorted_indices(self):
        idx = initialmaxtwin(self.X, self.Y, 0.2)
        assert np.all(idx[:-1] <= idx[1:])

    def test_correct_size(self):
        idx = initialmaxtwin(self.X, self.Y, 0.2)
        assert len(idx) == 20

    def test_indices_unique(self):
        idx = initialmaxtwin(self.X, self.Y, 0.2)
        assert len(np.unique(idx)) == len(idx)


class TestDeterminism:
    def test_same_seed_same_result(self):
        np.random.seed(123)
        X = np.random.rand(100, 3)
        Y = np.random.randn(100)
        idx1 = maxtwin(X, Y, 0.2)
        idx2 = maxtwin(X, Y, 0.2)
        np.testing.assert_array_equal(idx1, idx2)


class TestSeparation:
    def test_separation_vs_random(self):
        """MaxTwin should have higher separation than random on paired data."""
        np.random.seed(42)
        N_half = 50
        p = 3
        X_base = np.random.rand(N_half, p)
        X_jitter = X_base + np.random.uniform(-0.005, 0.005, (N_half, p))
        X = np.vstack([X_base, X_jitter])
        Y = np.sin(X[:, 0]) + X[:, 1] ** 2

        N = len(X)
        idx_mt = maxtwin(X, Y, 0.2)

        # Compute separation for maxtwin
        train_mt = np.setdiff1d(np.arange(N), idx_mt)
        from scipy.spatial import KDTree
        tree = KDTree(X[train_mt])
        sep_mt = np.mean(tree.query(X[idx_mt])[0])

        # Compare with random (average over several seeds)
        seps_rand = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            idx_rand = np.sort(rng.choice(N, size=len(idx_mt), replace=False))
            train_rand = np.setdiff1d(np.arange(N), idx_rand)
            tree_r = KDTree(X[train_rand])
            seps_rand.append(np.mean(tree_r.query(X[idx_rand])[0]))

        assert sep_mt > np.median(seps_rand)


class TestEdgeCases:
    def test_univariate(self):
        np.random.seed(42)
        X = np.random.rand(50, 1)
        Y = np.sin(X[:, 0])
        idx = maxtwin(X, Y, 0.2)
        assert len(idx) == 10
        assert np.all(idx >= 0) and np.all(idx < 50)

    def test_high_dimensional(self):
        np.random.seed(42)
        X = np.random.rand(200, 20)
        Y = X[:, 0] + X[:, 1]
        idx = maxtwin(X, Y, 0.2)
        assert len(idx) == 40

    def test_paired_data(self):
        np.random.seed(42)
        X_base = np.random.rand(50, 2)
        X_jitter = X_base + np.random.uniform(-0.001, 0.001, (50, 2))
        X = np.vstack([X_base, X_jitter])
        Y = X[:, 0] ** 2 + X[:, 1]
        idx = maxtwin(X, Y, 0.2)
        assert len(idx) == 20
        assert len(np.unique(idx)) == 20
