import numpy as np

from asra.utils import all_genotypes, rank_corner_coords, compute_block_weights_from_Q


def test_all_genotypes_basic():
    """
    all_genotypes should enumerate the full cartesian product of levels_per_pos
    in row-major order.
    """
    levels_per_pos = [2, 3]  # pos0: 0,1; pos1: 0,1,2
    X_all = all_genotypes(levels_per_pos)

    # 2 * 3 = 6 genotypes, 2 positions
    assert X_all.shape == (6, 2)

    expected = np.array([
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
    ], dtype=int)

    assert np.array_equal(X_all, expected)


def test_rank_corner_coords_shapes_and_ranges():
    """
    rank_corner_coords should return (x, y) of correct shape and values
    within [0, 1] for a simple multiblock example.
    """
    levels_per_pos = [3, 2]  # pos0: states 0,1,2; pos1: states 0,1
    X_all = all_genotypes(levels_per_pos)

    # Define orderings: pos0: 2 > 1 > 0, pos1: 1 > 0
    orderings = [
        np.array([2, 1, 0], dtype=int),
        np.array([1, 0], dtype=int),
    ]

    x, y = rank_corner_coords(X_all, orderings, levels_per_pos)

    # Shapes
    assert x.shape == (X_all.shape[0],)
    assert y.shape == (X_all.shape[0],)

    # Values should be in [0,1]
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    assert np.all(y >= 0.0) and np.all(y <= 1.0)


def test_rank_corner_coords_expected_values():
    """
    Check that rank_corner_coords gives intuitive coordinates:
    - best-best genotype is at (1, 1)
    - worst-worst genotype has mean rank 0, var 0 => y = 1
    - an intermediate genotype gets intermediate mean rank and <1 variance.
    """
    levels_per_pos = [3, 2]
    X_all = all_genotypes(levels_per_pos)

    # pos0 ordering: 2 (best), 1 (middle), 0 (worst)
    # pos1 ordering: 1 (best), 0 (worst)
    orderings = [
        np.array([2, 1, 0], dtype=int),
        np.array([1, 0], dtype=int),
    ]

    x, y = rank_corner_coords(X_all, orderings, levels_per_pos)

    # Helper to find index of a given genotype
    def idx_of(genotype):
        mask = np.all(X_all == np.asarray(genotype, dtype=int), axis=1)
        return int(np.where(mask)[0][0])

    idx_best_best = idx_of([2, 1])
    idx_worst_worst = idx_of([0, 0])
    idx_mixed = idx_of([2, 0])  # best at pos0, worst at pos1

    # Best-best: ranks = (1, 1) => mean=1, var=0 => x=1, y=1
    assert np.isclose(x[idx_best_best], 1.0)
    assert np.isclose(y[idx_best_best], 1.0)

    # Worst-worst: ranks = (0, 0) => mean=0, var=0 => x=0, y=1
    assert np.isclose(x[idx_worst_worst], 0.0)
    assert np.isclose(y[idx_worst_worst], 1.0)

    # Mixed: pos0 rank=1, pos1 rank=0 => mean=0.5
    # var = ((1-0.5)^2 + (0-0.5)^2)/2 = 0.25/2 = 0.125 => y=1-0.125=0.875
    assert np.isclose(x[idx_mixed], 0.5)
    assert np.isclose(y[idx_mixed], 0.875)

    # Best-best should have x >= any other genotype
    assert np.all(x[idx_best_best] >= x - 1e-12)



def test_rank_corner_coords_block_weights():
    levels_per_pos = [3, 2]
    X_all = all_genotypes(levels_per_pos)

    # pos0: 2 > 1 > 0, pos1: 1 > 0
    orderings = [
        np.array([2, 1, 0], dtype=int),
        np.array([1, 0], dtype=int),
    ]

    # Equal weights baseline
    x_eq, y_eq = rank_corner_coords(X_all, orderings, levels_per_pos)

    # Heavily weight pos0
    x_w, y_w = rank_corner_coords(X_all, orderings, levels_per_pos, block_weights=[10.0, 1.0])

    # Genotype that differs only at pos0 should move more under weighting
    # e.g. (2,0) vs (0,0)
    def idx_of(g):
        mask = np.all(X_all == np.array(g, int), axis=1)
        return int(np.where(mask)[0][0])

    i_best0 = idx_of([2, 0])
    i_worst0 = idx_of([0, 0])

    # Under heavier pos0 weighting, the x-distance between best0 and worst0 should increase
    dist_eq = x_eq[i_best0] - x_eq[i_worst0]
    dist_w = x_w[i_best0] - x_w[i_worst0]
    assert dist_w > dist_eq


def test_compute_block_weights_from_Q_spread_drives_weight():
    """
    Positions with larger Q spread (MAD) should get larger weights.
    """
    # Three positions:
    # - pos0: flat Qs (no effect)
    # - pos1: moderate spread
    # - pos2: large spread
    Qs = [
        np.array([1.0, 1.0, 1.0]),        # pos0, MAD = 0
        np.array([0.0, 1.0, 2.0]),        # pos1, MAD > 0
        np.array([-5.0, 0.0, 5.0]),       # pos2, MAD >> pos1
    ]

    weights = compute_block_weights_from_Q(Qs)

    # Shape and normalization
    assert weights.shape == (3,)
    assert np.all(weights > 0.0)
    assert np.isclose(np.sum(weights), 1.0)

    # pos2 should get strictly larger weight than pos1, which should be >= pos0
    w0, w1, w2 = weights
    assert w2 > w1
    assert w1 >= w0


def test_compute_block_weights_from_Q_all_flat_positions():
    """
    When all positions are essentially flat (no spread),
    weights should fall back to being ~equal.
    """
    Qs = [
        np.array([1.0, 1.0, 1.0]),
        np.array([2.0, 2.0, 2.0]),
        np.array([-3.0, -3.0, -3.0]),
    ]

    weights = compute_block_weights_from_Q(Qs)

    assert weights.shape == (3,)
    assert np.isclose(np.sum(weights), 1.0)

    # All positions are symmetric; weights should be approximately equal.
    # Allow small numerical differences.
    assert np.allclose(weights, np.ones(3) / 3.0, rtol=1e-6, atol=1e-6)


def test_compute_block_weights_from_Q_ignores_nans():
    """
    NaN entries in Qs should be ignored when computing spreads.
    """
    Qs = [
        np.array([np.nan, np.nan, np.nan]),     # entirely missing
        np.array([0.0, np.nan, 2.0]),           # some missing
        np.array([-3.0, 0.0, 3.0]),            # good spread
    ]

    weights = compute_block_weights_from_Q(Qs)

    assert weights.shape == (3,)
    assert np.isclose(np.sum(weights), 1.0)

    # Third position has biggest spread, so highest weight
    w0, w1, w2 = weights
    assert w2 > w1
    # pos0 only gets floor, so it should be smallest
    assert w0 <= w1
    assert w0 <= w2
