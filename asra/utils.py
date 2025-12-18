import numpy as np
from itertools import product


def all_genotypes(levels_per_pos):
    """Enumerate full combinatorial space."""
    grids = [range(L) for L in levels_per_pos]
    return np.array(list(product(*grids)), dtype=int)



def rank_corner_coords(
    X_all,
    orderings,
    levels_per_pos,
    block_weights=None,
):
    """
    Compute (x, y) embedding for full combinatorial space.

    Parameters
    ----------
    X_all : array, shape (n_genotypes, d)
        Genotypes encoded as state indices per position.
    orderings : sequence of length d
        orderings[pos] is an array of state indices at that position,
        sorted from best to worst (ASRA order).
    levels_per_pos : sequence of length d
        Number of states at each position.
    block_weights : sequence of length d, optional
        Relative importance of each position. If None, all positions
        are weighted equally.

    Returns
    -------
    x : array, shape (n_genotypes,)
        Weighted mean rank per genotype (0 = worst, 1 = best).
    y : array, shape (n_genotypes,)
        1 - weighted variance of ranks; high when ranks are uniform.
    """
    X_all = np.asarray(X_all, dtype=int)
    d = X_all.shape[1]

    if len(orderings) != d:
        raise ValueError("len(orderings) must equal number of positions (X_all.shape[1]).")
    if len(levels_per_pos) != d:
        raise ValueError("len(levels_per_pos) must equal number of positions.")

    # Block weights
    if block_weights is None:
        w = np.ones(d, dtype=float)
    else:
        w = np.asarray(block_weights, dtype=float)
        if w.shape != (d,):
            raise ValueError("block_weights must have shape (d,), where d is number of positions.")
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError("Sum of block_weights must be positive.")
    w_norm = w / w_sum  # normalized weights

    # Rank lookup: rank[pos][state] in [0,1]
    rank_lookup = []
    for pos in range(d):
        L = int(levels_per_pos[pos])
        ordering = np.asarray(orderings[pos], dtype=int)
        if ordering.shape[0] != L:
            raise ValueError(f"orderings[{pos}] length must equal levels_per_pos[{pos}].")
        r = np.full(L, np.nan, float)
        # Best state gets rank 1.0, worst gets 0.0
        for idx, state in enumerate(ordering):
            r[state] = 1.0 - idx / max(1, (L - 1))
        rank_lookup.append(r)

    # R[g,pos] = normalized rank at each position
    R = np.zeros_like(X_all, dtype=float)
    for pos in range(d):
        R[:, pos] = rank_lookup[pos][X_all[:, pos]]

    # Weighted mean rank per genotype (distance to best corner)
    # x[g] = sum_j w_j * r_j(g)
    x = np.dot(R, w_norm)

    # Weighted variance of ranks per genotype
    # var[g] = sum_j w_j * (r_j(g) - x[g])^2
    diff = R - x[:, None]
    var = np.dot(diff**2, w_norm)

    # Consistency: high when ranks are similar across blocks
    y = 1.0 - var

    return x, y


def compute_block_weights_from_Q(Qs):
    """
    Compute automatic block weights from ASRA Q-values.

    For each position j, we compute a robust spread (MAD) of Qs[j],
    and use that as an effect-size proxy. Positions with larger spread
    get larger weight. Positions with almost no spread still get a small
    floor weight.

    Parameters
    ----------
    Qs : list of 1D arrays
        Qs[j] is an array of Q_m values for position j.

    Returns
    -------
    weights : 1D array of shape (d,)
        Non-negative weights that sum to 1.
    """
    spreads = []
    for Q in Qs:
        Q = np.asarray(Q, dtype=float)
        finite = Q[np.isfinite(Q)]
        if finite.size == 0:
            spreads.append(0.0)
            continue
        med = np.median(finite)
        mad = np.median(np.abs(finite - med))
        spreads.append(mad)

    spreads = np.asarray(spreads, dtype=float)

    # Small floor so that flat positions don't get exactly zero weight
    max_spread = float(np.max(spreads)) if spreads.size > 0 else 0.0
    floor = max(1e-12, 1e-3 * max_spread)  # 0.1% of max spread or tiny epsilon
    w_raw = spreads + floor

    # Normalize to sum to 1
    total = float(np.sum(w_raw))
    if total <= 0.0:
        # Fallback: all equal if everything is degenerate
        return np.ones_like(w_raw) / max(1, w_raw.size)

    return w_raw / total


def build_additive_model(Qs, normalize=True):
    """
    Returns a list of arrays s_j[m] = additive score for each block state.

    If normalize=True, each position is z-scored so that scales become comparable.
    """
    scores = []
    for Q in Qs:
        s = -np.asarray(Q, float)   # flip so higher = better
        if normalize:
            finite = s[np.isfinite(s)]
            if finite.size > 1:
                mean = np.mean(finite)
                std = np.std(finite)
                if std > 0:
                    s = (s - mean) / std
        scores.append(s)
    return scores


def additive_predict(scores, genotype):
    """
    genotype: list or 1D array of block-state indices
    scores: list of per-position score arrays
    """
    return sum(scores[pos][state] for pos, state in enumerate(genotype))
