import numpy as np
from itertools import product


def all_genotypes(levels_per_pos):
    """Enumerate full combinatorial space."""
    grids = [range(L) for L in levels_per_pos]
    return np.array(list(product(*grids)), dtype=int)


def rank_corner_coords(X_all, orderings, levels_per_pos):
    """Compute (x,y) embedding for full combinatorial space."""
    X_all = np.asarray(X_all)
    d = X_all.shape[1]

    rank_lookup = []
    for pos in range(d):
        L = levels_per_pos[pos]
        ordering = orderings[pos]
        r = np.full(L, np.nan, float)
        for idx, state in enumerate(ordering):
            r[state] = 1.0 - idx / max(1, (L - 1))
        rank_lookup.append(r)

    R = np.zeros_like(X_all, dtype=float)
    for pos in range(d):
        R[:, pos] = rank_lookup[pos][X_all[:, pos]]

    x = np.mean(R, axis=1)
    y = 1.0 - np.var(R, axis=1)

    return x, y
