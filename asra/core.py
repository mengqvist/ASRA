import numpy as np
from typing import Optional, Tuple, Sequence, List


def _build_point_weights(
    y: np.ndarray,
    sigma: Optional[np.ndarray],
    w: float,
    mode: str,
) -> Optional[np.ndarray]:
    """
    Construct per-point weights w_i = 1 / (1 + w * s_i),
    where s_i is a relative standard deviation.

    mode = "paper":  s_i = sigma_i / |y_i|, small absolute epsilon, no cap
    mode = "robust": s_i = sigma_i / max(|y_i|, eps_eff) with median-based eps and RSD cap
    """
    if sigma is None or w == 0.0:
        return None

    sigma = np.asarray(sigma, dtype=float)
    y = np.asarray(y, dtype=float)

    if mode == "paper":
        eps_abs = 1e-12
        rsd = sigma / np.maximum(np.abs(y), eps_abs)
    elif mode == "robust":
        med_abs_y = float(np.median(np.abs(y))) if y.size > 0 else 0.0
        eps_eff = max(1e-12, 0.05 * med_abs_y)
        rsd = sigma / np.maximum(np.abs(y), eps_eff)
        rsd = np.minimum(rsd, 2.0)  # cap RSD at 2.0
    else:
        raise ValueError("mode must be 'paper' or 'robust'.")

    return 1.0 / (1.0 + w * rsd)


def order_states_missing_first_worst_to_best(Q: np.ndarray, *, higher_is_better: bool = True) -> np.ndarray:
    """
    Return state indices ordered as:
      [missing (nan/inf), worst ... best]

    If higher_is_better=True: finite states are sorted ascending (low->high).
    If higher_is_better=False: finite states are sorted descending (high->low).
    """
    Q = np.asarray(Q, dtype=float)
    good_idx = np.where(np.isfinite(Q))[0]
    bad_idx = np.where(~np.isfinite(Q))[0]

    if good_idx.size == 0:
        return bad_idx.astype(int)

    if higher_is_better:
        finite_sorted = good_idx[np.argsort(Q[good_idx])]    # worst -> best
    else:
        finite_sorted = good_idx[np.argsort(-Q[good_idx])]   # worst -> best when lower is better

    return np.concatenate([bad_idx, finite_sorted]).astype(int)


def asra_orderings_multiblock(
    X: np.ndarray,
    y: np.ndarray,
    levels_per_pos: Sequence[int],
    sigma: Optional[np.ndarray] = None,
    w: float = 1.0,
    *,
    mode: str = "paper",   # "paper" or "robust"
    tau: float = 0.0,      # shrinkage (used mainly in robust mode)
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    General ASRA Step-2: matched-background pairwise differences with noise weighting,
    allowing each position to have a different number of states.

    Parameters
    ----------
    X : array, shape (n_samples, d)
        Encoded substituent indices at each position.
    y : array, shape (n_samples,)
        Measured property values.
    levels_per_pos : sequence of length d
        Number of possible states at each position (e.g. [3,4,3,4]).
    sigma : array, shape (n_samples,), optional
        Standard deviation / uncertainty for each y.
    w : float
        Weight factor in the paper. w=0 disables noise weighting.
    mode : {"paper", "robust"}
        - "paper": literal-ish implementation of the paper's weighting.
        - "robust": median-based epsilon, RSD cap, optional shrinkage.
    tau : float
        Shrinkage in the denominator:
        Q_m = sum(w_ij * Δ_ij) / (sum(w_ij) + tau)

    Returns
    -------
    orderings : list of length d
        orderings[pos] is an array of length levels_per_pos[pos],
        giving state indices sorted from smallest Q_m to largest.
    Qs : list of length d
        Qs[pos] is a float array of length levels_per_pos[pos],
        containing the Q_m scores for that position (nan if no valid pairs).
    """
    X = np.asarray(X, dtype=int)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1D and match number of rows in X.")

    d = X.shape[1]
    levels_per_pos = list(levels_per_pos)
    if len(levels_per_pos) != d:
        raise ValueError("levels_per_pos must have length equal to number of positions (X.shape[1]).")

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape != y.shape:
            raise ValueError("sigma must have same shape as y.")

    point_w = _build_point_weights(y, sigma, w, mode)

    orderings: List[np.ndarray] = []
    Qs: List[np.ndarray] = []

    for pos in range(d):
        L = int(levels_per_pos[pos])

        # Background key = all other positions
        bg = np.delete(X, pos, axis=1)
        if bg.shape[1] == 1:
            keys = bg[:, 0]
        else:
            keys = [tuple(r.tolist()) for r in bg]

        # For each state m at this position, map background -> sample index
        maps: List[dict] = []
        for m in range(L):
            dmap = {}
            rows = np.where(X[:, pos] == m)[0]
            for i in rows:
                if keys[i] not in dmap:
                    dmap[keys[i]] = i
            maps.append(dmap)

        Q = np.full(L, np.nan, dtype=float)

        # Compute Q_m for each state
        for m in range(L):
            dm = maps[m]
            if not dm:
                continue

            diffs: List[float] = []
            weights: List[float] = []

            for mp in range(L):
                if mp == m:
                    continue
                dmp = maps[mp]
                if not dmp:
                    continue

                common = set(dm.keys()) & set(dmp.keys())
                for k in common:
                    i = dm[k]
                    j = dmp[k]
                    diffs.append(y[i] - y[j])

                    if point_w is None:
                        weights.append(1.0)
                    else:
                        weights.append(float(point_w[i] * point_w[j]))

            if not diffs:
                continue

            diffs_arr = np.asarray(diffs, float)
            weights_arr = np.asarray(weights, float)

            num = float(np.sum(weights_arr * diffs_arr))
            den = float(np.sum(weights_arr) + (tau if mode == "robust" else 0.0))

            if den != 0.0:
                Q[m] = num / den
            else:
                Q[m] = np.nan

        # Sort states by Q (nan go to end)
        ordering = order_states_missing_first_worst_to_best(Q, higher_is_better=True)
        # ordering = invert_ordering_keep_missing_last(ordering, Q)
        orderings.append(ordering.astype(int))
        Qs.append(Q)

    return orderings, Qs


def asra_ordering(
    X: np.ndarray,
    y: np.ndarray,
    levels: int,
    sigma: Optional[np.ndarray] = None,
    w: float = 1.0,
    *,
    mode: str = "paper",
    tau: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backwards-compatible wrapper for uniform-depth positions.

    Parameters are the same as before, but internally this calls
    `asra_orderings_multiblock` with `levels_per_pos = [levels] * n_pos`.

    Returns
    -------
    ordering : array, shape (levels, n_pos)
        For each position, indices of states sorted from smallest Q_m to largest.
    Q : array, shape (levels, n_pos)
        Q_m scores for each state at each position.
    """
    X = np.asarray(X, dtype=int)
    n_samples, n_pos = X.shape
    levels_per_pos = [levels] * n_pos

    orderings, Qs = asra_orderings_multiblock(
        X=X,
        y=y,
        levels_per_pos=levels_per_pos,
        sigma=sigma,
        w=w,
        mode=mode,
        tau=tau,
    )

    ordering_arr = np.zeros((levels, n_pos), dtype=int)
    Q_arr = np.full((levels, n_pos), np.nan, dtype=float)

    for pos in range(n_pos):
        ordering_arr[:, pos] = orderings[pos]
        Q_arr[:, pos] = Qs[pos]

    return ordering_arr, Q_arr
