import numpy as np
from typing import Optional, Tuple



# def asra_ordering(
#     X: np.ndarray,
#     y: np.ndarray,
#     levels: int,
#     sigma: Optional[np.ndarray] = None,
#     w: float = 100.0,
#     *,
#     mode: str = "paper",   # "paper" or "robust"
#     tau: float = 0.0,       # shrinkage (used only in robust mode)
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     ASRA Step-2: matched-background pairwise differences with noise weighting.

#     Parameters
#     ----------
#     X : array, shape (n_samples, n_pos)
#         Encoded substituent indices at each position.
#     y : array, shape (n_samples,)
#         Measured property values.
#     levels : int
#         Number of possible states (e.g. 20 amino acids) at each position.
#     sigma : array, shape (n_samples,), optional
#         Standard deviation / uncertainty for each y.
#     w : float
#         Weight factor in the paper. w=0 disables noise weighting.
#     mode : {"paper", "robust"}
#         - "paper": literal-ish implementation of the paper's weighting.
#         - "robust": more numerically stable variant (median-based epsilon, RSD cap, shrinkage).
#     tau : float
#         Shrinkage applied only in "robust" mode:
#         Q_m = sum(w_ij * Δ_ij) / (sum(w_ij) + tau)

#     Returns
#     -------
#     ordering : array, shape (levels, n_pos)
#         For each position, indices of states sorted from smallest Q_m to largest.
#     Q : array, shape (levels, n_pos)
#         Q_m scores for each state at each position (nan if no valid pairs).
#     """
#     X = np.asarray(X, dtype=int)
#     y = np.asarray(y, dtype=float)

#     if X.ndim != 2:
#         raise ValueError("X must be 2D.")
#     if y.ndim != 1 or y.shape[0] != X.shape[0]:
#         raise ValueError("y must be 1D and match number of rows in X.")
#     if sigma is not None:
#         sigma = np.asarray(sigma, dtype=float)
#         if sigma.shape != y.shape:
#             raise ValueError("sigma must have same shape as y.")

#     if mode not in ("paper", "robust"):
#         raise ValueError("mode must be 'paper' or 'robust'.")

#     # Build per-point weights if sigma and w are provided
#     point_w = None
#     if sigma is not None and w != 0.0:
#         if mode == "paper":
#             # Literal: s_i = sigma_i / |y_i|, small absolute epsilon
#             eps_abs = 1e-12
#             rsd = sigma / np.maximum(np.abs(y), eps_abs)
#             # No RSD cap in paper mode
#         else:  # mode == "robust"
#             # Robust: eps based on median(|y|), RSD capped
#             med_abs_y = float(np.median(np.abs(y))) if y.size > 0 else 0.0
#             eps_eff = max(1e-12, 0.05 * med_abs_y)
#             rsd = sigma / np.maximum(np.abs(y), eps_eff)
#             rsd = np.minimum(rsd, 2.0)  # cap RSD at 2.0

#         point_w = 1.0 / (1.0 + w * rsd)

#     n_samples, n_pos = X.shape
#     Q = np.full((levels, n_pos), np.nan, dtype=float)

#     for pos in range(n_pos):
#         # Background key = all other positions
#         bg = np.delete(X, pos, axis=1)
#         if bg.shape[1] == 1:
#             keys = bg[:, 0]
#         else:
#             keys = [tuple(r.tolist()) for r in bg]

#         # For each state m at this position, map backgrounds -> sample index
#         maps = []
#         for m in range(levels):
#             d = {}
#             rows = np.where(X[:, pos] == m)[0]
#             for i in rows:
#                 if keys[i] not in d:
#                     d[keys[i]] = i
#             maps.append(d)

#         # Compute Q_m for each state
#         for m in range(levels):
#             dm = maps[m]
#             if not dm:
#                 continue

#             diffs = []
#             weights = []

#             for mp in range(levels):
#                 if mp == m:
#                     continue
#                 dmp = maps[mp]
#                 if not dmp:
#                     continue

#                 common = set(dm.keys()) & set(dmp.keys())
#                 for k in common:
#                     i = dm[k]
#                     j = dmp[k]
#                     diffs.append(y[i] - y[j])

#                     if sigma is None or w == 0.0 or point_w is None:
#                         weights.append(1.0)
#                     else:
#                         weights.append(float(point_w[i] * point_w[j]))

#             if not diffs:
#                 continue

#             diffs = np.asarray(diffs, float)
#             weights = np.asarray(weights, float)

#             num = float(np.sum(weights * diffs))
#             if mode == "paper":
#                 den = float(np.sum(weights))
#             else:  # robust
#                 den = float(np.sum(weights) + tau)

#             if den != 0.0:
#                 Q[m, pos] = num / den
#             else:
#                 Q[m, pos] = np.nan

#     # Sort amino acids / states by Q (nan go to end)
#     ordering = np.zeros((levels, n_pos), dtype=int)
#     for pos in range(n_pos):
#         qcol = Q[:, pos]
#         good = np.where(~np.isnan(qcol))[0]
#         bad = np.where(np.isnan(qcol))[0]
#         ordering[:, pos] = np.concatenate([good[np.argsort(qcol[good])], bad])

#     return ordering, Q




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
        good = np.where(~np.isnan(Q))[0]
        bad = np.where(np.isnan(Q))[0]
        ordering = np.concatenate([good[np.argsort(Q[good])], bad])
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
