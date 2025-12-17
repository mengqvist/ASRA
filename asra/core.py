import numpy as np
from typing import Optional, Tuple



def asra_ordering(
    X: np.ndarray,
    y: np.ndarray,
    levels: int,
    sigma: Optional[np.ndarray] = None,
    w: float = 100.0,
    *,
    mode: str = "paper",   # "paper" or "robust"
    tau: float = 0.0,       # shrinkage (used only in robust mode)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ASRA Step-2: matched-background pairwise differences with noise weighting.

    Parameters
    ----------
    X : array, shape (n_samples, n_pos)
        Encoded substituent indices at each position.
    y : array, shape (n_samples,)
        Measured property values.
    levels : int
        Number of possible states (e.g. 20 amino acids) at each position.
    sigma : array, shape (n_samples,), optional
        Standard deviation / uncertainty for each y.
    w : float
        Weight factor in the paper. w=0 disables noise weighting.
    mode : {"paper", "robust"}
        - "paper": literal-ish implementation of the paper's weighting.
        - "robust": more numerically stable variant (median-based epsilon, RSD cap, shrinkage).
    tau : float
        Shrinkage applied only in "robust" mode:
        Q_m = sum(w_ij * Δ_ij) / (sum(w_ij) + tau)

    Returns
    -------
    ordering : array, shape (levels, n_pos)
        For each position, indices of states sorted from smallest Q_m to largest.
    Q : array, shape (levels, n_pos)
        Q_m scores for each state at each position (nan if no valid pairs).
    """
    X = np.asarray(X, dtype=int)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1D and match number of rows in X.")
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape != y.shape:
            raise ValueError("sigma must have same shape as y.")

    if mode not in ("paper", "robust"):
        raise ValueError("mode must be 'paper' or 'robust'.")

    # Build per-point weights if sigma and w are provided
    point_w = None
    if sigma is not None and w != 0.0:
        if mode == "paper":
            # Literal: s_i = sigma_i / |y_i|, small absolute epsilon
            eps_abs = 1e-12
            rsd = sigma / np.maximum(np.abs(y), eps_abs)
            # No RSD cap in paper mode
        else:  # mode == "robust"
            # Robust: eps based on median(|y|), RSD capped
            med_abs_y = float(np.median(np.abs(y))) if y.size > 0 else 0.0
            eps_eff = max(1e-12, 0.05 * med_abs_y)
            rsd = sigma / np.maximum(np.abs(y), eps_eff)
            rsd = np.minimum(rsd, 2.0)  # cap RSD at 2.0

        point_w = 1.0 / (1.0 + w * rsd)

    n_samples, n_pos = X.shape
    Q = np.full((levels, n_pos), np.nan, dtype=float)

    for pos in range(n_pos):
        # Background key = all other positions
        bg = np.delete(X, pos, axis=1)
        if bg.shape[1] == 1:
            keys = bg[:, 0]
        else:
            keys = [tuple(r.tolist()) for r in bg]

        # For each state m at this position, map backgrounds -> sample index
        maps = []
        for m in range(levels):
            d = {}
            rows = np.where(X[:, pos] == m)[0]
            for i in rows:
                if keys[i] not in d:
                    d[keys[i]] = i
            maps.append(d)

        # Compute Q_m for each state
        for m in range(levels):
            dm = maps[m]
            if not dm:
                continue

            diffs = []
            weights = []

            for mp in range(levels):
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

                    if sigma is None or w == 0.0 or point_w is None:
                        weights.append(1.0)
                    else:
                        weights.append(float(point_w[i] * point_w[j]))

            if not diffs:
                continue

            diffs = np.asarray(diffs, float)
            weights = np.asarray(weights, float)

            num = float(np.sum(weights * diffs))
            if mode == "paper":
                den = float(np.sum(weights))
            else:  # robust
                den = float(np.sum(weights) + tau)

            if den != 0.0:
                Q[m, pos] = num / den
            else:
                Q[m, pos] = np.nan

    # Sort amino acids / states by Q (nan go to end)
    ordering = np.zeros((levels, n_pos), dtype=int)
    for pos in range(n_pos):
        qcol = Q[:, pos]
        good = np.where(~np.isnan(qcol))[0]
        bad = np.where(np.isnan(qcol))[0]
        ordering[:, pos] = np.concatenate([good[np.argsort(qcol[good])], bad])

    return ordering, Q
