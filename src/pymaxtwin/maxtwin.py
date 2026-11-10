"""MaxTwin algorithm for robust deterministic data splitting."""

from numbers import Real

import numpy as np
from scipy.spatial import KDTree
from scipy.special import beta as beta_fn
from twinning import twin


def _validate_inputs(X, Y, splitratio):
    """Validate inputs for maxtwin functions."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array.")
    N, p = X.shape
    if Y.shape != (N,):
        raise ValueError("Y must be a 1D array with length equal to nrow(X).")
    if not np.issubdtype(X.dtype, np.floating) and not np.issubdtype(X.dtype, np.integer):
        raise ValueError("X must be numeric.")
    if not np.issubdtype(Y.dtype, np.floating) and not np.issubdtype(Y.dtype, np.integer):
        raise ValueError("Y must be numeric.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or Inf.")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Y contains NaN or Inf.")

    if (
        isinstance(splitratio, bool)
        or not isinstance(splitratio, Real)
        or not np.isfinite(splitratio)
        or splitratio <= 0
    ):
        raise ValueError("splitratio must be a single finite positive number.")

    r = 1.0 / splitratio
    if abs(r - round(r)) > 1e-8 or round(r) < 2:
        raise ValueError(
            "splitratio must be the reciprocal of an integer r >= 2 because "
            "twinning.twin() expects integer r = 1/splitratio. "
            f"Received splitratio = {splitratio}, so 1/splitratio = {r}. "
            "Examples: 0.5, 1/3, 0.25, 0.2, 0.1."
        )
    r_int = int(round(r))
    if N < 2 * r_int:
        raise ValueError(f"N must be at least 2 * (1/splitratio) = {2 * r_int}.")

    col_vars = np.var(X, axis=0)
    if np.any(col_vars == 0):
        raise ValueError("X contains constant column(s). Remove or perturb them.")
    if np.var(Y) == 0:
        raise ValueError("Y is constant. Provide meaningful response values.")


def _run_twinning(Z, splitratio, initialization, standardization):
    """Run twinning algorithm with initialization options.

    Returns 0-indexed numpy array of twin indices.
    """
    r = int(round(1.0 / splitratio))

    if standardization:
        Z_in = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    else:
        Z_in = Z

    if initialization:
        zbar = Z.mean(axis=0)
        dists = np.sum((Z - zbar) ** 2, axis=1)
        u1 = int(np.argmin(dists))
        return twin(Z_in, r=r, u1=u1)
    else:
        return twin(Z_in, r=r)


def _knn_dist(X, k=1):
    """Compute k-th nearest neighbor distance for each point in X."""
    tree = KDTree(X)
    dists, _ = tree.query(X, k=k + 1)  # +1 because self is included
    return dists[:, k]  # k-th neighbor (0-indexed, so skip self at 0)


def _knnx_dist(ref, query, k=1):
    """Compute k-th nearest neighbor distance from query to ref."""
    tree = KDTree(ref)
    dists, _ = tree.query(query, k=k)
    if k == 1:
        return dists
    return dists[:, k - 1]


def _knnx_index(ref, query, k=1):
    """Find k nearest neighbor indices in ref for each point in query."""
    tree = KDTree(ref)
    _, indices = tree.query(query, k=k)
    if k == 1:
        return indices
    return indices[:, k - 1]


def initialmaxtwin(X, Y, splitratio, initialization=True, standardization=True):
    """Initial MaxTwin (Step 1 only).

    Selects testing points by maximizing nearest-neighbor distance within
    each Voronoi cell of the twinning set. This is the first step of
    Algorithm 1, without the iterative separation improvement.

    Parameters
    ----------
    X : ndarray, shape (N, p)
        Numeric matrix of inputs.
    Y : ndarray, shape (N,)
        Numeric vector of outputs.
    splitratio : float
        Fraction of data for the testing set. Must be 1/r where r is
        an integer >= 2 (e.g., 0.5, 0.25, 0.2, 0.1).
    initialization : bool, default True
        Start twinning from the point closest to the data centroid.
    standardization : bool, default True
        Standardize the augmented data (X, Y) before twinning.

    Returns
    -------
    ndarray
        Sorted 0-indexed array of testing point indices.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    _validate_inputs(X, Y, splitratio)

    N, p = X.shape

    Z = np.column_stack([X, Y])
    twin_ind = _run_twinning(Z, splitratio, initialization, standardization)
    Ntest = len(twin_ind)

    s = X[twin_ind]
    cluster = _knnx_index(s, X, k=1)
    nndis = _knn_dist(X, k=1)

    test_idx = np.empty(Ntest, dtype=int)
    for i in range(Ntest):
        cell = np.where(cluster == i)[0]
        test_idx[i] = cell[np.argmax(nndis[cell])]

    return np.sort(test_idx)


def maxtwin(X, Y, splitratio, initialization=True, standardization=True):
    """MaxTwin Algorithm for robust deterministic data splitting.

    Step 1 selects testing points using twinning + Voronoi + max NN distance.
    Step 2 iteratively improves separation when the threshold condition
    (Proposition 1) is violated, ensuring robustness to paired/clustered data.

    Parameters
    ----------
    X : ndarray, shape (N, p)
        Numeric matrix of inputs.
    Y : ndarray, shape (N,)
        Numeric vector of outputs.
    splitratio : float
        Fraction of data for the testing set. Must be 1/r where r is
        an integer >= 2 (e.g., 0.5, 0.25, 0.2, 0.1).
    initialization : bool, default True
        Start twinning from the point closest to the data centroid.
    standardization : bool, default True
        Standardize the augmented data (X, Y) before twinning.

    Returns
    -------
    ndarray
        Sorted 0-indexed array of testing point indices.

    References
    ----------
    Joseph, V.R. and Cho, Y. (2026). A Robust Deterministic Data Splitting
    Method for Model Validation. Journal of Computational and Graphical
    Statistics, 1-16. https://doi.org/10.1080/10618600.2026.2754089
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    _validate_inputs(X, Y, splitratio)

    N, p = X.shape
    gam = splitratio

    idx = np.arange(N)

    # ---- Step 1: Twinning ----
    Z = np.column_stack([X, Y])
    twin_ind = _run_twinning(Z, splitratio, initialization, standardization)
    Ntest = len(twin_ind)

    ind_sel = twin_ind.copy()
    s = X[ind_sel]

    # Rearrange: test candidates first, training after
    rest = np.setdiff1d(np.arange(N), ind_sel)
    X = np.vstack([s, X[rest]])
    idx = np.concatenate([idx[ind_sel], idx[rest]])
    ind_sel = np.arange(Ntest)

    # Voronoi assignment + select max-d in each cell
    cluster = _knnx_index(s, X, k=1)
    d = _knn_dist(X, k=1)

    for i in range(Ntest):
        ind = np.where(cluster == i)[0]
        if len(ind) > 1:
            ind_sel[i] = ind[np.argmax(d[ind])]

    # Compute separation distance
    train_mask = np.ones(len(X), dtype=bool)
    train_mask[ind_sel] = False
    dis = _knnx_dist(X[train_mask], X[ind_sel], k=1)
    dbar = np.mean(dis)

    # Reorder by separation (worst first)
    ord_ = np.argsort(dis)
    ind_sel = ind_sel[ord_]
    s = X[ind_sel]

    # Rearrange with updated selection
    rest = np.setdiff1d(np.arange(len(X)), ind_sel)
    X = np.vstack([s, X[rest]])
    idx = np.concatenate([idx[ind_sel], idx[rest]])

    # ---- Step 2: Ensuring Separation ----
    inds = np.arange(Ntest)
    cluster = _knnx_index(s, X, k=1)
    d = _knn_dist(X, k=1)

    K = max(min(int(np.ceil(2 * p)), Ntest // 2), 1)
    s_tree = KDTree(s)
    r_vals = s_tree.query(s, k=K + 1)[0][:, K]  # K-th neighbor (skip self)
    rbar = np.mean(r_vals)

    # Pre-compute constant part of S_min
    Smin_const = 0.5 * (1 - gam) / (gam + (1 - gam) * p) * beta_fn(1 / p, (1 - gam) / gam)

    if dbar < Smin_const * rbar:
        # Step 2 loop: ell = 2, ..., Ntest per Algorithm 1
        for i in range(1, Ntest):  # 0-indexed, so i=1 corresponds to ell=2
            # Voronoi for remaining active centers (i:Ntest),
            # excluding t_0..t_{i-1}
            active_s = s[i:]
            cluster = _knnx_index(active_s, X, k=1) + i
            d_full = np.zeros(len(X))
            d_full[i:] = _knn_dist(X[i:], k=1)

            cell_size = np.sum(cluster == i)

            if cell_size > 1:
                ind = np.where(cluster == i)[0]

                # Ball constraint: penalize points outside r_i radius
                rad = np.sqrt(np.sum((X[ind] - s[i]) ** 2, axis=1))
                d_full[ind[rad > r_vals[i]]] = -1

                inds[i] = ind[np.argmax(d_full[ind])]
            else:
                inds[i] = i

            # Rearrange with updated selection
            s = X[inds]
            rest = np.setdiff1d(np.arange(len(X)), inds)
            X = np.vstack([s, X[rest]])
            idx = np.concatenate([idx[inds], idx[rest]])
            inds = np.arange(Ntest)

            # Recheck separation threshold
            dbar = np.mean(_knnx_dist(X[Ntest:], s, k=1))
            r_vals = KDTree(s).query(s, k=K + 1)[0][:, K]
            rbar = np.mean(r_vals)

            if dbar > Smin_const * rbar:
                break

    return np.sort(idx[:Ntest])
