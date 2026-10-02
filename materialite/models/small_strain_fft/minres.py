import logging

import numpy as np
from scipy.sparse.linalg._isolve.utils import make_system


def minres(A, b, x0=None, rtol=1e-5, atol=None, maxiter=None, callback=None):
    """
    Minimal Residual method for solving Ax = b where A is a linear operator.
    """
    logger = logging.getLogger("minres")
    A, _, x, b = make_system(A, None, x0, b)
    matvec = A.matvec
    xtype = x.dtype
    n = len(b)
    b_norm = np.linalg.norm(b)
    btol = rtol * b_norm
    epsilon = np.finfo(xtype).eps

    if x0 is None:
        residual = b
        beta_curr = b_norm
    else:
        residual = b - A(x)
        beta_curr = np.linalg.norm(residual)

    if atol is None:
        atol = epsilon

    if maxiter is None:
        maxiter = 5 * n

    if beta_curr < atol:
        return x, 0

    v_curr = residual / beta_curr
    v_old = np.zeros_like(b)
    search_dir_old = np.zeros_like(b)
    search_dir_old_old = np.zeros_like(b)
    c = -1.0
    s = 0.0
    old_residual_norm = beta_curr
    step_size = 0.0
    precomputed_diag_term = 0.0
    off_off_diag = 0.0
    max_diag = 0.0
    min_diag = np.inf
    logger.debug("Iteration  Residual norm")

    for k in range(maxiter):
        # Lanczos step
        w = matvec(v_curr)
        w = w - beta_curr * v_old
        alpha = np.dot(w, v_curr)
        w = w - alpha * v_curr
        beta_new = np.linalg.norm(w)

        # Apply previous two Givens rotations
        diag_initial = s * precomputed_diag_term - c * alpha
        off_diag = c * precomputed_diag_term + s * alpha
        off_off_diag_new = s * beta_new
        precomputed_diag_term = -c * beta_new

        # New Givens rotation to zero out beta_new
        rotation_norm = np.linalg.norm([diag_initial, beta_new])
        diag = np.max([rotation_norm, epsilon])
        c = diag_initial / diag
        s = beta_new / diag

        # Solution update
        step_size = c * old_residual_norm
        residual_norm = s * old_residual_norm
        search_dir_curr = (
            v_curr - off_diag * search_dir_old - off_off_diag * search_dir_old_old
        ) / diag
        x += step_size * search_dir_curr

        # Check convergence
        max_diag = np.max([max_diag, np.abs(diag)])
        min_diag = np.min([min_diag, np.abs(diag)])
        condition_number = max_diag / min_diag
        residual_norm_difference = old_residual_norm - residual_norm
        if k < 5 or (k + 1) % 10 == 0:
            logger.debug(f"{(k + 1):8d} {residual_norm / b_norm:16.8e}")

        if callback is not None:
            callback(x)

        if np.abs(residual_norm) < btol + atol:
            return_code = "Residual norm converged"
            break
        if condition_number > 0.1 / epsilon:
            return_code = "Condition number too large"
            break
        if residual_norm_difference < old_residual_norm * epsilon:
            return_code = "Residual norm stagnation"
            break
        if k == maxiter - 1:
            return_code = "Maximum iterations reached without convergence"

        # Update variables for next iteration
        v_old = v_curr
        v_curr = w / beta_new
        search_dir_old_old = search_dir_old
        search_dir_old = search_dir_curr
        beta_curr = beta_new
        off_off_diag = off_off_diag_new
        old_residual_norm = residual_norm

    return (x, return_code)
