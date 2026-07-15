"""
Expectation-Maximization (EM) Algorithm for gpSLDS.

This is the core learning algorithm:

E-step: Infer p(x_t | y_1:T) using filtering/smoothing
M-step: Update model parameters to maximize expected log-likelihood

The innovation over standard SLDS EM is in the M-step:
- Standard SLDS: Learn hard discrete switches
- gpSLDS: Learn continuous GP-based smooth switches
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import jax.scipy as jsp

from .inference import kalman_filter, rts_smoother
from .kernels import rbf_kernel, add_jitter
from .likelihoods import gaussian_loglik, poisson_loglik


def fit_gpslds(y, D_latent, num_modes=2, max_iter=100,
               likelihood="gaussian", verbose=True):
    """
    Main function: Fit gpSLDS to data.

    This is what you call from the notebook. It runs the full EM loop.

    Args:
        y: [T, D_obs] observations
        D_latent: dimensionality of latent state
        num_modes: number of local linear modes
        max_iter: maximum EM iterations
        likelihood: "gaussian" or "poisson"
        verbose: print progress

    Returns:
        params: dict of learned parameters
        x_smooth: [T, D_latent] smoothed latent states
        ll_history: [max_iter] log-likelihood per iteration
    """
    T, D_obs = y.shape
    key = jax.random.PRNGKey(0)

    # Initialize parameters randomly
    params = {
        'A': jax.random.normal(key, (num_modes, D_latent, D_latent)) * 0.5,
        'Q': jnp.array([0.1 * jnp.eye(D_latent) for _ in range(num_modes)]),
        'C': jax.random.normal(key, (D_obs, D_latent)) * 0.5,
        'R': 0.1 * jnp.eye(D_obs),
        'pi_0': jnp.ones(num_modes) / num_modes,
        'kernel_scale': 1.0,  # GP kernel length scale
    }

    ll_history = []

    for iteration in range(max_iter):
        # E-step: Infer latent states
        # (Simplified: assume mode sequence is known or use mode-averaged A)
        A_avg = jnp.mean(params['A'], axis=0)
        Q_avg = jnp.mean(params['Q'], axis=0)

        mu_filt, S_filt, ll = kalman_filter(
            A_avg, Q_avg, params['C'], params['R'], y,
            jnp.zeros(D_latent), jnp.eye(D_latent)
        )

        mu_smooth, S_smooth = rts_smoother(
            mu_filt, S_filt,
            jnp.array([A_avg @ mu_filt[t] for t in range(T)]),
            jnp.array([A_avg @ S_filt[t] @ A_avg.T + Q_avg for t in range(T)]),
            A_avg
        )

        ll_history.append(float(ll))

        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: LL = {ll:.4f}")

        # M-step: Update parameters
        # (Simplified M-step: update C and R)
        # Full version would update A, Q, pi, kernel_scale

        # Update observation matrix C
        X_smooth = mu_smooth.T  # [D_latent, T]
        Y = y.T  # [D_obs, T]

        params['C'] = Y @ X_smooth.T @ jnp.linalg.pinv(X_smooth @ X_smooth.T)

        # Update observation noise
        residual = Y - params['C'] @ X_smooth
        params['R'] = (residual @ residual.T) / T
        params['R'] = jnp.clip(params['R'], 1e-4, None)

    return params, mu_smooth, jnp.array(ll_history)


def e_step_simplified(y, A, Q, C, R, x0_mean, x0_cov):
    """
    E-step: Forward-backward inference.

    Given parameters, infer latent states using Kalman filter + RTS smoother.

    Args:
        y: [T, D_obs] observations
        A: [D, D] dynamics (mode-averaged or single)
        Q: [D, D] process noise
        C: [D_obs, D] observation matrix
        R: [D_obs, D_obs] observation noise
        x0_mean: [D] initial state mean
        x0_cov: [D, D] initial state covariance

    Returns:
        x_smooth: [T, D] smoothed latent states
        S_smooth: [T, D, D] smoothed state covariances
        ll: scalar log-likelihood
    """
    mu_filt, S_filt, ll = kalman_filter(A, Q, C, R, y, x0_mean, x0_cov)

    # Compute predicted states for smoother
    T, D = mu_filt.shape
    mu_pred = jnp.array([A @ mu_filt[t-1] if t > 0
                         else x0_mean for t in range(T)])
    S_pred = jnp.array([A @ S_filt[t-1] @ A.T + Q if t > 0
                        else x0_cov for t in range(T)])

    x_smooth, S_smooth = rts_smoother(mu_filt, S_filt, mu_pred, S_pred, A)

    return x_smooth, S_smooth, ll


def m_step_simplified(y, x_smooth, x_cov, likelihood="gaussian"):
    """
    M-step: Update observation model parameters.

    Simplified version: updates C and R only.
    Full version would also update A, Q (dynamics) and switching probabilities.

    Args:
        y: [T, D_obs] observations
        x_smooth: [T, D_latent] smoothed states
        x_cov: [T, D_latent, D_latent] state covariances
        likelihood: "gaussian" or "poisson"

    Returns:
        C: [D_obs, D_latent] updated observation matrix
        R: [D_obs, D_obs] updated observation noise (if Gaussian)
    """
    T, D_obs = y.shape
    D_latent = x_smooth.shape[1]

    # Maximum likelihood estimate of C
    # C = Y @ X^T @ (X @ X^T)^{-1}
    X_smooth = x_smooth.T  # [D_latent, T]
    Y = y.T  # [D_obs, T]

    # Account for state uncertainty in posterior covariance
    ExxT = X_smooth @ X_smooth.T / T  # E[x_t x_t^T]
    for t in range(T):
        ExxT = ExxT + x_cov[t] / T

    C = Y @ X_smooth.T @ jnp.linalg.pinv(ExxT)

    if likelihood == "gaussian":
        # Update observation noise
        residual = Y - C @ X_smooth
        R = (residual @ residual.T) / T
        R = jnp.clip(R, 1e-4, None)
        return C, R
    else:  # poisson
        return C, None
