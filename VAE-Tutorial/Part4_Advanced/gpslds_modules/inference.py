"""
Inference Algorithms: Filtering and Smoothing.

E-step of EM algorithm: Given current parameters, infer latent states.

Two main algorithms:
1. Forward Pass (Kalman Filter): p(x_t | y_1:t)
2. Backward Pass (Rauch-Tung-Striebel Smoother): p(x_t | y_1:T)
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.linalg import solve, cho_solve, cholesky


@jit
def kalman_filter_step(mu_prev, S_prev, A, Q, C, R, y_t):
    """
    Single step of Kalman Filter (forward pass).

    E-step 1: Predict
        x_t | y_{1:t-1} ~ N(A @ mu_{t-1}, A @ S_{t-1} @ A^T + Q)

    E-step 2: Update
        x_t | y_{1:t} ~ N(mu_t, S_t)

    Args:
        mu_prev: [D] previous state mean
        S_prev: [D, D] previous state covariance
        A: [D, D] dynamics matrix
        Q: [D, D] process noise covariance
        C: [D_obs, D] observation matrix
        R: [D_obs, D_obs] observation noise covariance
        y_t: [D_obs] current observation

    Returns:
        mu_pred: [D] predicted mean before update
        S_pred: [D, D] predicted covariance before update
        mu_filt: [D] filtered mean after update
        S_filt: [D, D] filtered covariance after update
        lp: scalar log predictive likelihood
    """
    # Predict
    mu_pred = A @ mu_prev
    S_pred = A @ S_prev @ A.T + Q

    # Observation predict
    y_pred_mean = C @ mu_pred
    S_obs = C @ S_pred @ C.T + R

    # Kalman gain
    K = S_pred @ C.T @ jnp.linalg.inv(S_obs)

    # Update
    innovation = y_t - y_pred_mean
    mu_filt = mu_pred + K @ innovation
    S_filt = S_pred - K @ S_obs @ K.T

    # Log likelihood of observation
    lp = -0.5 * jnp.sum(innovation**2 @ jnp.linalg.inv(S_obs) @ innovation.T)
    lp -= 0.5 * jnp.linalg.slogdet(S_obs)[1]  # log det term

    return mu_pred, S_pred, mu_filt, S_filt, lp


def kalman_filter(A, Q, C, R, y, x0_mean, x0_cov):
    """
    Forward pass: Kalman filter over entire sequence.

    Args:
        A: [D, D] dynamics
        Q: [D, D] process noise
        C: [D_obs, D] observation matrix
        R: [D_obs, D_obs] observation noise
        y: [T, D_obs] observations
        x0_mean: [D] initial state mean
        x0_cov: [D, D] initial state covariance

    Returns:
        mu_filt: [T, D] filtered means
        S_filt: [T, D, D] filtered covariances
        ll: scalar log likelihood
    """
    T, D_obs = y.shape
    D = A.shape[0]

    # Initialize storage
    mu_filt = jnp.zeros((T, D))
    S_filt = jnp.zeros((T, D, D))

    # Initial step
    mu_filt = mu_filt.at[0].set(x0_mean)
    S_filt = S_filt.at[0].set(x0_cov)

    ll = 0.0
    mu_prev, S_prev = x0_mean, x0_cov

    # Forward pass
    for t in range(1, T):
        mu_pred, S_pred, mu_t, S_t, lp = kalman_filter_step(
            mu_prev, S_prev, A, Q, C, R, y[t]
        )
        mu_filt = mu_filt.at[t].set(mu_t)
        S_filt = S_filt.at[t].set(S_t)
        ll += lp
        mu_prev, S_prev = mu_t, S_t

    return mu_filt, S_filt, ll


@jit
def rts_smoother_step(mu_filt_t, S_filt_t, mu_filt_next, S_filt_next,
                      mu_pred_next, S_pred_next, A):
    """
    Backward step of Rauch-Tung-Striebel (RTS) Smoother.

    After forward pass, run backward to get smoothed estimates:
    x_t | y_{1:T} ~ N(mu_smooth_t, S_smooth_t)

    Args:
        mu_filt_t: [D] filtered mean at time t
        S_filt_t: [D, D] filtered covariance at time t
        mu_filt_next: [D] filtered mean at time t+1
        S_filt_next: [D, D] filtered covariance at time t+1
        mu_pred_next: [D] predicted mean at time t+1
        S_pred_next: [D, D] predicted covariance at time t+1
        A: [D, D] dynamics matrix

    Returns:
        mu_smooth_t: [D] smoothed mean
        S_smooth_t: [D, D] smoothed covariance
    """
    # Backward Kalman gain
    J_t = S_filt_t @ A.T @ jnp.linalg.inv(S_pred_next)

    # Smooth
    mu_smooth_t = mu_filt_t + J_t @ (mu_filt_next - mu_pred_next)
    S_smooth_t = S_filt_t + J_t @ (S_filt_next - S_pred_next) @ J_t.T

    return mu_smooth_t, S_smooth_t


def rts_smoother(mu_filt, S_filt, mu_pred, S_pred, A):
    """
    Backward pass: RTS Smoother.

    Args:
        mu_filt: [T, D] filtered means
        S_filt: [T, D, D] filtered covariances
        mu_pred: [T, D] predicted means
        S_pred: [T, D, D] predicted covariances
        A: [D, D] dynamics

    Returns:
        mu_smooth: [T, D] smoothed means
        S_smooth: [T, D, D] smoothed covariances
    """
    T, D = mu_filt.shape

    mu_smooth = jnp.zeros_like(mu_filt)
    S_smooth = jnp.zeros_like(S_filt)

    # Initialize with final filtered state
    mu_smooth = mu_smooth.at[-1].set(mu_filt[-1])
    S_smooth = S_smooth.at[-1].set(S_filt[-1])

    # Backward pass
    for t in range(T - 2, -1, -1):
        mu_smooth_t, S_smooth_t = rts_smoother_step(
            mu_filt[t], S_filt[t],
            mu_smooth[t + 1], S_smooth[t + 1],
            mu_pred[t + 1], S_pred[t + 1],
            A
        )
        mu_smooth = mu_smooth.at[t].set(mu_smooth_t)
        S_smooth = S_smooth.at[t].set(S_smooth_t)

    return mu_smooth, S_smooth
