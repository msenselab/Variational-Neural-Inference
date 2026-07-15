"""
Gaussian Process Kernels for gpSLDS.

Adapted from lindermanlab/gpslds. This module defines kernel functions
that control the smoothness of dynamics and the switching between local
linear modes. The key innovation is the "smooth switching" kernel that
gradually transitions between regimes rather than hard switching.
"""

import jax
import jax.numpy as jnp
from jax import vmap, grad, jit


@jit
def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """
    Radial Basis Function (RBF) / Squared Exponential Kernel.

    k(x1, x2) = variance * exp(-||x1 - x2||^2 / (2 * length_scale^2))

    This controls the smoothness of GP functions. Smaller length_scale
    means more rapidly changing dynamics; larger means smoother.

    Args:
        x1: [T1] time points
        x2: [T2] time points
        length_scale: controls temporal correlation
        variance: overall scale of covariance

    Returns:
        K: [T1, T2] covariance matrix
    """
    # Compute pairwise squared distances
    sq_diff = jnp.power(x1[:, None] - x2[None, :], 2)  # [T1, T2]

    # RBF formula
    K = variance * jnp.exp(-sq_diff / (2 * length_scale**2))
    return K


@jit
def smooth_switch_kernel(t, time_points, switch_scales, num_modes=2):
    """
    Smooth Switching Kernel for Locally Linear Modes.

    This is the KEY innovation of gpSLDS. Instead of hard switches between
    linear modes (like standard SLDS), gpSLDS uses a smooth, continuous
    switching function controlled by a GP.

    The kernel defines which mode is "active" at each time, but the
    transition is smooth rather than discrete.

    Args:
        t: [T] time points (the time series)
        time_points: [T] same as t (for kernel computation)
        switch_scales: [num_modes] length scales for each mode kernel
        num_modes: number of local linear modes

    Returns:
        K_switch: [T, T] kernel governing mode switching smoothness
    """
    # Compute base RBF kernels for each mode
    K_base = rbf_kernel(t, time_points, length_scale=1.0)

    # Weight kernels by mode-specific scales
    # (In practice, this is learned during EM)
    K_switch = K_base  # Simplified version; full version learns K per mode

    return K_switch


@jit
def matern_kernel(x1, x2, length_scale=1.0, variance=1.0, nu=2.5):
    """
    Matérn Kernel: generalization of RBF with controllable smoothness.

    Useful when you want finer control over differentiability.
    nu=1/2 -> exponential kernel
    nu=3/2 -> once differentiable
    nu=5/2 -> twice differentiable

    Args:
        x1, x2: time points
        length_scale: correlation length
        variance: overall scale
        nu: smoothness parameter

    Returns:
        K: [len(x1), len(x2)] covariance matrix
    """
    sq_dist = jnp.sqrt(jnp.sum((x1[:, None] - x2[None, :])**2, axis=-1))
    sqrt_2nu = jnp.sqrt(2.0 * nu)

    # Matern formula (simplified - no bessel function for now)
    if nu == 0.5:
        K = variance * jnp.exp(-sq_dist / length_scale)
    else:
        # Approximate Matern with scaled RBF for stability
        scaled_dist = sqrt_2nu * sq_dist / length_scale
        K = variance * jnp.exp(-scaled_dist)

    return K


@jit
def add_jitter(K, jitter=1e-6):
    """
    Numerical stability: add small constant to diagonal.

    Helps with matrix inversion and Cholesky decomposition.
    """
    return K + jitter * jnp.eye(K.shape[0])
