"""
Utility functions for gpSLDS: data generation, matrix operations, etc.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap


@jit
def softmax(z, axis=-1):
    """Numerically stable softmax."""
    z_max = jnp.max(z, axis=axis, keepdims=True)
    ez = jnp.exp(z - z_max)
    return ez / jnp.sum(ez, axis=axis, keepdims=True)


@jit
def logistic(x):
    """Sigmoid / logistic function."""
    return 1.0 / (1.0 + jnp.exp(-x))


def generate_synthetic_gpslds_data(T, D_latent, D_obs, num_modes=2,
                                   seed=0, likelihood="gaussian"):
    """
    Generate synthetic data from a true gpSLDS model.

    This is used in the demo notebook to show that gpSLDS can
    recover known ground-truth dynamics.

    Args:
        T: number of time steps
        D_latent: latent dimensionality
        D_obs: observation dimensionality
        num_modes: number of local linear modes
        seed: random seed
        likelihood: "gaussian" or "poisson"

    Returns:
        y: [T, D_obs] observations
        x_true: [T, D_latent] true latent states
        mode_seq: [T] true mode sequence
        params_true: dict of true parameters
    """
    key = jax.random.PRNGKey(seed)

    # True parameters
    A_modes = jnp.array([
        jax.random.normal(key, (D_latent, D_latent)) * 0.95  # Stable dynamics
        for _ in range(num_modes)
    ])

    Q = 0.1 * jnp.eye(D_latent)  # Process noise
    C = jax.random.normal(key, (D_obs, D_latent)) * 0.5  # Observation matrix
    R = 0.5 * jnp.eye(D_obs) if likelihood == "gaussian" else None  # Obs noise

    # Simulate trajectory
    x = jnp.zeros((T, D_latent))
    mode_seq = jax.random.randint(key, (T,), 0, num_modes)
    y = jnp.zeros((T, D_obs))

    # Simple sequential generation (not optimized)
    for t in range(1, T):
        mode = mode_seq[t]
        x = x.at[t].set(A_modes[mode] @ x[t-1] +
                        jax.random.normal(key, (D_latent,)) * jnp.sqrt(0.1))

        if likelihood == "gaussian":
            y = y.at[t].set(C @ x[t] +
                           jax.random.normal(key, (D_obs,)) * jnp.sqrt(0.5))
        else:  # poisson
            rate = jnp.clip(C @ x[t], 0, None)
            y = y.at[t].set(jax.random.poisson(key, jnp.exp(rate)))

    params_true = {
        'A': A_modes,
        'Q': Q,
        'C': C,
        'R': R,
        'pi_0': jnp.ones(num_modes) / num_modes,
    }

    return y, x, mode_seq, params_true


def normalize_data(y, axis=0):
    """Normalize observations to zero mean, unit variance."""
    mean = jnp.mean(y, axis=axis, keepdims=True)
    std = jnp.std(y, axis=axis, keepdims=True)
    return (y - mean) / (std + 1e-8), mean, std


def reshape_for_batching(y, window_size):
    """Reshape time series into overlapping windows for batching."""
    T, D = y.shape
    num_windows = T - window_size + 1
    windows = jnp.zeros((num_windows, window_size, D))
    for i in range(num_windows):
        windows = windows.at[i].set(y[i:i+window_size])
    return windows
