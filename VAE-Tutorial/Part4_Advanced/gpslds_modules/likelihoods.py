"""
Observation Models (Likelihoods) for gpSLDS.

Neural data can be observed as:
- Continuous voltage recordings (Gaussian likelihood)
- Spike counts (Poisson likelihood)

This module defines both observation models used in variational inference.
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit


@jit
def gaussian_loglik(y, y_mean, y_var):
    """
    Gaussian observation likelihood.

    log p(y_t | x_t) = -0.5 * (y_t - mean_t)^2 / var_t - 0.5 * log(2π * var_t)

    Used for continuous neural recordings (e.g., LFP, voltage-clamp).

    Args:
        y: [T, D] observations
        y_mean: [T, D] predicted mean from linear dynamics
        y_var: [D] observation variance (learned parameter)

    Returns:
        ll: [T, D] log likelihood per timestep per dimension
    """
    # Gaussian log likelihood formula
    ll = (-0.5 * jnp.power(y - y_mean, 2) / y_var -
          0.5 * jnp.log(2 * jnp.pi * y_var))
    return ll


@jit
def poisson_loglik(y, rate, add_eps=1e-10):
    """
    Poisson observation likelihood for spike counts.

    log p(y_t | x_t) = y_t * log(rate_t) - rate_t - log(y_t!)

    Used for spike count data from neural recordings.

    Args:
        y: [T, D] spike counts (non-negative integers)
        rate: [T, D] predicted firing rate from dynamics
        add_eps: numerical stability term

    Returns:
        ll: [T, D] log likelihood per timestep per neuron
    """
    rate = jnp.clip(rate, add_eps, None)  # Avoid log(0)

    # Poisson log likelihood (ignoring constant factorial term)
    ll = y * jnp.log(rate) - rate

    return ll


@jit
def sample_gaussian_observation(y_mean, y_var, key):
    """Sample from Gaussian observation model."""
    return y_mean + jnp.sqrt(y_var) * jax.random.normal(key, y_mean.shape)


@jit
def sample_poisson_observation(rate, key):
    """Sample from Poisson observation model."""
    return jax.random.poisson(key, rate)
