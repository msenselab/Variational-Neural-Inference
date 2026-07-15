"""
Generate synthetic gpSLDS data for demonstrations in the notebook.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle


def generate_switching_dynamics_data(T=1000, D_latent=3, D_obs=20,
                                     num_modes=3, switch_period=200,
                                     likelihood="gaussian", seed=42):
    """
    Generate synthetic data from a gpSLDS model with known dynamics.

    Args:
        T: number of time steps
        D_latent: latent dimensionality
        D_obs: observation dimensionality
        num_modes: number of local linear modes
        switch_period: how often to switch modes (for structured switching)
        likelihood: "gaussian" or "poisson"
        seed: random seed

    Returns:
        data_dict: {
            'y': [T, D_obs] observations,
            'x_true': [T, D_latent] true latent states,
            'mode_seq': [T] true mode index at each time,
            'A_true': [num_modes, D_latent, D_latent] true dynamics,
            'C_true': [D_obs, D_latent] true observation matrix,
            'Q_true': [D_latent, D_latent] process noise,
            'R_true': [D_obs, D_obs] observation noise,
        }
    """
    key = jax.random.PRNGKey(seed)

    # Generate true parameters
    keys = jax.random.split(key, 5)

    # Dynamics matrices: stable, contractive
    A_true = jax.random.normal(keys[0], (num_modes, D_latent, D_latent)) * 0.3
    A_true = A_true / jnp.linalg.norm(A_true, axis=(1, 2), keepdims=True)
    A_true = A_true * 0.9  # Ensure stability

    # Observation matrix
    C_true = jax.random.normal(keys[1], (D_obs, D_latent)) * 0.5

    # Noise covariances
    Q_true = 0.1 * jnp.eye(D_latent)
    if likelihood == "gaussian":
        R_true = 0.5 * jnp.eye(D_obs)
    else:
        R_true = None

    # Mode sequence: periodic switching for structure
    mode_seq = jnp.array([i % num_modes for i in range(T)])

    # Simulate latent trajectories
    x_true = jnp.zeros((T, D_latent))
    key = keys[2]

    for t in range(1, T):
        mode = mode_seq[t]
        noise = jax.random.normal(key, (D_latent,)) * jnp.sqrt(0.1)
        x_true = x_true.at[t].set(A_true[mode] @ x_true[t-1] + noise)
        key, _ = jax.random.split(key)

    # Generate observations
    if likelihood == "gaussian":
        y = C_true @ x_true.T  # [D_obs, T]
        key = keys[3]
        noise = jax.random.normal(key, y.shape) * jnp.sqrt(0.5)
        y = y + noise
        y = y.T  # [T, D_obs]
    else:  # poisson
        y = jnp.zeros((T, D_obs))
        for t in range(T):
            rate = jnp.clip(jnp.exp(C_true @ x_true[t]), 1e-2, 100)
            key, _ = jax.random.split(key)
            y = y.at[t].set(jax.random.poisson(key, rate))

    # Return as dict
    data_dict = {
        'y': np.array(y),
        'x_true': np.array(x_true),
        'mode_seq': np.array(mode_seq),
        'A_true': np.array(A_true),
        'C_true': np.array(C_true),
        'Q_true': np.array(Q_true),
        'R_true': np.array(R_true) if likelihood == "gaussian" else None,
        'T': T,
        'D_latent': D_latent,
        'D_obs': D_obs,
        'num_modes': num_modes,
        'likelihood': likelihood,
    }

    return data_dict


if __name__ == "__main__":
    # Generate and save demo datasets
    data_gauss = generate_switching_dynamics_data(
        T=1000, D_latent=3, D_obs=20, num_modes=3,
        likelihood="gaussian", seed=42
    )

    data_poisson = generate_switching_dynamics_data(
        T=1000, D_latent=3, D_obs=15, num_modes=2,
        likelihood="poisson", seed=43
    )

    # Save to pickle files
    with open("demo_data_gaussian.pkl", "wb") as f:
        pickle.dump(data_gauss, f)

    with open("demo_data_poisson.pkl", "wb") as f:
        pickle.dump(data_poisson, f)

    print("OK: Generated demo data:")
    print(f"  - Gaussian likelihood: {data_gauss['y'].shape}")
    print(f"  - Poisson likelihood: {data_poisson['y'].shape}")
