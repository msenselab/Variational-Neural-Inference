# Part 4: Advanced Methods

Part 4 introduces the Gaussian Process Switching Linear Dynamical System
(gpSLDS), a probabilistic model for learning uncertain nonlinear latent dynamics
from neural population activity.

## Notebook

- `08_gpslds.ipynb` presents the model, the smoothly switching linear kernel,
  sparse GP conditioning, Poisson observations, quadrature under latent-state
  uncertainty, and local stability analysis.

The notebook uses selected core classes from the official implementation in
`../../external/gpslds`. It is a self-contained teaching workflow rather than a
replacement for the complete variational EM training pipeline.

## Learning Objectives

After completing the notebook, you should be able to:

- distinguish an SLDS with discrete switches from a gpSLDS with smooth,
  state-dependent regime boundaries;
- explain how local linear kernels preserve interpretable dynamics;
- construct and inspect the smoothly switching linear kernel;
- use inducing points to compute a sparse GP posterior over a vector field;
- propagate latent-state uncertainty with Gaussian quadrature;
- interpret fixed points and Jacobian eigenvalues of learned dynamics;
- identify the array shapes required by the full upstream fitting code.

## Prerequisites

- Part 2: hidden Markov models and linear dynamical systems
- Part 3: variational inference, variational EM, and latent neural dynamics
- Basic familiarity with JAX and Gaussian distributions

## Implementation Note

The tutorial runs with the repository's current JAX environment on CPU. The
vendored upstream source contains a small pure-JAX compatibility adjustment for
the kernel utilities, avoiding a TensorFlow Probability import conflict with
NumPy 2. The full upstream fitting demo has additional dependencies and is best
run in a dedicated environment with GPU-enabled JAX.

## References

- Hu, A., Zoltowski, D., Nair, A., Anderson, D., Duncker, L., and Linderman, S.
  (2024). *Modeling Latent Neural Dynamics with Gaussian Process Switching
  Linear Dynamical Systems*. NeurIPS 2024.
- Official implementation: <https://github.com/lindermanlab/gpslds>
- Paper: <https://arxiv.org/abs/2408.03330>
