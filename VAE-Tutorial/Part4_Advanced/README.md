# Part 4: Advanced Methods - Gaussian Process Switching Linear Dynamical Systems (gpSLDS)

## Overview

**gpSLDS** extends the ideas from Part 3 (LFADS) to model more complex neural dynamics using:
- **Multiple local linear modes**: Neural dynamics that switch between different regimes
- **Gaussian Process smoothing**: Continuous, smooth transitions between modes (not hard switches)
- **Principled uncertainty quantification**: GP framework provides confidence estimates

## What You'll Learn

1. **Why gpSLDS?** - Limitations of single-mode LFADS
2. **Gaussian Processes 101** - Kernels, priors, function spaces
3. **Switching Linear Systems** - How to compose multiple dynamics modes
4. **Smooth Mode Transitions** - The gpSLDS innovation over standard SLDS
5. **Inference & Learning** - Variational EM for gpSLDS
6. **JAX Implementation** - Code patterns and efficient computation
7. **Discovering Neural Dynamics** - Real applications

## Prerequisites

- ✅ Complete Part 0-3 (especially Part 3: LFADS)
- ✅ Comfort with: variational inference, neural VAE models, latent dynamics
- ✅ Optional: JAX basics (we'll teach JAX patterns as we go)

## Time Requirements

- **Self-study (full depth)**: 3-4 hours
- **Presentation (theory + demo)**: 1-1.5 hours
- **Quick review (concepts only)**: 30-45 min

## Files

- `08_gpslds.ipynb` - Main teaching notebook (you are here)
- `gpslds_modules/` - Core implementation (adapted from lindermanlab/gpslds)
  - `kernels.py` - Gaussian process kernels
  - `likelihoods.py` - Observation models (Gaussian, Poisson)
  - `em.py` - EM learning algorithm
  - `inference.py` - Kalman filter & RTS smoother
  - `utils.py` - Utilities and data generation
- `data/` - Synthetic data for demonstrations
  - `generate_demo_data.py` - Script to create datasets

## Attribution

This material is adapted from:
- **Official gpSLDS**: https://github.com/lindermanlab/gpslds (Pandarinath et al., 2018)
- **Stanford STATS 320**: https://slinderman.github.io/ml4nd/

See `references/ATTRIBUTION.md` for detailed credits.

## Key References

- **Main Paper**: Pandarinath et al. (2018). "Inferring single-trial neural population dynamics using sequential autoencoders." *Nature Methods*.
  - This Part goes beyond LFADS (Part 3) by adding multiple switching modes and GP smoothness

- **Related**: Linderman et al. (2017). "A latent contaminant model for robust clustering of microbiota reads." 
  - Shows how GP priors help with dynamics modeling

## Common Questions

**Q: Do I need to understand all the math?**
A: The notebook provides both intuitive explanations and detailed math. Skim the math if you prefer; the code demonstrates the concepts.

**Q: Is this production-ready code?**
A: No—it's educational code optimized for clarity, not speed. For research use, use the official `lindermanlab/gpslds` package.

**Q: How does this compare to LFADS (Part 3)?**

| Aspect | LFADS | gpSLDS |
|--------|-------|--------|
| Latent dynamics | Single linear + non-linearity | Multiple switching linear modes |
| Uncertainty | Point estimates | GP provides uncertainty |
| Mode switching | Implicit in RNN | Explicit & smooth |
| Complexity | Simpler, faster | More flexible, slower |

---

**Ready to dive in?** Start with Section 1 of `08_gpslds.ipynb` to understand the motivation.
