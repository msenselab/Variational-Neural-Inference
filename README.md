# Variational Neural Inference: VAE and Latent Dynamical Systems

A comprehensive, tutorial-style library on **Variational Autoencoders (VAE)** and **Latent Dynamical Systems** modeling for neural data analysis. Includes theory, mathematical derivations, and PyTorch implementations—designed for 1-2 hour group presentations or self-study.

---

## Library Overview

This library is organized into **4 parts**, progressing from foundations to advanced variational inference:

| Part | Topic | Content | Time |
|------|-------|---------|------|
| **Part 0** | Foundations | PyTorch basics, tensors, autograd | 15 min |
| **Part 1** | Probability & Models | Distribution theory, MLE, Bayesian inference, mixture models, EM | 2-3 hours |
| **Part 2** | Dynamics | Hidden Markov Models, Linear Dynamical Systems | 2 hours |
| **Part 3** | Variational Inference | VAE, Variational EM, LFADS for neural dynamics | 2-3 hours |

---

## Folder Structure

```
Variational-Neural-Inference/
│
├── README.md (this file)
├── references/
│   ├── references.bib
│   └── ATTRIBUTION.md
│
├── Part0_Foundations/
│   └── 00_pytorch_basics.ipynb
│
├── Part1_Basics/
│   ├── 01_probabilistic_modeling.ipynb
│   └── 02_mixture_models.ipynb
│
├── Part2_Dynamics/
│   └── 03_hmm_lds.ipynb
│
└── Part3_Variational/
    ├── 04_standard_vae.ipynb
    ├── 05_variational_em.ipynb
    └── 06_lfads.ipynb
```

**Total**: 7 notebooks, ~7.2 MB, fully self-contained

---

## Getting Started

### Installation

```bash
pip install torch numpy scipy matplotlib scikit-learn jupyter seaborn
```

### Recommended Learning Path

**For 1-2 hour group presentation:**
1. Part 0: Quick PyTorch intro (10 min)
2. Theory lecture with slides (15 min)
3. Live demo: `04_standard_vae.ipynb` (45 min)
4. Live demo: `06_lfads.ipynb` (45 min)

**For self-study (12-15 hours total):**
1. Part 0: `00_pytorch_basics.ipynb` (1 hour)
2. Part 1: Notebooks 1-2 (3-4 hours)
3. Part 2: `03_hmm_lds.ipynb` (2 hours)
4. Part 3: Notebooks 4-6 (5-6 hours)

---

## Key Concepts

### What's Covered

- **Probabilistic Modeling**: Distributions, likelihood, Bayesian inference
- **Maximum Likelihood Estimation**: Theory and applications
- **Conjugate Priors**: Gamma-Poisson example
- **Mixture Models**: Clustering and latent variables
- **Expectation-Maximization**: Algorithm for hidden variable models
- **Variational Inference**: ELBO, mean-field approximation
- **Variational Autoencoders**: Encoder-decoder architecture, reparameterization trick
- **Latent Dynamical Systems**: Modeling neural population dynamics
- **LFADS**: Sequential VAE for discovering latent factors in neural data

### Notebook Contents

| Notebook | Focus | Key Topics |
|----------|-------|-----------|
| `00_pytorch_basics` | Setup | Tensors, broadcasting, autograd |
| `01_probabilistic_modeling` | Foundations | Poisson, MLE, Bayesian, conjugate priors, MAP |
| `02_mixture_models` | Clustering | Gaussian mixtures, EM algorithm |
| `03_hmm_lds` | Sequences | Hidden Markov Models, Linear Dynamical Systems |
| `04_standard_vae` | Generative | Encoder/decoder, ELBO loss, training |
| `05_variational_em` | Theory | Variational inference, posterior approximation |
| `06_lfads` | Applications | Sequential VAE, neural dynamics discovery |

---

## Source & Attribution

This library is **adapted from Stanford STATS 320: Machine Learning Methods for Neural Data Analysis**.

- **Official Course**: https://slinderman.github.io/ml4nd/
- **Instructor**: Scott Linderman
- **Institution**: Stanford University, Department of Statistics

### What We Did

✅ Adapted Stanford course materials for VAE and latent dynamics  
✅ Converted to fully English notebooks  
✅ Complete implementations with working code  
✅ Tutorial-style organization for teaching  
✅ Focused scope (VAE + dynamics, not full course)

See `references/ATTRIBUTION.md` for detailed credits.

---

## Key Papers

- **LFADS**: Pandarinath et al. (2018). "Inferring single-trial neural population dynamics using sequential autoencoders." *Nature Methods*.
- **VAE**: Kingma & Welling (2013). "Auto-Encoding Variational Bayes." *arXiv preprint*.
- **EM**: Dempster, Laird, Rubin (1977). "Maximum likelihood from incomplete data via the EM algorithm."
- **Neural Dynamics**: Vyas et al. (2020). "Computation through neural population dynamics." *Annual Review of Neuroscience*.

See `references/references.bib` for complete bibliography.

---

## Requirements

- **Python**: 3.8+
- **Core Libraries**: PyTorch, NumPy, SciPy, Matplotlib
- **Optional**: scikit-learn, seaborn
- **Environment**: Jupyter Notebook

---

## How to Use

### Option 1: Local Jupyter

```bash
cd Variational-Neural-Inference
jupyter notebook
# Open any Part X folder and start with Part 0
```

### Option 2: Group Presentation

1. Open notebooks in presentation mode
2. Run cells live during presentation
3. Use slides for theory (not included; create your own or use notebook markdown)

### Option 3: Self-Study

Follow the recommended learning path above.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| PyTorch not found | `pip install torch` |
| Notebook won't load | Ensure Jupyter is installed: `pip install jupyter` |
| Graphics not showing | `%matplotlib inline` should be automatic |
| Missing data errors | Some notebooks download data on first run (needs internet) |

---

## What's NOT Included

- **Part 4 (Applications)**: Real neural data case studies (future work)
- **Presentation slides**: Create using your favorite tool
- **Exercises solutions**: You should work through these!
- **Complete course content**: Focus on VAE + dynamics only

---

## Citation

If you use this library in your work, please cite:

```bibtex
@online{linderman_stats320,
  title={Machine Learning Methods for Neural Data Analysis},
  author={Linderman, Scott},
  year={2024},
  url={https://slinderman.github.io/ml4nd/},
  school={Stanford University}
}
```

---

## Questions?

- **About Stanford materials**: See https://slinderman.github.io/ml4nd/
- **About specific notebooks**: Review the markdown sections in each notebook
- **About LFADS**: Check the original paper (Pandarinath et al., 2018)

---

**Last Updated**: 2026-07-15  
**Status**: ✅ Complete and ready for teaching  
**Next Step**: Start with Part 0, run a notebook, and dive in!
