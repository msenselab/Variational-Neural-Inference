# Variational Neural Inference

An executable tutorial on probabilistic modeling, variational inference, and
latent dynamical systems for neural data. The nine notebooks form a progression
from tensors and latent variables to sequential VAEs and interpretable nonlinear
dynamics.

## Course Map

| Part | Notebook | Role | Recommended use |
|---|---|---|---|
| 0 | `00_pytorch_basics.ipynb` | Exercise-based PyTorch primer | Self-study and assignments |
| 0 | `01_pytorch_neuroscience_introduction.ipynb` | Visual PyTorch introduction | Seminar prerequisite |
| 1 | `02_probabilistic_modeling.ipynb` | Probability and latent-variable foundations | Core |
| 2 | `03_hmm_lds.ipynb` | Gaussian HMM and AR-HMM extended workshop | Core sections plus optional application |
| 3 | `04_standard_vae.ipynb` | Standard VAE and amortized inference | Core teaching anchor |
| 3 | `05_variational_em.ipynb` | CAVI and variational EM | Core synthetic tutorial; optional Kato data |
| 3 | `06_lfads.ipynb` | Transparent PyTorch LFADS | Main sequential VAE lesson |
| 3 | `07_FULL_LFADS_Tutorial.ipynb` | Full JAX LFADS workflow | Advanced reference |
| 4 | `08_gpslds.ipynb` | gpSLDS and interpretable nonlinear dynamics | Advanced conceptual capstone |

Each notebook begins with its role, expected runtime, hardware recommendation,
data requirements, and validation scope.

## Conceptual Progression

```text
probability
    -> latent variables
    -> temporal states
    -> variational inference
    -> recurrent neural dynamics
    -> nonlinear but locally interpretable dynamics
```

- Mixture models introduce hidden variables.
- HMMs add temporal structure.
- VAEs add amortized inference.
- LFADS adds recurrent latent dynamics for spike trains.
- gpSLDS adds uncertain nonlinear dynamics assembled from local linear regimes.

## Recommended Seminar Narrative

The clearest main presentation uses four notebooks:

1. `02_probabilistic_modeling.ipynb`: why latent variables are useful.
2. `04_standard_vae.ipynb`: the ELBO and amortized inference.
3. `06_lfads.ipynb`: a temporal VAE for neural spike trains.
4. `08_gpslds.ipynb`: interpretable nonlinear latent dynamics.

Use notebooks 00 and 01 as prerequisites, notebook 03 for classical temporal
models, notebook 05 for explicit coordinate-ascent inference, and notebook 07
as an advanced LFADS reference.

## Repository Layout

```text
Variational-Neural-Inference/
|-- README.md
|-- requirements-core.txt
|-- requirements-hmm-nwb.txt
|-- requirements-jax.txt
|-- external/
|   |-- computation-thru-dynamics/  # Git submodule for notebook 07
|   `-- gpslds/                      # Git submodule for notebook 08
|-- references/
`-- VAE-Tutorial/
    |-- Part0_Foundations/
    |   |-- 00_pytorch_basics.ipynb
    |   `-- 01_pytorch_neuroscience_introduction.ipynb
    |-- Part1_Basics/
    |   `-- 02_probabilistic_modeling.ipynb
    |-- Part2_Dynamics/
    |   `-- 03_hmm_lds.ipynb
    |-- Part3_Variational/
    |   |-- 04_standard_vae.ipynb
    |   |-- 05_variational_em.ipynb
    |   |-- 06_lfads.ipynb
    |   `-- 07_FULL_LFADS_Tutorial.ipynb
    `-- Part4_Advanced/
        |-- 08_gpslds.ipynb
        `-- README.md
```

## Installation

Clone the external implementations together with the main repository:

```bash
git clone --recurse-submodules <repository-url>
cd Variational-Neural-Inference
```

For an existing clone:

```bash
git submodule update --init --recursive
python scripts/apply_external_patches.py
```

The patch command is idempotent. It applies the small modern-JAX and NumPy 2
compatibility adjustments used by notebooks 07 and 08, and reports when they
are already present.

Use a dedicated environment for each dependency tier.

### Core PyTorch notebooks

```bash
python -m pip install -r requirements-core.txt
```

### HMM, NWB, and video workshop

```bash
python -m pip install -r requirements-hmm-nwb.txt
```

### JAX LFADS and gpSLDS notebooks

```bash
python -m pip install -r requirements-jax.txt
```

GPU-enabled JAX and PyTorch installations depend on the local CUDA version;
follow the official framework installation instructions when GPU acceleration
is required.

## Reproducibility Boundaries

- Notebook 05 runs its complete synthetic CAVI and variational EM tutorial
  without the Kato dataset. The real-data section is opt-in.
- Notebook 06 splits trials before creating loaders: 950 training trials and 50
  genuinely held-out test trials.
- Notebook 03 is an extended workshop. The MoSeq/NWB and crowd-movie sections
  are optional and require a large download.
- Notebook 07 is a dependency-heavy reference implementation rather than the
  default live demonstration.
- Notebook 08 is a teaching demonstration of the gpSLDS computational core,
  not a replacement for the full upstream variational EM fitting pipeline.

## Model Comparison

| Model | Latent representation | Temporal dynamics | Observation model | Inference | Main interpretation |
|---|---|---|---|---|---|
| HMM | Discrete state | Markov transitions | Gaussian or autoregressive | Forward-backward and EM | Recurring behavioral regimes |
| VAE | Continuous latent | Independent observations | Neural decoder | Amortized variational inference | Nonlinear low-dimensional structure |
| LFADS | Continuous generator state and inferred inputs | Recurrent neural network | Poisson spikes | Sequential amortized inference | Trial-specific neural dynamics |
| gpSLDS | Continuous latent state with local regimes | GP-SDE vector field | Gaussian or Poisson | Variational GP inference | Smooth, uncertain, locally linear dynamics |

## Sources and Attribution

The tutorial adapts material from Stanford STATS 320 and the original model
implementations. See `references/ATTRIBUTION.md` for detailed attribution.

- STATS 320 course: <https://slinderman.github.io/ml4nd/>
- LFADS paper: Pandarinath et al. (2018), *Nature Methods*.
- VAE paper: Kingma and Welling (2013), *Auto-Encoding Variational Bayes*.
- gpSLDS paper: Hu et al. (2024), *NeurIPS*.
- Computation through dynamics: Vyas et al. (2020), *Annual Review of Neuroscience*.

## Current Status

The tutorial is organized as a public-facing release candidate. The compact
core notebooks and capstone have executable workflows; dependency-heavy and
external-data sections are explicitly marked as advanced or optional rather
than being implied prerequisites.
