# Variational Neural Inference

An executable tutorial on probabilistic modeling, variational inference, and
latent dynamical systems for neural data. The eleven notebooks form a progression
from tensors and latent variables to sequential VAEs and interpretable nonlinear
dynamics.

## Course Map

| Part | Notebook | Role | Recommended use |
|---|---|---|---|
| 0 | `00_pytorch_basics.ipynb` | Exercise-based PyTorch primer | Self-study and assignments |
| 0 | `01_pytorch_neuroscience_introduction.ipynb` | Visual PyTorch introduction | Seminar prerequisite |
| 1 | `02_probabilistic_modeling.ipynb` | Probability and latent-variable foundations | Core |
| 1 | `02b_mixtures_em.ipynb` | Mixture models, EM, the ELBO, and stochastic EM | Core bridge |
| 2 | `03a_hmm_foundations.ipynb` | HMM inference, decoding, sampling, and Baum-Welch EM | Core |
| 2 | `03_hmm_lds.ipynb` | Gaussian and AR-HMM extended workshop | Advanced plus optional application |
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
    -> mixture models and EM
    -> temporal states and dynamic programming
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

Use notebooks 00 and 01 as prerequisites. The complete classical inference
track is `02 -> 02b -> 03a`; notebook 03 extends it to AR-HMMs and optional
neural/behavioral data. Use notebook 05 for explicit coordinate-ascent
inference and notebook 07 as an advanced LFADS reference.

## Repository Layout

```text
Variational-Neural-Inference/
|-- README.md
|-- requirements-all.txt
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
    |   |-- 02_probabilistic_modeling.ipynb
    |   `-- 02b_mixtures_em.ipynb
    |-- Part2_Dynamics/
    |   |-- 03a_hmm_foundations.ipynb
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

### One-command setup for a fresh clone

Downloading individual notebooks is not sufficient for notebooks 07 and 08.
They use external implementations registered as Git submodules. Clone the
repository, enter it, and run the setup command:

```bash
git clone https://github.com/msenselab/Variational-Neural-Inference.git
cd Variational-Neural-Inference
python scripts/setup_all.py
```

`setup_all.py` performs all required installation steps:

1. initializes both Git submodules;
2. applies the modern-JAX and NumPy compatibility patches;
3. uses `uv pip` to install the complete CPU-compatible environment for
   notebooks 00-08.

The script automatically detects `uv`. If `uv` is unavailable, it falls back
to standard `pip`. To require uv rather than allowing fallback, run
`python scripts/setup_all.py --installer uv`.

The command is idempotent and can be run again safely. Preview its actions
without changing the environment with `python scripts/setup_all.py --dry-run`.

The local Git settings used by a contributor to hide patched submodule files
are not transferred to other computers. The tracked patch files and the script
are the reproducible source of those compatibility changes.

The all-in-one command installs CPU JAX because CUDA builds depend on the local
driver and CUDA version. GPU users should install the matching JAX build from
the official JAX installation guide after setup.

### Smaller installations

Use a dedicated environment and one of the following files when the complete
environment is not needed.

### Core PyTorch notebooks

```bash
uv pip install -r requirements-core.txt
```

### HMM, NWB, and video workshop

```bash
uv pip install -r requirements-hmm-nwb.txt
```

For the full NVIDIA GPU environment used by the VAE/LFADS notebooks on
Windows, install the CUDA overlay instead of `requirements-all.txt`:

```bash
uv pip install -r requirements-gpu.txt
```

### JAX LFADS and gpSLDS notebooks

```bash
uv pip install -r requirements-jax.txt
```

GPU-enabled JAX and PyTorch installations depend on the local CUDA version;
follow the official framework installation instructions when GPU acceleration
is required.

### Which environment should I install?

| Notebooks | Requirements | Hardware and data notes |
|---|---|---|
| 00-08 | `requirements-all.txt` | Complete CPU-compatible environment used by `setup_all.py` |
| 00-06 | `requirements-core.txt` | CPU supported; CUDA recommended for notebook 06 |
| 03 optional NWB/video route | `requirements-hmm-nwb.txt` | Downloads an approximately 578 MB archive |
| 07-08 | `requirements-jax.txt` | GPU recommended for full LFADS; notebook 08 teaching demo runs on CPU |

Notebook 05 downloads the Kato dataset only when its optional real-data flag is
enabled. The default synthetic workflow does not require external data.

## Reproducibility Boundaries

- Notebook 05 runs its complete synthetic CAVI and variational EM tutorial
  without the Kato dataset. The real-data section is opt-in.
- Notebook 06 splits trials before creating loaders: 950 training trials and 50
  genuinely held-out test trials.
- Notebook 03a is the self-contained HMM core. Notebook 03 is an extended
  workshop; its MoSeq/NWB and crowd-movie sections
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
