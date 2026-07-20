"""Reduce 03_hmm_lds.ipynb to its phenomenon-to-latent-dynamics main line."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "VAE-Tutorial" / "Part2_Dynamics" / "03_hmm_lds.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def clean_code(cell: dict, *, hidden: bool = False) -> dict:
    cell = dict(cell)
    cell["execution_count"] = None
    cell["outputs"] = []
    cell["metadata"] = dict(cell.get("metadata", {}))
    if hidden:
        cell["metadata"]["jupyter"] = {"source_hidden": True}
    return cell


def strip_docstrings(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def selected_definitions(source: str, names: set[str]) -> str:
    tree = ast.parse(source)
    pieces = [
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    return "\n\n".join(piece for piece in pieces if piece) + "\n"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if len(notebook["cells"]) < 98:
        original = subprocess.run(
            [
                "git",
                "-c",
                "safe.directory=D:/Variational-Neural-Inference",
                "show",
                "HEAD:VAE-Tutorial/Part2_Dynamics/03_hmm_lds.ipynb",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
        notebook = json.loads(original)
    old = notebook["cells"]

    intro = markdown(
        r"""
# From Behavioral Phenomena to Latent Dynamics

Freely moving behavior looks continuous and high-dimensional, yet recurring
sub-second motifs suggest that a smaller latent process may organize it. We use
mouse depth-video principal components as observations and ask:

> Can a latent-state model discover reusable behavioral states and their dynamics?

The path is deliberately narrow:

1. observed behavior and low-dimensional trajectories;
2. a discrete latent-state hypothesis;
3. HMM inference with forward–backward and EM;
4. Gaussian HMMs for state-dependent posture;
5. AR-HMMs for state-dependent short-timescale dynamics;
6. the conceptual step from AR-HMMs to continuous-state LDS models.

![Behavioral motifs](https://ars.els-cdn.com/content/image/1-s2.0-S0896627315010375-gr1.jpg)

**References:** Wiltschko et al. (2015), *Neuron*; Markowitz et al.
(2018), *Cell*.
"""
    )

    helper_source = (
        'sns.set_context("notebook")\n'
        'palette = sns.xkcd_palette(["windows blue", "red", "medium green", '
        '"dusty purple", "greyish", "orange", "amber", "clay", "pink"])\n\n'
        + selected_definitions(
            "".join(old[4]["source"]),
            {
                "plot_data_and_states",
                "extract_syllable_slices",
                "plot_average_pcs",
            },
        )
    )
    helper_cell = clean_code(old[4], hidden=True)
    helper_cell["source"] = strip_docstrings(helper_source)

    load_cell = clean_code(old[16], hidden=True)
    load_source = "".join(load_cell["source"])
    load_source = load_source.replace("load_frames: bool = True", "load_frames: bool = False")
    load_source = load_source.replace(
        "load_dataset(num_pcs=data_dim, indices=indices)",
        "load_dataset(num_pcs=data_dim, indices=indices, load_frames=False)",
    )
    load_cell["source"] = strip_docstrings(load_source)

    standardize = clean_code(old[20], hidden=True)
    standardize["source"] = strip_docstrings("".join(standardize["source"]))

    def implementation(index: int) -> dict:
        cell = clean_code(old[index], hidden=True)
        cell["source"] = strip_docstrings("".join(cell["source"]))
        return cell

    def result(index: int) -> dict:
        return clean_code(old[index])

    def compact_fit(index: int) -> dict:
        cell = clean_code(old[index])
        source = "".join(cell["source"])
        source = source.replace("num_states = 50", "num_states = 20")
        source = source.replace(
            "            observations)",
            "            observations,\n            num_iters=15)",
        )
        cell["source"] = source
        return cell

    cells = [
        intro,
        markdown(
            """
## 1. From observations to a modeling question

The raw video is high-dimensional. Precomputed principal components retain the
dominant posture variation and give a multivariate trajectory $x_t$. The colored
background in the first visualization is an external MoSeq labeling used only
as orientation—not as supervision for fitting our models.
"""
        ),
        clean_code(old[2]),
        clean_code(old[3]),
        helper_cell,
        markdown(
            """
### Load the MoSeq/NWB application data

The setup cell downloads the public archive only when it is absent. The main
analysis loads eight sessions and ten principal components; video frames are
disabled because they are not needed for the modeling argument.
"""
        ),
        clean_code(old[6], hidden=True),
        load_cell,
        standardize,
        result(22),
        markdown(
            r"""
## 2. The discrete latent-state hypothesis

An HMM assumes that an unobserved discrete state $z_t$ evolves according to a
Markov transition matrix and generates the observation $x_t$:

$$p(z_{1:T},x_{1:T})=p(z_1)\prod_{t=2}^T p(z_t\mid z_{t-1})
\prod_{t=1}^T p(x_t\mid z_t).$$

The scientific interpretation is concrete: each state represents a recurring
behavioral regime, while transitions describe how behavior switches over time.
"""
        ),
        markdown(
            """
### Posterior inference: forward–backward

Forward messages combine past evidence with the current observation; backward
messages carry future evidence. Their product yields the smoothed state
posterior $p(z_t\mid x_{1:T})$. The implementation is retained below as
supporting code, but the inference principle—not the bookkeeping—is the main
point.
"""
        ),
        implementation(26),
        implementation(28),
        implementation(32),
        implementation(36),
        implementation(42),
        markdown(
            r"""
## 3. Gaussian HMM: state-dependent posture

The Gaussian HMM gives every discrete state its own mean posture and covariance:

$$p(x_t\mid z_t=k)=\mathcal N(x_t\mid \mu_k,\Sigma_k).$$

EM alternates between inferring state probabilities (E-step) and updating the
state-specific observation parameters (M-step).
"""
        ),
        implementation(45),
        implementation(48),
        implementation(50),
        implementation(52),
        compact_fit(54),
        markdown("### What did the Gaussian HMM discover?"),
        result(56),
        result(58),
        result(60),
        markdown(
            """
### Limitation

A Gaussian HMM clusters instantaneous posture. It can identify recurring
regions of observation space, but it does not explicitly describe how posture
evolves *within* a state. That limitation motivates an autoregressive emission
model.
"""
        ),
        markdown(
            r"""
## 4. AR-HMM: state-dependent local dynamics

An AR-HMM lets each state define a local dynamical rule:

$$p(x_t\mid x_{t-1:t-G},z_t=k)
=\mathcal N\!\left(\sum_{g=1}^{G}A_{k,g}x_{t-g}+b_k,Q_k\right).$$

The latent state now selects not just a posture distribution, but a short-term
movement pattern.
"""
        ),
        implementation(73),
        implementation(75),
        implementation(77),
        compact_fit(79),
        markdown("### What did the AR-HMM discover?"),
        result(81),
        result(83),
        clean_code({
            **old[85],
            "source": "plot_average_pcs(arhmm_order[3], train_dataset[:1], train_posteriors[:1])\n",
        }),
        markdown(
            r"""
## 5. From AR-HMMs to LDS models

The AR-HMM still makes the *observed* trajectory autoregressive and keeps the
latent state discrete. A linear dynamical system introduces a continuous latent
trajectory $h_t$ instead:

$$h_t=A h_{t-1}+\epsilon_t,\qquad y_t=C h_t+\eta_t.$$

This distinction matters:

| Model | Latent state | What evolves dynamically? |
|---|---|---|
| Gaussian HMM | discrete $z_t$ | state sequence |
| AR-HMM | discrete $z_t$ | observed trajectory within each state |
| LDS | continuous $h_t$ | latent trajectory |
| Switching/GP-LDS | discrete + continuous | regime-specific latent dynamics |

Thus the main progression is **behavioral motifs → discrete latent states →
local dynamics → continuous latent dynamics**. The GP-SLDS notebook continues
from this point.
"""
        ),
        markdown(
            """
## Take-home message

- High-dimensional behavior can be summarized by a lower-dimensional trajectory.
- HMMs turn recurring structure into interpretable discrete latent states.
- Gaussian emissions describe state-dependent posture.
- Autoregressive emissions add state-dependent local dynamics.
- LDS-family models move the dynamics into a continuous latent space, providing
  the bridge to modern neural latent-dynamics models.
"""
        ),
    ]

    for index, cell in enumerate(cells):
        cell["id"] = f"mainline-{index:02d}"

    notebook["cells"] = cells
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Pruned {len(old)} cells to {len(cells)} cells: {NOTEBOOK}")


if __name__ == "__main__":
    main()
