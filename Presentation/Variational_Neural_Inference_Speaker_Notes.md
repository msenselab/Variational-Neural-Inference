# Variational Neural Inference: Speaker Notes

Target duration: 60 minutes. All timing cues are approximate.

## 1. Variational Neural Inference

Timing: 1:00

Today I will build one continuous story from the basic language of probability to modern latent dynamical models. The goal is not to catalogue algorithms. The goal is to understand which modeling assumption is added at each step, which posterior quantity must be computed, and what scientific interpretation becomes possible.

## 2. One question connects the entire talk

Timing: 1:15

Neural data are observations, not explanations. Latent-variable models insert an explicit hidden layer between measurement and interpretation. Every model in this talk differs in the structure assigned to that hidden layer and in the method used to infer it.

## 3. Roadmap: each model adds one assumption

Timing: 1:15

The progression is cumulative. Probability gives us uncertainty. Mixtures introduce hidden assignments. HMMs connect those assignments through time. VAEs replace per-datapoint optimization with a learned inference network. LFADS equips the latent state with recurrent dynamics. gpSLDS then trades some neural-network flexibility for local dynamical interpretation.

## 4. 1

Timing: 0:30

I will start from the smallest set of ideas needed for every later model.

## 5. A probabilistic model is a data-generating hypothesis

Timing: 1:40

A probabilistic model says how data could have been generated. Observed variables are what the experiment records. Latent variables are explanatory quantities that are not directly measured. Parameters define the strength and shape of the relationships. This distinction prevents us from treating a low-dimensional visualization as if it were automatically a generative explanation.

## 6. Bayes' rule turns a generative model into inference

Timing: 1:40

Bayes' rule is the central computational operation in latent-variable modeling. The numerator combines the likelihood and prior. The denominator, the marginal likelihood or evidence, sums or integrates over all latent explanations. In simple models this is analytic. In modern models it is usually the source of intractability.

## 7. ↓

Timing: 1:20

It is useful to separate three operations. Inference estimates hidden states for fixed parameters. Learning estimates parameters from data, usually by repeatedly invoking inference. Prediction applies the fitted model to unseen or future observations. Many apparent disagreements between algorithms disappear once we ask which of these operations they approximate.

## 8. Gradients tell us how to change parameters

Timing: 1:50

A gradient is a local sensitivity. It tells us how much the objective would change under a small parameter change. Gradient descent moves in the negative-gradient direction. Backpropagation is not a separate learning principle; it is an efficient application of the chain rule. Automatic differentiation records the computation and evaluates those derivatives exactly up to numerical precision.

## 9. Optimization does not remove modeling assumptions

Timing: 1:25

Better optimization cannot rescue a poorly specified scientific model. We still need to choose an objective, a parameterization, and a validation strategy. Latent models also face identifiability: labels may permute, latent axes may rotate, and multiple parameter settings may induce the same observations. Interpretation therefore requires constraints and validation beyond training loss.

## 10. 2

Timing: 0:30

Mixture models are the smallest setting where uncertainty over a hidden assignment affects parameter learning.

## 11. A mixture model explains heterogeneity

Timing: 1:40

A Gaussian mixture assumes that each observation came from one of K components. The latent variable z is discrete. The mixture weights describe component prevalence; each component has a mean and covariance. The marginal density is a weighted sum because the generating component is unobserved.

## 12. M

Timing: 1:55

If assignments were known, fitting each Gaussian would be easy. If parameters were known, computing assignment probabilities would also be easy. EM exploits this conditional simplicity. The E-step computes soft assignments. The M-step treats those responsibilities as fractional counts. The process monotonically improves or preserves the observed-data likelihood, although it can converge to a local optimum.

## 13. Soft assignments preserve uncertainty

Timing: 1:30

A hard clustering algorithm collapses uncertainty immediately. EM keeps a probability for every component. Points near overlapping regions receive mixed colors and lower confidence. This uncertainty is not cosmetic; it determines how strongly each observation contributes to each component's sufficient statistics.

## 14. +

Timing: 1:55

The ELBO identity is the bridge from EM to variational inference. The log evidence equals a lower bound plus a nonnegative KL divergence. In ordinary EM, the E-step can set q equal to the exact posterior, making the gap zero. The M-step then optimizes the bound with respect to parameters. Variational inference keeps the same logic when the exact posterior is unavailable.

## 15. EM produces a fitted density, not only labels

Timing: 1:25

The scientific output of a mixture model is more than a colored partition. It is a probability density with component frequencies, centers, and uncertainty geometry. The likelihood trace is a basic implementation check, but held-out density and stability across initializations are stronger validation criteria.

## 16. ≠

Timing: 1:20

Mixture observations are exchangeable. The model sees component counts but not order. Two sequences with the same histogram are equivalent. Neural and behavioral data contain persistence, switching, and temporal prediction. The minimal extension is to connect consecutive latent assignments with a Markov transition model.

## 17. 3

Timing: 0:30

An HMM changes one assumption: the hidden assignment now depends on the previous hidden state.

## 18. An HMM adds a Markov chain to a mixture

Timing: 1:45

The hidden state follows a first-order Markov chain: the next state depends on the current state, not the entire past. Each observation depends only on its current state. These conditional-independence assumptions are strong, but they enable exact dynamic programming rather than summing over K to the T possible state sequences.

## 19. Dynamic programming makes exact sequence inference possible

Timing: 2:00

The forward message summarizes all evidence up to time t. The backward message summarizes evidence after time t. Their product gives the smoothed marginal posterior for each state. This is exact under the HMM assumptions and costs order T K squared, rather than enumerating exponentially many state paths. Scaling or log-space calculations are necessary for numerical stability.

## 20. Filtering and smoothing answer different questions

Timing: 1:35

Filtering is the online posterior and is appropriate for real-time decoding. Smoothing is an offline posterior and can revise ambiguous moments using future observations. The distinction matters experimentally: an offline state estimate should not be reported as if it were available causally at that time.

## 21. Three posterior summaries serve different goals

Timing: 1:35

HMM inference does not have one universal output. Marginals quantify uncertainty at each time. Viterbi decoding gives the single most probable joint sequence, which is not the same as choosing the most probable state independently at every time. Posterior path samples preserve sequence-level uncertainty and are often more honest for downstream analyses.

## 22. Baum-Welch is EM for an HMM

Timing: 1:45

Baum-Welch repeats exact HMM inference and parameter updates. The E-step computes smoothed state probabilities and pairwise transition marginals. The M-step updates the initial distribution, transition matrix, and emission parameters. As with mixture EM, labels may permute and local optima remain possible, so recovery is judged after label alignment and across initializations.

## 23. HMM assumptions determine the scientific interpretation

Timing: 1:30

A state label means only what the emission and transition assumptions make it mean. In a Gaussian HMM, states represent recurring observation distributions. In an AR-HMM, each state defines local autoregressive dynamics. In an SLDS, a discrete switch selects continuous linear dynamics. This hierarchy is useful, but increasing expressiveness also weakens identifiability and raises computational cost.

## 24. 4

Timing: 0:30

Exact inference breaks when the decoder becomes nonlinear. Variational inference replaces the intractable posterior with an optimized approximation.

## 25. Variational inference turns inference into optimization

Timing: 1:45

Variational inference chooses a tractable family q and minimizes its KL divergence to the posterior indirectly by maximizing the ELBO. The approximation can be factorized, Gaussian, sequential, or represented by a neural network. A high ELBO does not prove that q captures every posterior dependency; it is an optimization target and a lower bound, not an automatic certificate of scientific validity.

## 26. Amortization learns a reusable inference rule

Timing: 1:35

Classical variational methods optimize local variational parameters separately for every observation. A VAE amortizes that work: an encoder predicts the variational parameters from x. This enables fast inference and generalization to new data, but the shared encoder may fail to reach the best variational parameters for every datapoint, creating an amortization gap.

## 27. The VAE objective balances reconstruction and regularization

Timing: 1:50

The first ELBO term rewards reconstructions under latent samples from the encoder. The KL term regularizes the approximate posterior toward the prior. Too little regularization gives an unstructured code; too much can make the decoder ignore z, known as posterior collapse. This is especially important with powerful sequential decoders.

## 28. A VAE combines a generative model and an inference network

Timing: 1:35

The decoder is the generative model: latent z generates observations. The encoder is an auxiliary inference model: it maps each observation to a distribution over z. Keeping these roles separate clarifies why a VAE is not simply an autoencoder with noise. The probabilistic objective and latent prior are essential.

## 29. The reparameterization trick differentiates through sampling

Timing: 1:45

Direct sampling appears to interrupt the computation graph. Reparameterization writes a Gaussian sample as a deterministic function of encoder outputs and parameter-free noise. Gradients then flow through mu and sigma. This produces a low-variance pathwise gradient estimator and makes end-to-end VAE training practical.

## 30. A latent space is useful only when validated

Timing: 1:30

A clean latent visualization is suggestive, not sufficient. We should test held-out likelihood or reconstruction, stability across seeds, sensitivity to latent dimension and prior, and correspondence with external variables that were not used to train the model. Latent axes can rotate or permute without changing the observation model.

## 31. 5

Timing: 0:30

LFADS extends the VAE from independent observations to entire spike-train sequences generated by a recurrent dynamical system.

## 32. LFADS separates inferred initial condition, dynamics, and rates

Timing: 1:55

LFADS treats a trial as a sequence generated by a dynamical system. The encoder summarizes the observed spikes and infers an initial condition. A recurrent generator evolves that state. A low-dimensional factor readout drives nonnegative firing rates, and spikes are modeled with a Poisson likelihood. More complete variants also infer time-varying inputs.

## 33. The sequential ELBO assigns uncertainty to a trajectory

Timing: 1:50

The LFADS objective is a sequential ELBO. The likelihood evaluates whether inferred rates explain spike counts. KL terms constrain trial-specific initial states and, when present, inferred controller inputs. Inputs are powerful but scientifically delicate: they may represent real unobserved perturbations, or they may compensate for inadequate autonomous dynamics. Their interpretation requires perturbation timing and held-out validation.

## 34. Learned dynamics must generalize beyond training trajectories

Timing: 1:40

For synthetic data, we can compare the learned vector field with the known system. For real data, we instead rely on held-out spike prediction, consistency across trials, perturbation responses, and dynamical invariants. Reconstruction alone can be achieved by a flexible encoder-decoder without recovering the correct autonomous dynamics.

## 35. 6

Timing: 0:30

The final model retains nonlinear flexibility while exposing local linear dynamics and uncertainty.

## 36. gpSLDS: nonlinear dynamics with local regimes

Timing: 1:55

A Gaussian-process switching linear dynamical system places a GP prior on the vector field. Local linear regimes define interpretable dynamical templates, while smooth gating functions blend them. Inducing points make GP inference scalable. The result is a nonlinear field with uncertainty and a local linear decomposition.

## 37. Local linear structure makes nonlinear dynamics inspectable

Timing: 1:40

The key scientific advantage is inspectability. At each location we can examine regime weights, local linear operators, fixed points, and uncertainty in the vector field. However, this interpretation still depends on the latent observation model and identifiability constraints. A smooth partition is not automatically a biological module.

## 38. Choose the simplest model that answers the scientific question

Timing: 2:00

Model choice should follow the scientific question, not a hierarchy of sophistication. Use a mixture when temporal order is irrelevant. Use an HMM for discrete recurring regimes. Use a VAE for nonlinear continuous representations. Use LFADS when trial-level recurrent dynamics are central. Use gpSLDS when nonlinear flow and local dynamical interpretation are both required. More flexible models demand stronger validation.

## 39. A practical validation ladder

Timing: 1:45

Validation should proceed in layers. First verify the implementation using synthetic data and mathematical invariants. Then test predictive generalization. Next test robustness across seeds and modeling choices. Finally ask whether latent variables correspond to external measurements or produce falsifiable perturbation predictions. A compelling visualization without these layers is exploratory evidence, not a mechanistic result.

## 40. Take-home message

Timing: 1:35

The unifying idea is that a generative model makes assumptions explicit, and inference connects those assumptions to observations. EM, HMM inference, VAEs, and sequential latent models are not isolated tricks; they are increasingly expressive solutions to the same posterior-learning problem. The value of the latent state comes from what the model constrains and what independent evidence validates.

