# From EM to Variational Inference: A Complete Tutorial

**Machine Learning Methods for Neural Data Analysis**  
*Scott Linderman STATS 220/320*

---

## Table of Contents

1. [Lecture 12: Foundations of the EM Algorithm](#lecture-12-foundations-of-the-em-algorithm)
2. [Lecture 13: Linear Dynamical Systems and SLDS](#lecture-13-linear-dynamical-systems-and-slds)
3. [Lecture 14: Variational Inference](#lecture-14-variational-inference)
4. [Applications and Implementation](#applications-and-implementation)

---

# Lecture 12: Foundations of the EM Algorithm

## 1. Problem Setting: Uncertainty in Clustering

### Scenario: Recognizing Behavior in Video

Imagine you have a video of animal behavior and need to identify what behavior is happening in each frame:
- Walking
- Resting
- Grooming

```
Raw video frame sequence:
y₁ → y₂ → y₃ → y₄ → y₅ → ... → y_T

Latent variables (true behavior):
z₁   z₂   z₃   z₄   z₅        z_T
```

### Naive Approach vs. Improved Approach

#### K-Means (hard assignment):
```
Assign each frame a definite behavior label
Frame 1: 100% Walking
Frame 2: 100% Resting
Frame 3: 100% Grooming
```

**Problem**: What about frames at the boundaries? It's unclear which category they belong to.

#### EM Algorithm (soft assignment):
```
Assign each frame a probability distribution
Frame 1: 50% Walking, 30% Resting, 20% Grooming
Frame 2: 10% Walking, 80% Resting, 10% Grooming
Frame 3: 5% Walking, 5% Resting, 90% Grooming
```

**Advantage**: Better matches real-world uncertainty and makes better use of the information.

---

## 2. Gaussian Mixture Model

### Model Definition

Assume each observation $y_t$ comes from one of $K$ Gaussian distributions:

$$y_t \sim \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(y_t | \mu_k, \Sigma_k)$$

where:
- $\pi_k$: the **prior probability** of choosing class $k$, satisfying $\sum_{k=1}^{K} \pi_k = 1$
- $\mu_k$: the **mean** (center) of class $k$
- $\Sigma_k$: the **covariance** (shape) of class $k$
- $z_t \in \{1,\ldots,K\}$: the **latent variable**, indicating the true class

### Joint Distribution

$$p(y_t, z_t) = p(z_t) \cdot p(y_t | z_t) = \pi_{z_t} \cdot \mathcal{N}(y_t | \mu_{z_t}, \Sigma_{z_t})$$

### The Key Problem

**We can only observe $y_t$; we never see $z_t$!**

We need to infer the latent variables from the observations.

---

## 3. K-Means: The Hard-Assignment Method

### Algorithm Steps

**Initialization**: randomly choose $K$ initial centers $\mu_1^{(0)}, \ldots, \mu_K^{(0)}$

**Iteration**: repeat the following steps until convergence

#### Step 1 (Assignment)
Assign each data point to the nearest cluster center:

$$z_t \leftarrow \arg\max_k \left[-\frac{1}{2}\|y_t - \mu_k\|_2^2\right] = \arg\min_k \|y_t - \mu_k\|_2^2$$

This is equivalent to maximizing:
$$\log \mathcal{N}(y_t | \mu_k, I)$$

#### Step 2 (Update)
Recompute each cluster's center:

$$\mu_k \leftarrow \frac{1}{N_k}\sum_{t: z_t=k} y_t$$

where $N_k = \sum_t \mathbb{1}[z_t=k]$ is the number of points assigned to class $k$.

### Visualization

```
Initialize        Assign            Update            Reassign
  ×                × ○              × ○                × ○
 × ×              × ○ ○             × + ○             × + ○
× × ×    →       × ○ ○ ○      →   × + + ○      →   × + + ○
 × ×             × ○ ○             × + ○             × + ○
  ×              × ○               × ○                × ○

(× = class 1, ○ = class 2, + = new centers)
```

### The Problem with K-Means

```
Reality: there is a point on a fuzzy boundary
        ╱─ Class 1
    • ╱  ← what about this point?
      ╱─ Class 2

K-Means forces it into a single class,
but it might actually belong to both!
```

---

## 4. The EM Algorithm: The Soft-Assignment Method

### Core Idea

**Instead of a hard assignment, compute the probability that each point belongs to each class**

### E-Step (Expectation): Compute Responsibilities

For each data point $y_t$ and each class $k$, compute the **posterior probability**:

$$q_{t,k} = \Pr(z_t = k | y_t; \theta) = \frac{\pi_k \mathcal{N}(y_t | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(y_t | \mu_j, \Sigma_j)}$$

This is called the **responsibility** or the **soft-assignment weight**.

#### Intuitive Explanation

- $q_{t,k}$ is the probability that $y_t$ belongs to class $k$, given the observation and the current parameters
- $\sum_k q_{t,k} = 1$ (probabilities sum to 1)
- If the model is confident $y_t$ belongs to some class, $q_{t,k}$ is close to 1; otherwise it is spread out

#### Example

Suppose $\mu_1 = 0, \mu_2 = 10$ and the observation is $y_t = 5$:

```
If π₁ = π₂ = 0.5 (equal probability)
and Σ₁ = Σ₂ = 1 (same variance)

q_{t,1} = 𝒩(5|0,1) / (𝒩(5|0,1) + 𝒩(5|10,1))
        = exp(-12.5) / (exp(-12.5) + exp(-12.5))
        = 0.5

q_{t,2} = 𝒩(5|10,1) / (𝒩(5|0,1) + 𝒩(5|10,1))
        = exp(-12.5) / (exp(-12.5) + exp(-12.5))
        = 0.5

Result: this point contributes equally to both classes!
```

### M-Step (Maximization): Update Parameters Using Soft Assignments

#### Update the means

Use a **weighted average** in place of the hard-assignment average:

$$\mu_k \leftarrow \frac{\sum_t q_{t,k} y_t}{\sum_t q_{t,k}} = \frac{1}{N_k} \sum_t q_{t,k} y_t$$

where the **effective number of data points** is defined as:
$$N_k = \sum_t q_{t,k}$$

This is the "effective" number of data points assigned to class $k$ (usually not an integer).

#### Update the covariances

$$\Sigma_k \leftarrow \frac{\sum_t q_{t,k}(y_t - \mu_k)(y_t - \mu_k)^T}{\sum_t q_{t,k}} = \frac{1}{N_k} \sum_t q_{t,k}(y_t - \mu_k)(y_t - \mu_k)^T$$

#### Update the priors

$$\pi_k \leftarrow \frac{\sum_t q_{t,k}}{T} = \frac{N_k}{T}$$

### E-step vs. M-step: Comparison with K-Means

| Step | K-Means | EM |
|------|---------|-----|
| **Assignment** | $z_t = \arg\max_k \mathcal{N}(y_t \vert \mu_k, I)$ | $q_{t,k} = \frac{\pi_k \mathcal{N}(y_t \vert \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(y_t \vert \mu_j, \Sigma_j)}$ |
| **Nature of assignment** | Hard (0 or 1) | Soft (between 0 and 1) |
| **Update** | $\mu_k \leftarrow \frac{1}{N_k}\sum_{t:z_t=k} y_t$ | $\mu_k \leftarrow \frac{1}{N_k}\sum_t q_{t,k} y_t$ |
| **Flexibility** | Low (variance fixed at I) | High (learns variance and prior) |

---

## 5. Mathematical Foundations of EM

### Why Do the E-step and M-step Work?

The EM algorithm maximizes the **log marginal likelihood**:

$$\log p(y_1:T; \theta) = \log \sum_{z_1:T} p(y_1:T, z_1:T | \theta)$$

### The Problem: Hard to Compute Directly

This sum has $K^T$ terms (for $T$ data points and $K$ classes), which is generally intractable.

### The EM Trick: A Lower Bound (ELBO)

Introduce a **lower bound**, called the **Evidence Lower Bound (ELBO)**:

$$\log p(y_1:T; \theta) \geq \mathbb{E}_{q}[\log p(y_1:T, z_1:T | \theta)] - \mathbb{E}_q[\log q(z_1:T)]$$

where $q(z_1:T)$ is an arbitrary distribution.

### Three Key Facts

**Fact 1**: The ELBO is a lower bound on the log likelihood
$$\log p(y; \theta) \geq \text{ELBO}(q, \theta)$$

**Fact 2**: Equality holds if and only if $q(z) = p(z|y; \theta)$ (the true posterior)

**Fact 3**: The EM algorithm alternates:
- **E-step**: fix $\theta$, optimize $q$ to maximize the ELBO
- **M-step**: fix $q$, optimize $\theta$ to maximize the ELBO

### Convergence

The EM algorithm guarantees:
1. The ELBO increases monotonically
2. The algorithm converges to a **local maximum of the likelihood**
3. Each iteration improves (or leaves unchanged) the log probability of the data

### Intuitive Illustration

```
log likelihood: log p(y|θ)
              ↑
              │         true posterior
              │          (q=p)
        ELBO ┤ ╱────────●────────
              │╱          ╲
              │            ╲ KL divergence
              │             ╲
        initial q ┤─────────────●
              │
        parameter space →θ

Each EM iteration ascends, eventually converging
```

---

## 6. Gaussian HMM: Adding Temporal Structure

### From GMM to HMM

**GMM**: each data point is independent
$$p(y_1:T, z_1:T) = \prod_t p(z_t) p(y_t | z_t)$$

**HMM**: add temporal dependence
$$p(y_1:T, z_1:T) = p(z_1) \prod_t p(z_t | z_{t-1}) \prod_t p(y_t | z_t)$$

### The Gaussian HMM Model

**Latent dynamics** (Markov chain):
$$z_1 \sim \text{Cat}(\pi)$$
$$z_t | z_{t-1} \sim \text{Cat}(P_{z_{t-1}})$$

where $P$ is the $K \times K$ **transition matrix**, $P_{i,j} = \Pr(z_t = j | z_{t-1} = i)$

**Emission model**:
$$y_t | z_t \sim \mathcal{N}(C_{z_t} + d, R)$$

or in unified notation:
$$y_t | z_t \sim \mathcal{N}(Cz_t + d, R)$$

### Parameters

The full HMM parameters:
$$\theta = (\pi, P, \{C_k, d_k, R\}_{k=1}^K)$$

### Why Is the HMM Harder Than the GMM?

#### In a GMM:
$z_t$ are independent, so the posterior factorizes:
$$q(z_1:T) = \prod_t q(z_t)$$

Each time step's posterior can be computed independently.

#### In an HMM:
$z_t$ are correlated through the transition matrix:
$$p(z_1:T) = p(z_1) \prod_t p(z_t | z_{t-1})$$

**The posterior no longer factorizes!**

Example:
```
GMM: z₁ ⊥ z₂ ⊥ z₃ ...
      (independent)

HMM: z₁ → z₂ → z₃ ...
     (Markov chain)

     The observations y impose temporal constraints on z
     An earlier z₁ affects the posterior of a later z₂
```

### The HMM E-step: The Forward-Backward Algorithm

Although the posterior is complex, there is an **efficient recursive algorithm**.

**Forward messages** (from past to present):
$$\alpha_t(z_t) = p(y_1:t, z_t)$$

Recursion:
$$\alpha_{t+1}(z_{t+1}) = p(y_t | z_t) \sum_{z_t} p(z_{t+1} | z_t) \alpha_t(z_t)$$

**Backward messages** (from future to present):
$$\beta_t(z_t) = p(y_{t+1:T} | z_t)$$

Recursion:
$$\beta_t(z_t) = \sum_{z_{t+1}} p(y_{t+1}|z_{t+1}) p(z_{t+1}|z_t) \beta_{t+1}(z_{t+1})$$

**Posterior marginal**:
$$p(z_t | y_1:T) \propto \alpha_t(z_t) \beta_t(z_t)$$

This algorithm computes all posteriors in $O(K^2 T)$ time!

---

## Lecture 12 Summary

### Key Concepts

1. **Soft assignment**: EM handles uncertainty with probabilities instead of hard labels
2. **ELBO**: the lower bound makes optimization tractable
3. **E-step and M-step**: alternately optimize the two parts of the lower bound
4. **Temporal structure**: HMMs introduce Markov dependence, but forward-backward is still tractable

### The Parameter-Learning Loop

```
Initialize parameters θ⁽⁰⁾

Iterate i = 1, 2, 3, ...
  E-step: compute q(z) = p(z | y; θ⁽ⁱ⁻¹⁾)
          (direct for GMM; forward-backward for HMM)

  M-step: θ⁽ⁱ⁾ = argmax_θ E_q[log p(y, z | θ)]
          (has a closed-form solution)

  Convergence check: if the likelihood change < ε, stop

Return the final parameters θ*
```

---

# Lecture 13: Linear Dynamical Systems and SLDS

## 1. From HMM to LDS: Upgrading the Latent Variable

### Dimensional Upgrade

| Property | HMM | LDS |
|------|-----|-----|
| **Latent variable** | $z_t \in \{1,\ldots,K\}$ | $x_t \in \mathbb{R}^d$ |
| **Type** | Discrete | Continuous |
| **Transition** | Probability table ($K \times K$ matrix) | Linear-Gaussian dynamics |
| **Parameters** | $O(K^2)$ | $O(d^2)$ |
| **Expressiveness** | Finite set of modes | Continuous trajectories |

### Intuitive Comparison

**HMM**: the system jumps between a few discrete states
```
State 1 ↔ State 2 ↔ State 3
   ↑
   └────────────────┘
```

**LDS**: the system evolves smoothly in a continuous space
```
x₁ → x₂ → x₃ → ... → x_T
a smooth trajectory in ℝᵈ
```

---

## 2. Gaussian Linear Dynamical System (Gaussian LDS)

### The Full Model

**Dynamics** — how the latent variable evolves over time:

$$x_{t+1} \sim \mathcal{N}(Ax_t + b, Q)$$

**Emission** — how observations are generated from the latent variable:

$$y_t \sim \mathcal{N}(Cx_t + d, R)$$

**Initial distribution**:

$$x_1 \sim \mathcal{N}(m, Q_0)$$

### Parameter Interpretation

#### Dynamics parameters
- **$A$ (d×d dynamics matrix)**: controls how the latent state evolves
  - The eigenvalues of $A$ determine the system's stability
  - $|λ| < 1$: decays to a fixed point
  - $|λ| = 1$: stays cyclic or grows linearly
  - $|λ| > 1$: diverges

- **$b$ (d-dim bias)**: a constant driving force or input

- **$Q$ (d×d process covariance)**: the magnitude of the process noise
  - High variance → more random behavior
  - Low variance → more deterministic behavior

#### Emission parameters
- **$C$ (p×d emission matrix)**: projects the latent variable to the observation space
  - $p$: observation dimension
  - $d$: latent dimension
  - usually $d \ll p$ (dimensionality reduction)

- **$d$ (p-dim bias)**: an offset on the observations

- **$R$ (p×p observation covariance)**: observation noise
  - High noise → unreliable observations
  - Low noise → reliable observations

### The Full Joint Probability

$$p(x_1:T, y_1:T | \theta) = \mathcal{N}(x_1 | m, Q_0) \prod_{t=2}^{T} \mathcal{N}(x_{t} | Ax_{t-1}+b, Q) \prod_{t=1}^{T} \mathcal{N}(y_t | Cx_t+d, R)$$

---

## 3. What Can an LDS Do? Rich Dynamical Behavior

### Theoretical Basis

Although it looks restrictive (only linear!), an LDS can actually model **rich dynamics**.

By changing the eigenvalues and eigenvectors of the $A$ matrix, we can create a variety of behaviors:

### Case 1: Point Attractor — Memory

$$A = \begin{pmatrix} 0.9 & 0 \\ 0 & 0.9 \end{pmatrix}$$

**Behavior**: the system decays to the origin from any initial state
```
Trajectory:
  ↖ ↑ ↗
   \|/
    ● (0,0) ← attractor
   /|\
  ↙ ↓ ↘
```

**Applications**:
- Memory effects
- Decay of neural activity
- Recovery after a decision

### Case 2: Rotational Dynamics — Oscillation

$$A = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

**Behavior**: the latent variable rotates in the 2D plane by angle $\theta$
```
Trajectory:
  ↖ ↑ ↗
   \|/    ↻ rotation
    ●
   /|\
  ↙ ↓ ↘
```

**Applications**:
- Motor control (motor circuits)
- Periodic activity
- Pendulum motion

### Case 3: Line Attractor — Integration

$$A = \begin{pmatrix} 1 & 0 \\ 0 & 0.9 \end{pmatrix}$$

**Behavior**:
- First dimension: preserved (no decay)
- Second dimension: decays

```
Trajectory (phase plane):
  x₂ ↑
     |
  0.9|  ╱──
     | ╱
  0  |────●─── x₁
     |  ╱
```

**Applications**:
- Integrator (accumulates information)
- Visual tracking (position memory)
- Decision integration (evidence accumulation)

### Case 4: Saddle Point — Competition

$$A = \begin{pmatrix} 1.1 & 0 \\ 0 & 0.9 \end{pmatrix}$$

**Behavior**:
- First dimension: amplified
- Second dimension: decays

```
Trajectory:
  x₂ ↑
     |╱─╲
  0  |─●─ x₁
     |╲─╱
```

**Applications**:
- Competitive dynamics (winner-take-all)
- Categorical decisions
- Neural competition

### Key Insight

$$A = PΛP^{-1}$$

where $Λ$ is the matrix of eigenvalues. By controlling the magnitude and sign of the eigenvalues $\lambda_i$, we get:
- $|\lambda| < 1$: decay
- $|\lambda| = 1$: cycling or boundary
- $|\lambda| > 1$: amplification

---

## 4. EM for the LDS: Parameter Fitting

### The Learning Problem

**Given**: an observation sequence $y_1:T$  
**Find**: parameters $\theta = (A, b, Q, C, d, R, m, Q_0)$

### Two-Step EM

#### E-step: infer the latent states

Compute the posterior distribution:
$$q(x_{1:T}) \approx p(x_{1:T} | y_{1:T}; \theta)$$

#### M-step: update the parameters

$$\theta \leftarrow \arg\max_\theta \mathbb{E}_{q}[\log p(x_{1:T}, y_{1:T} | \theta)]$$

### Key Finding: Linear-Gaussian Systems Can Be Solved Exactly!

For a linear-Gaussian system, the **posterior can be computed exactly**.

This is because:
1. Linear transformations preserve Gaussianity
2. The product of Gaussians is still Gaussian
3. The posterior is a Gaussian (or, for the $T$ marginals, a set of Gaussians)

**Algorithm**: the Kalman filter and smoother

---

## 5. The Kalman Filter: Forward Inference

### Basic Idea

At each time $t$, track two quantities:
1. **Prediction**: an estimate of $x_t$ based on information up to $t-1$
2. **Update**: a refined estimate of $x_t$ after seeing $y_t$

### Recursions

Define:
- $\mu_{t|t-1}$ = the predicted mean of $x_t$ before seeing $y_t$
- $\Sigma_{t|t-1}$ = the covariance of that prediction
- $\mu_{t|t}$ = the posterior mean of $x_t$ after seeing $y_t$
- $\Sigma_{t|t}$ = the corresponding posterior covariance

### Algorithm Steps

#### Initialization
$$\mu_{1|0} = m, \quad \Sigma_{1|0} = Q_0$$

#### For each time $t = 1, 2, \ldots, T$:

**1) Prediction**:

From the previous time step's posterior, predict the next state
$$\mu_{t|t-1} = A\mu_{t-1|t-1} + b$$

$$\Sigma_{t|t-1} = A\Sigma_{t-1|t-1}A^T + Q$$

**2) Update**:

After observing $y_t$, refine the estimate of $x_t$

First compute the **Kalman gain**:
$$K_t = \Sigma_{t|t-1}C^T(C\Sigma_{t|t-1}C^T + R)^{-1}$$

Then update the mean and covariance:
$$\mu_{t|t} = \mu_{t|t-1} + K_t(y_t - C\mu_{t|t-1} - d)$$

$$\Sigma_{t|t} = (I - K_tC)\Sigma_{t|t-1}$$

### Intuitive Explanation

#### The Kalman gain $K_t$

$$K_t = \frac{\text{prediction uncertainty}}{\text{prediction uncertainty + observation uncertainty}}$$

- If the prediction is accurate ($\Sigma_{t|t-1}$ small), $K_t$ is small → trust the prediction
- If the observation is accurate ($R$ small), $K_t$ is large → trust the observation

#### The prediction error (innovation)

$$\tilde{y}_t = y_t - C\mu_{t|t-1} - d$$

This is the "surprise" of the observation — the difference between what was actually seen and what was predicted.

### Worked Example

```
Suppose d=1, A=0.9, b=0, R=1

Time t=1: initial prediction
  μ₁|₀ = m, Σ₁|₀ = Q₀
  observe y₁=5
  K₁ = Q₀/(Q₀+1)
  if Q₀=1, K₁=0.5
  → prediction and observation each contribute 50%

Time t=2:
  predict: μ₂|₁ = 0.9·μ₁|₁
  if μ₁|₁=2.5, then μ₂|₁=2.25
  observe y₂=6
  apply the Kalman gain again...

Result: a smooth trajectory balancing dynamics and observations
```

---

## 6. The Kalman Smoother: Backward Correction

### Limitation

The Kalman filter uses only **past information** ($y_1:t$).

But we have the **entire sequence** $y_1:T$! Why not use the future information too?

### The Idea of the Kalman Smoother

**Propagate backward**, combining future information to improve estimates of the past.

### Algorithm

#### Forward pass (same as the Kalman filter)

Compute all $\mu_{t|t}, \Sigma_{t|t}$ and $\mu_{t|t-1}, \Sigma_{t|t-1}$

#### Backward pass

From $t = T-1$ back to $1$, for each $t$ compute:

The **smoothing gain**:
$$J_t = \Sigma_{t|t}A^T \Sigma_{t+1|t}^{-1}$$

Update the mean and covariance:
$$\mu_{t|T} = \mu_{t|t} + J_t(\mu_{t+1|T} - \mu_{t+1|t})$$

$$\Sigma_{t|T} = \Sigma_{t|t} + J_t(\Sigma_{t+1|T} - \Sigma_{t+1|t})J_t^T$$

### Key Properties

- **Variance reduction**: $\Sigma_{t|T} \leq \Sigma_{t|t}$ (smoothing gives smaller uncertainty)
- **Variance reduction**: $\Sigma_{t|T} \leq \Sigma_{t|t-1}$ (better than one-directional prediction)

Intuitive illustration:
```
Kalman filter (past only):
  uncertainty: ▓▓▓▓▓▓▓▓▓▓▓▓
            ↓
           ▓▓▓▓▓▓▓▓▓↘

Kalman smoother (whole sequence):
  uncertainty: ▓▓▓▓▓▓▓▓▓▓▓▓
            ↓
           ▓▓▓▓▓▓   ↘
                ↗ ▓

The smoother yields a cleaner trajectory
```

---

## 7. EM for the LDS — The Complete Loop

### Closed-Form M-step

Given the sufficient statistics (from the Kalman smoother):
- $\mathbb{E}[x_t]$
- $\text{Cov}[x_t]$
- $\mathbb{E}[x_{t+1}x_t^T]$

the parameters have **closed-form updates**:

#### Dynamics parameters

$$A \leftarrow \left(\sum_{t=1}^{T-1} \mathbb{E}[x_{t+1}x_t^T]\right) \left(\sum_{t=1}^{T-1} \mathbb{E}[x_tx_t^T]\right)^{-1}$$

This is **weighted least squares**:
$$A = \arg\min_A \sum_t (x_{t+1} - Ax_t)^2$$

#### Emission parameters

$$C \leftarrow \left(\sum_{t=1}^{T} \mathbb{E}[y_tx_t^T]\right) \left(\sum_{t=1}^{T} \mathbb{E}[x_tx_t^T]\right)^{-1}$$

Similarly, this is also least squares:
$$C = \arg\min_C \sum_t (y_t - Cx_t)^2$$

### The Complete EM-for-LDS Loop

```
Initialize parameters θ⁽⁰⁾

Iterate i = 1, 2, 3, ...

  E-step:
    • Kalman filter: forward pass, obtain μₜ|ₜ, Σₜ|ₜ
    • Kalman smoother: backward pass, obtain μₜ|ₜ, Σₜ|ₜ
    • Compute sufficient statistics: E[xₜ], Cov[xₜ], E[xₜxₜ₊₁ᵀ]

  M-step:
    • Compute A, b, Q, C, d, R from the sufficient statistics
    • (closed-form solution, no optimization needed)

  Convergence check

Return the final parameters θ*
```

### Important Properties

1. **Exact inference**: for an LDS, the E-step can be computed exactly
2. **Scalability**: computational complexity $O(d^3 T)$, where $d$ is the latent dimension
3. **Global optimum**: for a given $\theta$, the Kalman smoother gives the optimal $q$
4. **Convergence**: EM guarantees the likelihood increases monotonically, converging to a local maximum

---

## 8. Switching Linear Dynamical Systems (SLDS)

### Upgrading the Problem: The Dynamics Are Not Fixed

Now assume the **system switches between different "modes"**.

### A Concrete Example: Mouse Behavior

A mouse has several behavioral modes:
- **Walking mode**: moving quickly, position grows linearly
- **Resting mode**: staying still, position unchanged
- **Grooming mode**: activity in place, small position fluctuations

The system switches between these modes.

### The Full SLDS Model

#### Discrete-state (mode) transitions — the HMM part

$$z_t | z_{t-1} \sim \text{Cat}(P_{z_{t-1}})$$

where $P$ is the $K \times K$ transition matrix.

#### Continuous-state dynamics — mode-dependent

$$x_{t+1} | x_t, z_t \sim \mathcal{N}(A_{z_t}x_t + b_{z_t}, Q_{z_t})$$

Key: **each mode $z_t$ has its own dynamics matrix $A_{z_t}$**

#### Shared emission

$$y_t | x_t \sim \mathcal{N}(Cx_t + d, R)$$

(usually the emission is assumed to be the same across all modes)

### Graphical Model

```
z₁ → z₂ → z₃ → ... → z_T    (discrete: mode transitions)
 ↓    ↓    ↓         ↓
x₁ → x₂ → x₃ → ... → x_T    (continuous: state evolution)
 ↓    ↓    ↓         ↓
y₁   y₂   y₃  ...  y_T     (observations)

z_t depends on z_{t-1} (HMM)
x_t depends on x_{t-1} and z_t (LDS, but the dynamics depend on z_t)
y_t depends only on x_t
```

### Joint Distribution

$$p(z_{1:T}, x_{1:T}, y_{1:T} | \theta) = p(z_1) \prod_{t=2}^T p(z_t|z_{t-1}) \prod_{t=1}^T p(x_t | x_{t-1}, z_t) \prod_{t=1}^T p(y_t | x_t)$$

where:
- $p(z_1) = \pi_{z_1}$
- $p(z_t|z_{t-1}) = P_{z_{t-1}, z_t}$
- $p(x_t | x_{t-1}, z_t) = \mathcal{N}(A_{z_t}x_{t-1}+b_{z_t}, Q_{z_t})$
- $p(y_t | x_t) = \mathcal{N}(Cx_t+d, R)$

---

## 9. The Difficulty with SLDS: Why Exact EM Fails

### The Problem: What Does the Posterior Look Like?

We want to compute:
$$p(z_{1:T}, x_{1:T} | y_{1:T})$$

### Explosion of Mixture Components

**The key observation**:

Because of the discrete $z_{1:T}$, the posterior is a **mixture of Gaussians**.

- How many possible mode sequences are there? $K^T$
- For each sequence, $p(x_{1:T}|z_{1:T})$ is Gaussian
- **Total posterior**: a mixture of $K^T$ Gaussians!

### Concrete Numbers

Suppose $T=100$ time steps and $K=3$ modes:

$$\text{number of mixture components} = 3^{100} \approx 5 \times 10^{47}$$

This is an **astronomical** number.

No computer can store or manipulate this many components.

### The Failure of Naive Message Passing

If you try the forward-backward algorithm:

$$\alpha_t(z_t, x_t) = \sum_{z_{t-1}} \int p(x_t, z_t | x_{t-1}, z_{t-1}) p(y_t | x_t, z_t) \alpha_{t-1}(z_{t-1}, x_{t-1}) dx_{t-1}$$

For each discrete state $z_t$, $\alpha_t$ is a Gaussian.

**The number of components grows exponentially with time**:

```
Time 1: K Gaussian components
Time 2: K × K = K² components
Time 3: K × K² = K³ components
...
Time t: K^t components

At t=100, K=3: 3^100 components 😱
```

### Why Does This Happen?

At each time step we must condition on **all possible mode sequences** from the previous step.

The number of mode sequences grows exponentially with time.

---

## 10. The Solution: Variational Inference

Since exact inference is intractable, we **approximate** the posterior.

### The Mean-Field Variational Family

Constrain the posterior to be a **product of independent factors**:

$$q(z_{1:T}, x_{1:T}) = q(z_{1:T}) \cdot q(x_{1:T})$$

This assumption says: **in our approximation, $z_{1:T}$ and $x_{1:T}$ are independent**.

In the true posterior they are highly correlated, but this assumption makes inference tractable.

### The Variational E-step: Two Sub-steps

#### 1) Update $q(z_{1:T})$

Given $q(x_{1:T})$, the optimal $q(z_{1:T})$ has the **form of an HMM posterior**:

$$q(z_{1:T}) \propto p(z_1) \prod_t p(z_t|z_{t-1}) \exp\left\{\mathbb{E}_{q(x)}[\log p(x_t|x_{t-1}, z_t)]\right\}$$

Define the **emission score** (really an expected log likelihood):

$$l_t(z_t) := \mathbb{E}_{q(x)}[\log p(x_t|x_{t-1}, z_t)]$$

then:
$$q(z_{1:T}) \propto p(z_1) \prod_t p(z_t|z_{t-1}) \prod_t e^{l_t(z_t)}$$

**This looks like a standard HMM, but the emission probability is replaced by $e^{l_t(z_t)}$!**

**Solution**:
- Use the **forward-backward algorithm** to compute the full $q(z_{1:T})$
- Or use the **Viterbi algorithm** to find the most likely mode sequence $z^* = \arg\max_z q(z)$

#### 2) Update $q(x_{1:T})$

Given $q(z_{1:T})$, the optimal $q(x_{1:T})$ has the **form of an LDS posterior**:

$$q(x_{1:T}) \propto p(x_1) \prod_t p(x_t | x_{t-1}, z_t^*) p(y_t | x_t)$$

where we use the discrete state $z_t^*$ (usually the most likely value or expectation under $q(z)$).

This can be written as:
$$q(x_{1:T}) \propto \mathcal{N}(x_1 | m, Q_0) \prod_t \mathcal{N}(x_t | A_{z_t^*}x_{t-1}+b_{z_t^*}, Q_{z_t^*}) \prod_t \mathcal{N}(y_t | Cx_t+d, R)$$

**This is an LDS problem!**

**Solution**: use the **Kalman smoother** to compute $q(x_{1:T})$

---

## 11. The Complete Variational EM Loop

### Algorithm

```
Initialize parameters θ⁽⁰⁾

Iterate i = 1, 2, 3, ...

  E-step (variational inference):
    repeat until convergence:

      • Update q(z):
        - if using mean-field + fully factored z:
          use Viterbi or forward-backward to find the optimal q(z)
        - compute the emission scores l_t(z_t)

      • Update q(x):
        - given q(z) or z^*
        - use the Kalman filter and smoother
        - obtain μₜ|ₜ, Σₜ|ₜ

      • Check ELBO convergence

  M-step:
    • Compute sufficient statistics (from q(z) and q(x))
    • Update θ = (A_k, b_k, Q_k, C, d, R, P, π)

  Convergence check: if the likelihood or ELBO change < ε, stop

Return the final parameters θ*
```

### Important Properties

1. **Tractability**: variational inference is now feasible!

2. **Computing sufficient statistics**:

We need to compute:
$$\mathbb{E}_{q(z,x)}[\mathbb{1}[z_t=k]], \quad \mathbb{E}_{q(z,x)}[\mathbb{1}[z_t=k]x_t], \quad \mathbb{E}_{q(z,x)}[\mathbb{1}[z_t=k]x_t x_{t+1}^T]$$

These can be computed separately from $q(z)$ and $q(x|z)$.

3. The **parameter updates** have closed-form solutions (just like the LDS)

---

## Lecture 13 Summary

| Aspect | LDS | SLDS (exact) | SLDS (variational) |
|------|-----|----------|-----------|
| **Latent variables** | $x_t$ (continuous) | $z_t, x_t$ | $z_t, x_t$ |
| **Posterior** | single Gaussian | $K^t$ Gaussians | two independent parts |
| **Inference** | Kalman smoothing | intractable | HMM + LDS |
| **Complexity** | $O(d^3 T)$ | $O(K^{2T})$ | $O(K^2 T + d^3 T)$ |
| **Exactness** | exact | exact | approximate |
| **Practicality** | ✓ | ✗ | ✓ |

---

# Lecture 14: Variational Inference

## 1. Recap of the Core Problem

### The SLDS Predicament

The true posterior $p(z_{1:T}, x_{1:T} | y_{1:T})$ cannot be computed exactly, because it has $K^T$ mixture components.

### A New Idea

Rather than computing it exactly, find a **simple, tractable distribution** $q$ to approximate it.

**Goal**: find the best approximation
$$q^* = \arg\min_q D_{KL}(q(z,x) \| p(z,x|y))$$

---

## 2. Kullback-Leibler (KL) Divergence

### Definition

The dissimilarity between two distributions $q$ and $p$:

$$D_{KL}(q \| p) = \mathbb{E}_q\left[\log\frac{q(x)}{p(x)}\right] = \mathbb{E}_q[\log q(x)] - \mathbb{E}_q[\log p(x)]$$

### Rewritten as Expectations

$$D_{KL}(q \| p) = \sum_x q(x) \log q(x) - \sum_x q(x) \log p(x)$$

First term: the **self-information** (negative entropy) of $q$  
Second term: the **cross-entropy** of $q$ under $p$

### Key Properties

1. **Non-negativity**: $D_{KL}(q \| p) \geq 0$

   Proof: by Jensen's inequality
   $$D_{KL}(q \| p) = \mathbb{E}_q\left[\log\frac{q}{p}\right] \geq \log\mathbb{E}_q\left[\frac{q}{p}\right] = \log 1 = 0$$

2. **Equals zero if and only if** $q = p$ (on the support of $q$)

3. **Asymmetric**: $D_{KL}(q \| p) \neq D_{KL}(p \| q)$

### Intuitive Understanding

```
KL divergence measures: "If I use q in place of p, how much information do I lose?"

High D_KL: q and p are very different
Low  D_KL: q is close to p
0    D_KL: q equals p
```

### The Important Difference Between the Two Directions

#### Forward KL: $D_{KL}(q \| p)$

Characteristics when minimized:
- Heavily penalized when $q$ has support where $p$ does not
- Result: $q$ tends to cover all the modes of $p$
- $q$ tends to be **wide**

```
p = a mixture of two Gaussians (bimodal)
q = a single Gaussian

Minimizing D_KL(q||p):
q "averages" between the two peaks
```

#### Reverse KL: $D_{KL}(p \| q)$

Characteristics when minimized:
- Heavily penalized when $p$ has support where $q$ does not
- Result: $q$ tends to concentrate on a single mode of $p$
- $q$ tends to be **narrow**

```
Same problem:
Minimizing D_KL(p||q):
q picks the most likely peak and ignores the others
```

### The Choice in EM

In variational inference, we minimize the **forward KL** $D_{KL}(q \| p)$:
- This gives us a more "averaged" posterior
- It covers multiple modes of the true posterior
- More stable, but possibly less focused

---

## 3. The Core Insight of Variational Inference

### Problem Statement

**Given**: observations $y$  
**Find**: the posterior $p(z, x | y)$ — **cannot be computed**

**Goal**: find a simple distribution $q(z, x)$ to approximate it

$$q^* = \arg\min_q D_{KL}(q(z,x) \| p(z, x|y))$$

### The Key Mathematical Trick

Write out the KL divergence:

$$D_{KL}(q \| p(\cdot|y)) = \mathbb{E}_q[\log q] - \mathbb{E}_q[\log p(z,x|y)]$$

Using Bayes' rule $p(z,x|y) = \frac{p(z,x,y)}{p(y)}$:

$$= \mathbb{E}_q[\log q] - \mathbb{E}_q[\log p(z,x,y)] + \mathbb{E}_q[\log p(y)]$$

$$= \mathbb{E}_q[\log q] - \mathbb{E}_q[\log p(z,x,y)] + \log p(y)$$

(since $p(y)$ is constant with respect to $q$)

Rearranging:

$$\log p(y) = \underbrace{\mathbb{E}_q[\log p(z,x,y) - \log q(z,x)]}_{\text{ELBO}} + D_{KL}(q \| p)$$

---

## 4. The ELBO: Evidence Lower Bound

### Definition

$$\text{ELBO}(q) := \mathbb{E}_q[\log p(z,x,y)] - \mathbb{E}_q[\log q(z,x)]$$

can also be written as:

$$\text{ELBO}(q) = \mathbb{E}_q[\log \frac{p(z,x,y)}{q(z,x)}]$$

### The Key Relationship

$$\log p(y) = \text{ELBO}(q) + D_{KL}(q \| p)$$

### Why Is It Called a "Lower Bound"?

Since $D_{KL}(q \| p) \geq 0$, we have:

$$\text{ELBO}(q) \leq \log p(y)$$

**The ELBO is a lower bound on the log likelihood!**

### The Key Corollary

Minimizing $D_{KL}(q \| p)$ is equivalent to maximizing $\text{ELBO}(q)$:

$$\min_q D_{KL}(q \| p) \Leftrightarrow \max_q \text{ELBO}(q)$$

because $\log p(y)$ is constant.

### Intuitive Illustration

```
log likelihood log p(y)
│
│  ∣
│  ├─ D_KL(q||p) ≥ 0
│  │
│  ▼
│  ELBO(q) ─────── the lower bound we can compute
│
└─────────────────→ choice of q

When q = p:
  ELBO(q) = log p(y) (maximized)
  D_KL = 0

When q ≠ p:
  ELBO(q) < log p(y) (a lower bound)
  D_KL > 0 (the gap)
```

---

## 5. The Mean-Field Variational Family

### The Problem

If we let $q$ be arbitrarily complex, optimization becomes very hard.

### The Solution

**Constrain** $q$ to belong to a **variational family** $\mathcal{Q}$, a class of simple distributions.

### The Mean-Field Assumption

Assume all variables are **mutually independent**:

$$q(z, x) = q(z) \cdot q(x)$$

or more generally ($n$ variables):

$$q(z_1, \ldots, z_n) = \prod_{i=1}^n q_i(z_i)$$

### The Variational Family

$$\mathcal{Q}_{MF} = \left\{q : q(z,x) = q(z) \cdot q(x)\right\}$$

### Why Mean-Field?

1. **Easy to optimize**: each factor can be updated independently
2. **Easy to compute**: expectations factor into products
3. **Effective in practice**: usually gives a reasonable approximation
4. **Good theoretical properties**: convergence can be proven

### The Cost: Ignoring Correlations

The mean-field assumption ignores the **correlations** between variables.

**In the true posterior**: $z_t$ and $x_t$ are highly correlated
- If in "walking mode," $x_t$ has a particular pattern
- If the position changes quickly, "walking mode" is more likely

**Under the mean-field assumption**: $z_t$ and $x_t$ are independent
- This is mathematically incorrect
- But it makes inference tractable
- In practice the error is usually acceptable

---

## 6. Coordinate Ascent Variational Inference (CAVI)

### The Problem

We now have the variational family $q(z) q(x)$.

How do we **maximize the ELBO**?

### The Answer: Coordinate Ascent

Optimize each factor in turn.

### Algorithm

```
Initialize q(z), q(x)

repeat:
  1. Fix q(x), optimize q(z)
  2. Fix q(z), optimize q(x)
  3. Check convergence

until the ELBO change < ε
```

### The Key: The Optimal Solution at Each Step

#### Step 1: Fix $q(x)$, optimize $q(z)$

The ELBO:
$$\text{ELBO} = \mathbb{E}_{q(z), q(x)}[\log p(z,x,y)] - \mathbb{E}_{q(z)}[\log q(z)] - \mathbb{E}_{q(x)}[\log q(x)]$$

For optimizing $q(z)$, the third term (which depends on $q(x)$) is constant:

$$\max_{q(z)} \text{ELBO} = \max_{q(z)} \left\{\mathbb{E}_{q(z), q(x)}[\log p(z,x,y)] - \mathbb{E}_{q(z)}[\log q(z)]\right\}$$

$$= \max_{q(z)} \mathbb{E}_{q(z)}\left\{\mathbb{E}_{q(x)}[\log p(z,x,y)] - \log q(z)\right\}$$

This is the problem of minimizing $D_{KL}(q(z) \| \tilde{p}(z))$, where:

$$\tilde{p}(z) \propto \exp\{\mathbb{E}_{q(x)}[\log p(z,x,y)]\}$$

**Optimal solution**:
$$q^*(z) \propto \exp\{\mathbb{E}_{q(x)}[\log p(z,x,y)]\}$$

#### Step 2: Fix $q(z)$, optimize $q(x)$

Similarly:

$$q^*(x) \propto \exp\{\mathbb{E}_{q(z)}[\log p(z,x,y)]\}$$

### General Form (For an Arbitrary Variable)

For any variable $z_i$:

$$q_i^*(z_i) \propto \exp\left\{\mathbb{E}_{q(\text{other variables})}[\log p(\text{all variables})]\right\}$$

The expectation is over all the other variables, and the log is of the full joint distribution.

### Convergence

**Theorem**: CAVI guarantees the ELBO increases monotonically, converging to a **local maximum**.

Proof sketch:
1. Fixing $q(x)$ and optimizing $q(z)$ strictly increases the ELBO
2. Fixing $q(z)$ and optimizing $q(x)$ strictly increases the ELBO
3. The iteration at least stays constant, is bounded, and therefore converges

---

## 7. Application to SLDS

### Model Recap

$$p(z_{1:T}, x_{1:T}, y_{1:T}) = p(z_1) \prod_t p(z_t|z_{t-1}) \prod_t p(x_t|x_{t-1}, z_t) \prod_t p(y_t|x_t)$$

### The Mean-Field Variational Family

$$\mathcal{Q} = \left\{q : q(z_{1:T}, x_{1:T}) = q(z_{1:T}) \cdot q(x_{1:T})\right\}$$

### E-step Part 1: Update $q(z_{1:T})$

Applying CAVI:

$$q^*(z_{1:T}) \propto \exp\{\mathbb{E}_{q(x)}[\log p(z_{1:T}, x_{1:T}, y_{1:T})]\}$$

Expanding:

$$= \exp\left\{\mathbb{E}_{q(x)}\left[\log p(z_1) + \sum_t \log p(z_t|z_{t-1}) + \sum_t \log p(x_t|x_{t-1}, z_t) + \sum_t \log p(y_t|x_t)\right]\right\}$$

The first two terms do not involve $q(x)$:

$$\propto \exp\left\{\log p(z_1) + \sum_t \log p(z_t|z_{t-1}) + \sum_t \mathbb{E}_{q(x)}[\log p(x_t|x_{t-1}, z_t)]\right\}$$

Define:
$$l_t(z_t) := \mathbb{E}_{q(x)}[\log p(x_t|x_{t-1}, z_t)]$$

then:
$$q^*(z_{1:T}) \propto p(z_1) \prod_t p(z_t|z_{t-1}) \prod_t \exp\{l_t(z_t)\}$$

**This is an HMM posterior!**

#### Solution

- Use the **forward-backward algorithm** to obtain the full distribution $q(z_{1:T})$
- Or use the **Viterbi algorithm** to find the most likely sequence $z^* = \arg\max_z q(z)$

### E-step Part 2: Update $q(x_{1:T})$

Applying CAVI:

$$q^*(x_{1:T}) \propto \exp\{\mathbb{E}_{q(z)}[\log p(z_{1:T}, x_{1:T}, y_{1:T})]\}$$

$$\propto \exp\left\{\mathbb{E}_{q(z)}\left[\sum_t \log p(x_t|x_{t-1}, z_t) + \sum_t \log p(y_t|x_t)\right]\right\}$$

(the other terms do not involve $x$)

For a linear-Gaussian model, this simplifies to:

$$q^*(x_{1:T}) \propto p(x_1) \prod_t \mathcal{N}(x_t | A_{z_t^*}x_{t-1}+b_{z_t^*}, Q_{z_t^*}) \prod_t \mathcal{N}(y_t | Cx_t+d, R)$$

where $z_t^*$ comes from $q(z)$.

**This is an LDS posterior!**

#### Solution

Use the **Kalman smoother** to compute all marginals $q(x_t)$.

---

## 8. Variational EM vs. Exact EM

### Exact EM (the ideal case, if possible)

$$q(z,x) = p(z,x|y; \theta) \quad \text{(exact)}$$

- **ELBO**: equals $\log p(y; \theta)$ (attains the maximum)
- **E-step**: exact inference ($K^T$ components, **intractable**)
- **Convergence**: to a local maximum of the likelihood
- **Exactness**: 100% exact

### Variational EM

$$q(z,x) = q(z) q(x) \quad \text{(mean-field approximation)}$$

- **ELBO**: $\leq \log p(y; \theta)$ (a lower bound)
- **E-step**: CAVI (HMM + LDS, **tractable**)
- **Convergence**: to a local maximum of the variational lower bound
- **Exactness**: approximate (error from the mean-field assumption)

### The Good News

Although the ELBO is a lower bound, it is **a valid objective function**.

Maximizing it still makes the model fit the data better.

### Trade-off Summary

```
Exact EM         Variational EM
───────          ──────
Exact: ✓          Exact: ✗ (mean-field error)
Tractable: ✗      Tractable: ✓

Exact is better than approximate, but if the exact
version is uncomputable, then approximate beats nothing!
```

---

## 9. Another Interpretation of the ELBO

### The Two Parts of the ELBO

$$\text{ELBO}(q) = \mathbb{E}_q[\log p(z,x,y)] - \mathbb{E}_q[\log q(z,x)]$$

$$= \underbrace{\mathbb{E}_q[\log p(y,z,x)]}_{\text{joint likelihood}} + \underbrace{\text{Entropy}(q)}_{\text{entropy of q}}$$

### Part One: $\mathbb{E}_q[\log p(y,z,x)]$

This is the **joint log likelihood** of the data and the latent variables.

Maximizing it means:
- Find parameters that make the observations probable
- Find latent-variable explanations that make the joint probable

### Part Two: $\text{Entropy}(q) = -\mathbb{E}_q[\log q(z,x)]$

This is the **entropy** of the posterior $q$.

Maximizing it means:
- Keep $q$ as "flat" or "wide" as possible
- Avoid overconfident posterior estimates

### The Combined Effect

A trade-off between these two objectives:

1. **Minimize reconstruction error**: explain $y$ as well as possible using $z, x$
2. **Minimize coding cost**: encode $z, x$ with as few bits as possible

This is analogous to the **Minimum Description Length (MDL)** principle from **information theory**:

$$\text{total cost} = \text{data cost} + \text{model cost}$$

---

## 10. A Concrete Example: Variational Inference for a Gaussian Mixture

Let's use a simple example to see CAVI in action.

### A Simple Model

**Model**:
$$z \in \{1, 2\}, \quad y | z \sim \mathcal{N}(\mu_z, 1)$$

**Parameters**: $\pi_1 = \pi_2 = 0.5$, $\mu_1 = -1$, $\mu_2 = 1$

**Observation**: $y = 0.5$

### The True Posterior

Using Bayes' rule:

$$p(z=1|y) \propto \pi_1 \mathcal{N}(0.5|-1,1) = 0.5 \times 0.242$$

$$p(z=2|y) \propto \pi_2 \mathcal{N}(0.5|1,1) = 0.5 \times 0.352$$

Normalizing:
$$p(z=1|y) \approx 0.408, \quad p(z=2|y) \approx 0.592$$

(the second class is more likely, because the observation is closer to $\mu_2=1$)

### The Mean-Field Variational Approximation

The variational family:
$$q(z) = \text{Bernoulli}(\phi) = \{\phi: p(z=1)=\phi, p(z=2)=1-\phi\}$$

### The CAVI Update

$$q^*(z) \propto \exp\{\mathbb{E}[\log p(z, y)]\}$$

$$\propto \exp\{\log\pi_z + \log\mathcal{N}(y|\mu_z, 1)\}$$

This simplifies to:
$$q^*(z=1) \propto \exp\{\log 0.5 + \log\mathcal{N}(0.5|-1,1)\}$$

$$\propto 0.5 \times 0.242 = 0.121$$

Similarly:
$$q^*(z=2) \propto 0.5 \times 0.352 = 0.176$$

Normalizing:
$$q^*(z=1) \approx 0.407, \quad q^*(z=2) \approx 0.593$$

### The Result

**The $q(z)$ from CAVI is very close to the true posterior!**

(In this simple example, it is almost exact.)

---

## 11. Why Mean-Field Can Be Imperfect

### A Contrasting Example

Suppose the true posterior looks like this:

**In the true joint posterior**, $z$ and $x$ are highly correlated:
- If $z=1$, then $x$ is more likely $\approx 0$
- If $z=2$, then $x$ is more likely $\approx 5$

### Visual Comparison

```
True posterior:          Mean-field approximation:
   x                        x
   │                        │
 5 │    ● cluster 2       5 │    ╱╲ averaged distribution
   │                        │   ╱  ╲
 2.5│╱╲                    2.5├─────┤ wide distribution
   │╱  ╲                     │╱    ╲
 0 │╱    ● cluster 1       0 │────────
   │     ╱                   │
   └──────── z               └──────── z
    1   2                     1   2

True: two separated clusters   Approx: "averages" them, wider
```

### The Cost of Mean-Field

1. **$q(x)$ is wider than the true posterior**
   - It "averages" between the two modes
   - Less certain about any single explanation

2. **The sufficient statistics may be less accurate**
   - $\mathbb{E}_q[x]$ may not represent any true mode
   - But usually still good enough to make the parameter updates work

3. **But inference remains tractable**
   - We obtain a computable approximation
   - The parameters still improve
   - It usually works well in practice

---

## 12. The General Form of the CAVI Algorithm

### For an Arbitrary Set of Variables

Suppose we want to infer $z_1, \ldots, z_m$ with the mean-field assumption:
$$q(z_1, \ldots, z_m) = \prod_{i=1}^m q_i(z_i)$$

### General CAVI

```
for i = 1 to m:
  for iteration = 1 to max_iter:

    for each factor i:
      q_i(z_i) ∝ exp{E_{q(others)}[log p(all)]}

    compute the ELBO
    if the ELBO has converged, break
```

### Implementation Notes

1. **Sequential updates**: one factor at a time

2. **Computing expectations**: requires expectations over the other factors
   - For Gaussians: easy
   - For discrete distributions: a sum
   - For complex models: may need numerical methods

3. **Convergence criterion**: monitor the ELBO or the parameter changes

---

## 13. Summary: The Big Picture of Variational Inference

### Core Idea

```
Problem:
  the posterior p(z,x|y) cannot be computed (too complex)

Solution:
  find a simple distribution q to approximate it

Mathematical framework:
  minimize the KL divergence ↔ maximize the ELBO

Practical algorithm:
  CAVI (coordinate ascent)
    ↓
  update each factor in turn
    ↓
  in SLDS: update q(z) (HMM) + update q(x) (LDS)
    ↓
  two tractable inference problems!

Trade-off:
  Exactness    ────→  ✗ (mean-field error)
  Tractability ────→  ✓ (feasible in reasonable time)
  Practicality ────→  ✓ (works well in practice)
```

---

## 14. Recap of Key Points

| Concept | Definition | Intuition |
|------|------|--------|
| **KL divergence** | $D_{KL}(q \mid\mid p)$ | dissimilarity between two distributions; asymmetric |
| **ELBO** | $E_q[\log p(y,z,x)] - E_q[\log q(z,x)]$ | lower bound on the log likelihood; maximizing the ELBO minimizes the KL |
| **Mean-Field** | $q(z,x) = q(z)q(x)$ | independent-factor assumption; makes inference tractable but ignores correlations |
| **CAVI** | Coordinate Ascent Variational Inference | optimize each factor in turn; guarantees monotone convergence |
| **In SLDS** | HMM inference + LDS inference | decompose a hard problem into two solvable sub-problems |

---

## 15. The Complete Story from Lecture 12 to Lecture 14

### Learning Path

```
Lecture 12: EM foundations
  ├─ soft assignment (GMM)
  ├─ temporal structure (HMM)
  └─ forward-backward algorithm

       ↓ upgrade to continuous latents

Lecture 13: LDS and SLDS
  ├─ Kalman filter/smoother (exact, LDS)
  ├─ switching modes (SLDS)
  ├─ problem: the posterior has K^T components
  └─ need a new method...

       ↓ introduce variational inference

Lecture 14: variational inference
  ├─ KL divergence and the ELBO
  ├─ mean-field assumption (decouple z and x)
  ├─ CAVI algorithm
  └─ application: variational EM for SLDS
     ├─ update q(z): an HMM problem
     └─ update q(x): an LDS problem

Result: a complete, tractable algorithm!
```

---

## 16. Why This Matters: Applications to Neural Data

### Neural Data in the Real World

When analyzing neural activity, we often encounter:

1. **Complex dynamics**
   - Neural circuits are not a single linear system
   - Different behaviors (walking, sniffing, feeding) have different dynamics
   - We need SLDS to model them

2. **Incomplete observations**
   - We can only record a few hundred neurons
   - But there may be thousands
   - The latent variables represent the full neural state

3. **The inference problem**
   - Recover the full neural activity from the observations
   - Understand what the system is doing

### The Role of Variational EM

**Variational EM makes all of the above possible**:

1. Use SLDS to model complex neural dynamics
2. Use variational inference to handle the intractable inference problem
3. Use EM to learn parameters from data

### The Practical Payoff

```
Observed data            Variational EM
(some neurons)  ────────→  parameter and latent-state estimates
                          ↓
                    understand the neural circuit
                    detect behavioral modes
                    predict future activity
```

---

**You now fully understand the complete path from EM to variational inference!** 🎓

This framework is the foundation of modern neural data analysis.
