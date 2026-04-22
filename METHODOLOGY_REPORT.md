# Methodology and Results Analysis
## Comparative Predictive Maintenance Using Deep Reinforcement Learning on NASA CMAPSS FD001

**Course:** Reinforcement Learning Programming — Final Project  
**Dataset:** NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation), Subset FD001  
**Algorithms:** Double Deep Q-Network (Double DQN) · Dyna-Q (Model-Based RL)  
**Baseline:** Fixed-Interval Policy (every 150 cycles) · Random Policy

---

## 1. Problem Formulation

Predictive Maintenance (PdM) is framed as a **sequential decision-making problem** under uncertainty. At each operational cycle, a maintenance agent must decide whether to continue operating an aircraft turbofan engine or to perform preventive maintenance. The objective is to minimise the **Total Cost of Ownership (TCO)** — the aggregate of maintenance expenditure and catastrophic failure penalties — while maximising engine utilisation.

This problem is formulated as a **Markov Decision Process** (MDP) $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:

| Component | Definition |
|---|---|
| $\mathcal{S}$ | Continuous state space $\mathbb{R}^{16}$ (15 normalised active sensors + cycle count) |
| $\mathcal{A}$ | Discrete action space $\{0 : \text{Continue},\ 1 : \text{Maintain}\}$ |
| $\mathcal{P}$ | Unknown transition dynamics (learned by the Dyna-Q world model) |
| $\mathcal{R}$ | Economic cost-benefit reward with proximity shaping (defined below) |
| $\gamma$ | Discount factor $= 0.99$ |

The reward function encodes the economic trade-off explicitly:

$$R(s_t, a_t) =
\begin{cases}
+1 & \text{if } a_t = 0 \text{ and } \mathrm{RUL}_t > 0 \hspace{1cm} \text{(safe operation)} \\
+1 - \rho(\mathrm{RUL}_t) & \text{if } a_t = 0 \text{ and } 0 < \mathrm{RUL}_t < \tau \hspace{0.4cm} \text{(proximity shaping)} \\
-20 & \text{if } a_t = 1 \hspace{4.2cm} \text{(scheduled maintenance)} \\
-100 & \text{if } a_t = 0 \text{ and } \mathrm{RUL}_t = 0 \hspace{1.3cm} \text{(catastrophic failure)}
\end{cases}$$

where the **proximity shaping penalty** $\rho$ ramps linearly from 0 at $\mathrm{RUL} = \tau = 20$ to $-2.0$ at $\mathrm{RUL} = 1$:

$$\rho(\mathrm{RUL}_t) = \lambda \cdot \frac{\tau - \mathrm{RUL}_t}{\tau}, \qquad \lambda = 2.0,\ \tau = 20$$

This ramp eliminates the abrupt reward cliff at $\mathrm{RUL} = 0$ by providing a smooth gradient signal in the near-failure zone, guiding the agent toward timely maintenance without introducing a discontinuous jump in policy incentives.

---

## 2. Data and Preprocessing

### 2.1 NASA CMAPSS FD001

The CMAPSS FD001 dataset contains run-to-failure time-series for 100 turbofan engines, each operating under a single flight condition (sea-level; moderate fan degradation). Each engine record contains 21 sensor channels (temperatures, pressures, fan and core speeds, efficiencies) and 3 operational setting channels sampled at every engine cycle until failure.

### 2.2 Sliding-Window State Representation

Raw sensor readings exhibit cycle-to-cycle noise that is uninformative for long-horizon planning. A **30-cycle trailing mean** is applied over each active sensor channel to produce a smoothed representation of the current operating condition. For engine $u$ at cycle $t$, the windowed sensor vector is:

$$\bar{x}^{(u)}_t = \frac{1}{W} \sum_{k=0}^{W-1} x^{(u)}_{t-k}, \qquad W = 30$$

The full state vector appended with the current cycle count forms the **16-dimensional** observation:

$$s_t = \left[\bar{x}^{(u)}_{t,1},\ \bar{x}^{(u)}_{t,2},\ \ldots,\ \bar{x}^{(u)}_{t,15},\ t\right]^{\top} \in \mathbb{R}^{16}$$

### 2.3 Feature Scaling

All 15 active sensor channels are normalised via **Min-Max Scaling** fitted exclusively on the training split, preventing data leakage into evaluation:

$$\tilde{x}_{i} = \frac{x_{i} - x_{i}^{\min}}{x_{i}^{\max} - x_{i}^{\min}}, \qquad \tilde{x}_i \in [0,\ 1]$$

Six sensors exhibiting zero variance in the FD001 subset (sensor\_1, 5, 10, 16, 18, 19) are removed at preprocessing time, reducing the active sensor count from 21 to 15 and the state dimension from 22 to 16. Retaining these dead channels would bias the Min-Max scaler and add uninformative dimensions to the Q-network input.

### 2.4 Remaining Useful Life Labels

Training labels are computed as the cycle count remaining until failure:

$$\mathrm{RUL}^{(u)}_t = T^{(u)}_{\max} - t$$

where $T^{(u)}_{\max}$ is the terminal cycle of engine $u$. The RUL is used exclusively for environment logic and the PER failure-boost condition; it is **not** provided to the agent as a state feature, preserving the challenge of estimation.

---

## 3. Environment Architecture: `TurboFanEnv`

The custom `TurboFanEnv` class, implementing the `gymnasium.Env` interface, serves as the simulation backbone against which all agents are trained. Each episode corresponds to a single engine run, beginning at the first post-warmup cycle and terminating upon one of three events:

1. Agent selects $a_t = 1$ (scheduled maintenance; penalty $= -20$)
2. $\mathrm{RUL}_t = 0$ with $a_t = 0$ (catastrophic failure; penalty $= -100$)
3. All recorded cycles are exhausted without intervention (natural completion)

This design allows the agent to discover the optimal **maintenance decision boundary** — cycles at which the expected future reward from continuing is exceeded by the guaranteed cost of early maintenance.

An **action masking guard** is enforced at the Gym API level: `step()` raises a `RuntimeError` if called after an episode has already terminated, preventing silent misuse during training loop bugs.

---

## 4. Algorithm 1: Double Deep Q-Network (Double DQN)

### 4.1 Bellman Optimality and the Q-Learning Objective

The optimal action-value function $Q^*$ satisfies the **Bellman Optimality Equation**:

$$Q^*(s, a) = \mathbb{E}\!\left[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \;\middle|\; S_t = s,\ A_t = a\right]$$

Deep Q-Networks (Mnih et al., 2015) parametrise $Q^*$ with a neural network $Q_\theta$, minimising the temporal-difference (TD) error:

$$\mathcal{L}^{\mathrm{DQN}}(\theta) = \mathbb{E}_{\mathcal{D}}\!\left[\left(Y_t^{\mathrm{DQN}} - Q(S_t, A_t; \theta)\right)^2\right]$$

$$Y_t^{\mathrm{DQN}} = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a';\; \theta^{-})$$

where $\theta^{-}$ denotes the periodically frozen **target network** weights.

### 4.2 Maximisation Bias and the Double DQN Correction

Standard DQN uses the same network to both *select* and *evaluate* the greedy next action, leading to **maximisation bias** (van Hasselt et al., 2016): upward-biased Q-value estimates that impair policy quality. Double DQN decouples these two operations:

$$\boxed{Y_t^{\mathrm{DDQN}} = R_{t+1} + \gamma\; Q\!\left(S_{t+1},\; \underbrace{\arg\max_{a'}\ Q(S_{t+1}, a';\;\theta_{\mathrm{online}})}_{\text{action selection: online network}}\;;\; \underbrace{\theta'_{\mathrm{target}}}_{\text{action evaluation: target network}}\right)}$$

**Action selection** ($\arg\max$) is performed by the online network $\theta$, whose parameters are updated every gradient step. **Action evaluation** (the Q-value assigned to that action) is performed by the lagged target network $\theta'$, updated every 100 gradient steps via a hard copy:

$$\theta' \leftarrow \theta \quad \text{every } K = 100 \text{ gradient steps}$$

This separation breaks the positive feedback loop between selection and evaluation, producing substantially more stable value estimates — a critical property when large negative rewards ($-100$) are sparse events.

### 4.3 Importance-Sampled Loss with PER Weights

To account for non-uniform sampling under Prioritised Experience Replay, the loss is weighted per-sample:

$$\mathcal{L}^{\mathrm{IS}}(\theta) = \frac{1}{B} \sum_{i=1}^{B} w_i \cdot \left(Y_i^{\mathrm{DDQN}} - Q(s_i, a_i;\;\theta)\right)^2$$

where $w_i = \left(N \cdot P(i)\right)^{-\beta}$ corrects the gradient for the sampling bias introduced by priority-based selection.

### 4.4 Neural Network Architecture

Both the online and target Q-networks share the following two-hidden-layer MLP architecture:

$$Q_\theta: \mathbb{R}^{16} \to \mathbb{R}^{2}$$

$$s_t \xrightarrow{\mathrm{Linear}(16,256)} \xrightarrow{\mathrm{ReLU}} \xrightarrow{\mathrm{Linear}(256,256)} \xrightarrow{\mathrm{ReLU}} \xrightarrow{\mathrm{Linear}(256,2)} \left[Q_\theta(s_t, 0),\ Q_\theta(s_t, 1)\right]$$

### 4.5 Hyperparameter Selection and Stability Measures

| Hyperparameter | Value | Justification |
|---|---|---|
| Discount factor $\gamma$ | 0.99 | Long engine lifecycles require far-sighted credit assignment |
| Learning rate $\alpha_0$ | $3 \times 10^{-4}$ | Canonical Adam rate for DQN; stabilises with $-100$ reward scale |
| LR schedule | Linear decay to $5\%$ over training | Prevents oscillation in the convergent phase |
| Batch size | 256 | Saturates GPU (RTX 3060 Ti) for 16-dimensional states |
| $\varepsilon$-decay | $0.99995$ per step, floor $0.02$ | Reaches floor at $\approx$78k steps ($\approx$500 ep × 200 avg cycles) |
| Gradient clipping | $\|\nabla\|_2 \leq 5.0$ | Tighter clipping for 256-unit hidden layers |
| Soft-update coefficient $\tau$ | $0.005$ | Polyak averaging every gradient step: $\theta' \leftarrow \tau\theta + (1-\tau)\theta'$ |

The Polyak (soft) target update replaces the periodic hard copy used in standard DQN. Copying the online network every $K$ steps introduces a periodic discontinuity in the loss curve as the target Q-values jump. Polyak averaging with $\tau = 0.005$ propagates online changes incrementally at every gradient step, producing smoother bootstrapped targets and improved loss stability in the presence of the large $-100$ failure penalty.

---

## 5. Algorithm 2: Dyna-Q (Model-Based RL)

### 5.1 Motivation: Sample Efficiency in Safety-Critical Domains

Aviation maintenance systems cannot be trained against real engines in production. Model-free methods such as Double DQN require a large volume of real interactions to converge. **Dyna-Q** (Sutton, 1990) addresses this by integrating a *world model* that generates synthetic experience, enabling the agent to effectively increase its sample count without additional real-world risk.

### 5.2 The Dyna Architecture

Dyna-Q performs three interleaved operations on every real timestep:

```
for each real interaction (s_t, a_t, r_t, s_{t+1}):
  ① Direct RL       → store (s,a,r,s') in buffer; update Q-network  (1 gradient step)
  ② Model learning  → update world model f_φ on the real transition   (1 gradient step)
  ③ Planning        → repeat N=100 times:
                         sample (s̃) from buffer
                         sample ã ~ Uniform{0,1}
                         predict ĉs̃' = s̃ + f_φ(s̃, ã)         (residual prediction)
                         compute simulated r̃, d̃ from world model
                         update Q-network on (s̃, ã, r̃, s̃', d̃)  (1 gradient step)
```

### 5.3 Direct Reinforcement Learning vs. Planning

The fundamental distinction is the **source of experience**:

| Property | Direct RL | Dyna-Q Planning |
|---|---|---|
| **Experience source** | Real environment transitions | Simulated transitions from world model |
| **Fidelity** | Perfect (ground truth) | Approximate (bounded by model accuracy) |
| **Cost per sample** | High (real engine cycle) | Near-zero (forward pass through $f_\phi$) |
| **Gradient steps per real step** | 1 | $1 + N = 101$ |
| **Risk of error propagation** | None | Compounded model inaccuracies at long horizons |

With $N = 100$ planning steps, the Dyna-Q agent effectively performs **102 Q-network updates for every single real engine cycle**, dramatically accelerating value function approximation during the early stages of training where real failure events are scarce.

### 5.4 World Model: Joint Prediction of Transition, Reward, and Termination

The world model $f_\phi: \mathbb{R}^{16+2} \to \mathbb{R}^{18}$ is a two-hidden-layer MLP conditioned on the current state and a one-hot encoded action. Its output is decomposed into three semantically distinct components:

$$f_\phi\!\left(s_t,\, \mathbf{1}_{a_t}\right) = \left[\hat{\Delta s}_t \in \mathbb{R}^{16},\ \hat{r}_t \in \mathbb{R},\ \hat{d}_t^{\text{logit}} \in \mathbb{R}\right]$$

$$\hat{s}_{t+1} = s_t + \hat{\Delta s}_t, \qquad \hat{d}_t = \sigma(\hat{d}_t^{\text{logit}}) > 0.5$$

The model is trained with a **three-term loss** that jointly optimises state residual accuracy, reward accuracy, and termination prediction:

$$\mathcal{L}_{\mathrm{world}} = \underbrace{\mathrm{MSE}(\hat{\Delta s},\, s' - s)}_{\text{state residual}} + \underbrace{\mathrm{MSE}(\hat{r},\, r)}_{\text{reward}} + \underbrace{\mathrm{BCE}(\hat{d}^{\text{logit}},\, d)}_{\text{termination}}$$

Predicting reward and termination from data — rather than using hardcoded heuristics — ensures the proximity-penalty ramp and the $-100$ failure cliff are automatically embedded in the planning loop. Simulated Q-updates thus use model-predicted rewards and done flags, faithfully propagating the full reward structure through every planning step.

The residual state prediction $\hat{\Delta s}$ remains critical: the 15 sensor channels produce residuals of order $10^{-2}$ while the cycle count increments by exactly $+1$ per step. An absolute-state MSE objective would be dominated by the cycle dimension by a factor of approximately $10^4$. Residual prediction equalises gradient contributions across all 16 state dimensions.

**Architecture:**

$$\mathbb{R}^{18} \xrightarrow{\mathrm{Linear}(18,256)} \xrightarrow{\mathrm{ReLU}} \xrightarrow{\mathrm{Linear}(256,256)} \xrightarrow{\mathrm{ReLU}} \xrightarrow{\mathrm{Linear}(256,18)} \left[\hat{\Delta s},\ \hat{r},\ \hat{d}^{\mathrm{logit}}\right]$$

---

## 6. Prioritised Experience Replay (PER)

Both agents employ Prioritised Experience Replay (Schaul et al., 2015) in place of a uniform replay buffer. In the CMAPSS dataset, safe-operation transitions account for the overwhelming majority of collected experience ($\approx 95\%$), while catastrophic failure and near-failure transitions (RUL $< 20$) are rare but highly informative for the policy.

PER addresses this imbalance by sampling transitions proportional to their **TD-error priority**:

$$p_i = |\delta_i| + \varepsilon, \qquad P(i) = \frac{p_i^{\,\alpha}}{\sum_k p_k^{\,\alpha}}$$

where $|\delta_i|$ is the absolute TD error of transition $i$, $\varepsilon = 10^{-6}$ ensures non-zero probability for all transitions, and $\alpha = 0.6$ controls the degree of prioritisation (0 = uniform; 1 = fully greedy).

An additional domain-specific **failure-boost multiplier** of $\times 5$ is applied to the initial priority of any transition satisfying RUL $< 30$ or reward $\leq -50$, ensuring the rare failure states receive disproportionate early exposure before TD-error-based priorities take over.

Importance-sampling (IS) weights $w_i = (N \cdot P(i))^{-\beta}$ correct for the resulting bias, with $\beta$ annealed from $0.4 \to 1.0$ over $2 \times 10^5$ steps.

---

## 7. Sensitivity Analysis: Economic Cost-Ratio Sweep

The RL agents are trained under a fixed cost assumption ($C_f = 100$, $C_m = 20$, ratio 5:1). A principled question is whether the agents have **learned the underlying MDP** or merely memorised a threshold calibrated to the training cost ratio.

To investigate this, `sensitivity_analysis.py` freezes each trained policy and sweeps the failure penalty over six scenarios:

| Scenario | $C_f$ | Ratio | Context |
|---|---|---|---|
| 2.5:1 | 50 | 2.5 | Commercial drone |
| **5:1** | **100** | **5** | **Industrial motor (training)** |
| 10:1 | 200 | 10 | Manufacturing line |
| 15:1 | 300 | 15 | Automotive powertrain |
| 25:1 | 500 | 25 | Commercial aviation |
| 50:1 | 1,000 | 50 | Safety-critical / medical |

For each scenario, the **per-engine cost** is recomputed from the observed rollout outcomes (number of failures and maintenance events) using the hypothetical $C_f$, while the agent's behaviour remains fixed. This tests whether the policy that is cost-optimal at 5:1 remains competitive as the ratio increases — a regime-change in the ranking indicates the policy is exploiting training-specific priors rather than a general risk-averse strategy.

A complementary **RUL-at-maintenance violin plot** reveals the agent's implicit decision threshold: a tight cluster at a low RUL indicates a risk-tolerant policy (calibrated for low failure cost), while a high or wide distribution indicates conservatism or heuristic behaviour respectively.

---

## 8. Baseline Policies

Two deterministic baselines are evaluated alongside the trained agents:

**Fixed-Interval Policy:** Triggers maintenance unconditionally once the engine reaches a pre-specified cycle threshold $T_{\mathrm{interval}} = 150$, irrespective of observed sensor state. This represents the traditional time-based maintenance paradigm and constitutes the primary industrial benchmark.

**Random Policy:** Selects $a \sim \mathrm{Uniform}\{0, 1\}$ independently at each step. This provides a lower bound representing the expected performance of an uninformed decision system.

---

## 9. Evaluation Metrics

### 9.1 Total Cost of Ownership (TCO)

$$\mathrm{TCO}_\pi = \sum_{u=1}^{N_{\mathrm{eval}}} \left( C_m \cdot \mathbb{1}[\text{engine } u \text{ maintained}] + C_f \cdot \mathbb{1}[\text{engine } u \text{ failed}] \right)$$

where $C_m = 20$ and $C_f = 100$ (training scenario defaults). The sensitivity analysis in Section 7 relaxes the fixed $C_f$ assumption. The normalised metric **cost per 1,000 operational cycles** enables fair comparison across policies with different maintenance frequencies.

### 9.2 False Discovery Rate (FDR)

The FDR quantifies the fraction of maintenance actions that were **premature** — executed on engines with substantial remaining life ($\mathrm{RUL} > 50$ cycles):

$$\mathrm{FDR} = \frac{FP}{TP + FP}, \qquad FP = \sum_t \mathbb{1}[a_t = 1 \land \mathrm{RUL}_t > 50]$$

This metric is of direct operational relevance: unnecessary early maintenance wastes scarce maintenance resources and reduces aircraft availability.

### 9.3 P(Maintenance | RUL)

The empirical probability of triggering maintenance, binned by the engine's RUL at the moment of decision, provides a qualitative view of each policy's **decision boundary**. An ideal policy should exhibit a monotonically increasing P(Maintenance | RUL) that rises sharply within the failure-risk zone (RUL $\leq 50$) and remains near zero at comfortable RUL levels.

### 9.4 Agent Uncertainty — Policy Entropy

The Streamlit dashboard computes **Shannon entropy** over the softmax probability distribution of the two Q-values at each decision step:

$$p_a = \frac{e^{Q(s,a) / T}}{\sum_{a'} e^{Q(s,a') / T}}, \qquad H = -\sum_a p_a \log p_a \in [0,\, \ln 2]$$

where temperature $T = 1$ (argmax equivalent for inference). Entropy $H = 0$ denotes a fully certain decision ($p = [1, 0]$ or $[0, 1]$); $H = \ln 2 \approx 0.693$ denotes maximum uncertainty ($p = [0.5, 0.5]$). Agent certainty is visualised as a progress bar: $\text{certainty} = 1 - H / \ln 2$.

---

## 10. Ethical Implications of RL in Aviation Maintenance

### 10.1 The Black-Box Problem and Accountability

Deep reinforcement learning agents, including the Double DQN and Dyna-Q systems presented in this work, are **inherently opaque**: their maintenance decisions emerge from non-linear transformations across hundreds of parameters, providing no human-interpretable justification. This opacity creates a fundamental accountability problem in aviation — one of the most heavily regulated safety domains globally.

Current regulatory frameworks administered by the Federal Aviation Administration (FAA) and the European Union Aviation Safety Agency (EASA) require that maintenance decisions be traceable to qualified human authority. FAA Advisory Circular AC 43.13-1B mandates that all maintenance actions on certified aircraft follow approved data, which implicitly presupposes human comprehension of the decision rationale. An autonomous RL agent that recommends deferring maintenance on a deteriorating engine — even if statistically optimal over a fleet — cannot currently satisfy this requirement unaided, because it cannot articulate *why* the sensor pattern observed at that moment is within acceptable limits.

The responsible deployment therefore requires **post-hoc explainability** mechanisms (e.g., SHAP attributions over sensor features, attention-weighted saliency maps) that allow a licensed aircraft maintenance engineer to inspect the features driving each recommendation. Without such tooling, the agent functions as an unsanctioned decision-maker rather than a certified decision-support system.

### 10.2 Human-in-the-Loop Requirements

The model presented in this project is best understood as a **decision-support tool** rather than an autonomous actuator. The operationally appropriate architecture positions the RL agent within a human-in-the-loop (HITL) system:

```
Sensor data → RL Agent (recommendation) → Human Engineer (review) → Maintenance decision
```

Under this paradigm, the agent reduces the cognitive load on the engineer by pre-screening thousands of sensor channels and surfacing engines at elevated risk, while preserving human authority over final action. This bifurcation is not merely procedural: it is a safety-critical design constraint. The $-100$ catastrophic failure penalty in the reward function reflects an economic cost, but the real-world consequence of an in-flight engine failure is measured in lives — a dimension the reward signal cannot capture.

HITL architecture also provides a mechanism for **value alignment**: feedback from maintenance engineers who disagree with agent recommendations can be collected as preference labels, enabling reward model refinement (Christiano et al., 2017) without retraining from scratch.

### 10.3 Distribution Shift and Operational Safety

The CMAPSS FD001 dataset represents a single flight-condition regime (sea-level; moderate degradation). Deployment against engines operating in Arctic conditions, high-altitude routes, or after non-standard repair events constitutes **covariate shift** — the test distribution differs from the training distribution in ways the agent cannot detect. Unlike a human engineer who would recognise an unusual sensor pattern and escalate, a DQN agent may produce confidently incorrect recommendations in novel operating regimes because Q-values are extrapolated from the training manifold without uncertainty quantification.

Mitigating this risk requires:
- **Epistemic uncertainty estimation** (e.g., Monte Carlo dropout, deep ensembles) to flag low-confidence states for human review
- **Continuous model monitoring** with statistical drift detection on incoming sensor streams
- **Conservative deployment policies** that default to preventive maintenance when the observed state falls outside a specified confidence region

### 10.4 Institutional Liability and Certification

No current regulatory framework provides a clear liability assignment when an AI system contributes to a maintenance decision that precedes an accident. The European Union's proposed AI Act (2024) classifies safety-critical infrastructure AI as high-risk, mandating conformity assessments, risk management systems, and human oversight provisions. Aviation maintenance systems trained on operational data additionally raise concerns under data governance regulations, as fleet sensor data may contain commercially sensitive information about aircraft utilisation patterns.

Certification pathways for ML-based maintenance tools are under active development within EASA's Artificial Intelligence Roadmap and the FAA's Runway to Safer Skies initiative, but no finalised standards exist as of the time of writing. Responsible deployment requires active engagement with regulatory authorities during design — not as a post-hoc compliance exercise.

### 10.5 Equity and Access

A subtler ethical dimension concerns the **distribution of benefits** from RL-based maintenance optimisation. Airlines with large modern fleets and sophisticated data infrastructure are best positioned to collect the sensor logs, labelling resources, and computational infrastructure required to train and validate such systems. Smaller regional carriers — which operate in many cases with older aircraft and thinner safety margins — may face precisely the conditions where predictive maintenance provides the greatest safety benefit, yet have the least capacity to adopt it. Industry standards bodies and civil aviation authorities bear a responsibility to ensure that safety-enhancing technology does not become an asymmetric advantage that widens the existing safety performance gap between large and small operators.

---

## 11. Summary

This project demonstrates that model-based reinforcement learning (Dyna-Q) and model-free reinforcement learning (Double DQN) can both learn substantially more cost-efficient maintenance policies than the Fixed-Interval baseline when evaluated on held-out CMAPSS FD001 engines. Key engineering contributions include: removal of 6 zero-variance dead sensors (state dim 22→16), a proximity-penalty reward ramp that eliminates the $-100$ cliff, a joint world model predicting $(\Delta s, r, d)$ with a three-term loss, Polyak soft-update of the target network ($\tau = 0.005$), and an economic sensitivity analysis demonstrating policy robustness across a 20× range of cost ratios.

The Dyna-Q agent achieves comparable or superior sample efficiency through its $N = 100$ planning steps per real interaction. The Double DQN agent benefits from the decoupled selection-evaluation mechanism and smooth Polyak target updates, producing stable value estimates in the presence of the large $-100$ failure penalty.

Both agents demonstrate reduced False Discovery Rates relative to the Fixed-Interval baseline, indicating a learned preference for later, more targeted maintenance interventions. The sensitivity analysis shows both RL agents maintain cost advantages over the Fixed-Interval baseline across all tested cost ratios, with the RUL-at-maintenance distribution confirming a consistent, calibrated decision threshold rather than a memorised rule.

The ethical analysis underscores that these results, while technically promising, are not sufficient conditions for deployment. Responsible real-world application requires interpretability tooling, uncertainty quantification, regulatory engagement, and mandatory human oversight at the point of action.

---

## References

- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with Double Q-learning. *AAAI*, 30(1).
- Sutton, R. S. (1990). Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. *ICML*, 216–224.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. *ICLR*.
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *ICIIS*, 1–9.
- Christiano, P. et al. (2017). Deep reinforcement learning from human preferences. *NeurIPS*, 30.
- EASA. (2023). *Artificial Intelligence Roadmap 2.0*. European Union Aviation Safety Agency.
- FAA. (2020). *Advisory Circular AC 43.13-1B: Acceptable Methods, Techniques, and Practices — Aircraft Inspection and Repair*. Federal Aviation Administration.
- European Parliament. (2024). *Regulation (EU) 2024/1689 — Artificial Intelligence Act*.
