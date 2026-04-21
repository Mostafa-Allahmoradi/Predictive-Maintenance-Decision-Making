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
| $\mathcal{S}$ | Continuous state space $\mathbb{R}^{22}$ (21 normalised sensors + cycle count) |
| $\mathcal{A}$ | Discrete action space $\{0 : \text{Continue},\ 1 : \text{Maintain}\}$ |
| $\mathcal{P}$ | Unknown transition dynamics (learned by the Dyna-Q world model) |
| $\mathcal{R}$ | Economic cost-benefit reward (defined below) |
| $\gamma$ | Discount factor $= 0.99$ |

The reward function encodes the economic trade-off explicitly:

$$R(s_t, a_t) =
\begin{cases}
+1 & \text{if } a_t = 0 \text{ and } \mathrm{RUL}_t > 0 \hspace{1cm} \text{(safe operation)} \\
-20 & \text{if } a_t = 1 \hspace{4.2cm} \text{(scheduled maintenance)} \\
-100 & \text{if } a_t = 0 \text{ and } \mathrm{RUL}_t = 0 \hspace{1.3cm} \text{(catastrophic failure)}
\end{cases}$$

The asymmetric penalty structure ($-100$ vs $-20$) deliberately encodes that unplanned failures are five times more costly than preventive action, which forces the agent to internalise safety risk during training.

---

## 2. Data and Preprocessing

### 2.1 NASA CMAPSS FD001

The CMAPSS FD001 dataset contains run-to-failure time-series for 100 turbofan engines, each operating under a single flight condition (sea-level; moderate fan degradation). Each engine record contains 21 sensor channels (temperatures, pressures, fan and core speeds, efficiencies) and 3 operational setting channels sampled at every engine cycle until failure.

### 2.2 Sliding-Window State Representation

Raw sensor readings exhibit cycle-to-cycle noise that is uninformative for long-horizon planning. A **30-cycle trailing mean** is applied over each sensor channel to produce a smoothed representation of the current operating condition. For engine $u$ at cycle $t$, the windowed sensor vector is:

$$\bar{x}^{(u)}_t = \frac{1}{W} \sum_{k=0}^{W-1} x^{(u)}_{t-k}, \qquad W = 30$$

The full state vector appended with the current cycle count forms the 22-dimensional observation:

$$s_t = \left[\bar{x}^{(u)}_{t,1},\ \bar{x}^{(u)}_{t,2},\ \ldots,\ \bar{x}^{(u)}_{t,21},\ t\right]^{\top} \in \mathbb{R}^{22}$$

### 2.3 Feature Scaling

All 21 sensor channels are normalised via **Min-Max Scaling** fitted exclusively on the training split, preventing data leakage into evaluation:

$$\tilde{x}_{i} = \frac{x_{i} - x_{i}^{\min}}{x_{i}^{\max} - x_{i}^{\min}}, \qquad \tilde{x}_i \in [0,\ 1]$$

The raw cycle count is intentionally left unscaled, as its absolute magnitude provides a meaningful temporal signal for the Q-network.

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

$$Q_\theta: \mathbb{R}^{22} \to \mathbb{R}^{2}$$

$$s_t \xrightarrow{\mathrm{Linear}(22,128)} \xrightarrow{\mathrm{ReLU}} \xrightarrow{\mathrm{Linear}(128,128)} \xrightarrow{\mathrm{ReLU}} \xrightarrow{\mathrm{Linear}(128,2)} \left[Q_\theta(s_t, 0),\ Q_\theta(s_t, 1)\right]$$

### 4.5 Hyperparameter Selection and Stability Measures

| Hyperparameter | Value | Justification |
|---|---|---|
| Discount factor $\gamma$ | 0.99 | Long engine lifecycles require far-sighted credit assignment |
| Learning rate $\alpha_0$ | $3 \times 10^{-4}$ | Canonical Adam rate for DQN; stabilises with $-100$ reward scale |
| LR schedule | Linear decay to $5\%$ over training | Prevents oscillation in the convergent phase |
| Batch size | 256 | Saturates GPU (RTX 3060 Ti) for 22-dimensional states |
| $\varepsilon$-decay | $0.995$ per step, floor $0.05$ | Smooth exploration → exploitation transition |
| Gradient clipping | $\|\nabla\|_2 \leq 10.0$ | Prevents gradient explosion from the $-100$ catastrophic penalty |
| Target sync interval $K$ | 100 steps | Balances target stability with convergence speed |

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
  ③ Planning        → repeat N=50 times:
                         sample (s̃) from buffer
                         sample ã ~ Uniform{0,1}
                         predict s̃' = s̃ + f_φ(s̃, ã)         (residual prediction)
                         compute simulated r̃, d̃ from s̃ and ã
                         update Q-network on (s̃, ã, r̃, s̃', d̃)  (1 gradient step)
```

### 5.3 Direct Reinforcement Learning vs. Planning

The fundamental distinction is the **source of experience**:

| Property | Direct RL | Dyna-Q Planning |
|---|---|---|
| **Experience source** | Real environment transitions | Simulated transitions from world model |
| **Fidelity** | Perfect (ground truth) | Approximate (bounded by model accuracy) |
| **Cost per sample** | High (real engine cycle) | Near-zero (forward pass through $f_\phi$) |
| **Gradient steps per real step** | 1 | $1 + N = 51$ |
| **Risk of error propagation** | None | Compounded model inaccuracies at long horizons |

With $N = 50$ planning steps, the Dyna-Q agent effectively performs **51 Q-network updates for every single real engine cycle**, dramatically accelerating value function approximation during the early stages of training where real failure events are scarce.

### 5.4 World Model: Residual State Prediction

The world model $f_\phi: \mathbb{R}^{22+2} \to \mathbb{R}^{22}$ is a two-hidden-layer MLP conditioned on the current state and a one-hot encoded action:

$$f_\phi\!\left(s_t,\, \mathbf{1}_{a_t}\right) = \hat{\Delta s}_t, \qquad \hat{s}_{t+1} = s_t + \hat{\Delta s}_t$$

**Critically**, the model predicts the state *residual* $\Delta s = s_{t+1} - s_t$ rather than the absolute next state. This design choice is non-trivial: the 21 sensor channels are scaled to $[0, 1]$, producing residuals of order $10^{-2}$, whereas the raw cycle count increments by exactly $+1$ per step. An absolute-state MSE objective would be dominated by the cycle dimension by a factor of approximately $10^4$, preventing the model from learning meaningful sensor dynamics. Residual prediction equalises gradient contributions across all 22 state dimensions:

$$\mathcal{L}_{\mathrm{world}} = \frac{1}{B}\sum_{i=1}^{B} \left\| f_\phi(s_i, \mathbf{1}_{a_i}) - (s'_i - s_i) \right\|^2$$

---

## 6. Prioritised Experience Replay (PER)

Both agents employ Prioritised Experience Replay (Schaul et al., 2015) in place of a uniform replay buffer. In the CMAPSS dataset, safe-operation transitions account for the overwhelming majority of collected experience ($\approx 95\%$), while catastrophic failure and near-failure transitions (RUL $< 20$) are rare but highly informative for the policy.

PER addresses this imbalance by sampling transitions proportional to their **TD-error priority**:

$$p_i = |\delta_i| + \varepsilon, \qquad P(i) = \frac{p_i^{\,\alpha}}{\sum_k p_k^{\,\alpha}}$$

where $|\delta_i|$ is the absolute TD error of transition $i$, $\varepsilon = 10^{-6}$ ensures non-zero probability for all transitions, and $\alpha = 0.6$ controls the degree of prioritisation (0 = uniform; 1 = fully greedy).

An additional domain-specific **failure-boost multiplier** of $\times 5$ is applied to the initial priority of any transition satisfying RUL $< 20$ or reward $\leq -50$, ensuring the rare failure states receive disproportionate early exposure before TD-error-based priorities take over.

Importance-sampling (IS) weights $w_i = (N \cdot P(i))^{-\beta}$ correct for the resulting bias, with $\beta$ annealed from $0.4 \to 1.0$ over $10^5$ steps.

---

## 7. Baseline Policies

Two deterministic baselines are evaluated alongside the trained agents:

**Fixed-Interval Policy:** Triggers maintenance unconditionally once the engine reaches a pre-specified cycle threshold $T_{\mathrm{interval}} = 150$, irrespective of observed sensor state. This represents the traditional time-based maintenance paradigm and constitutes the primary industrial benchmark.

**Random Policy:** Selects $a \sim \mathrm{Uniform}\{0, 1\}$ independently at each step. This provides a lower bound representing the expected performance of an uninformed decision system.

---

## 8. Evaluation Metrics

### 8.1 Total Cost of Ownership (TCO)

$$\mathrm{TCO}_\pi = \sum_{u=1}^{N_{\mathrm{eval}}} \left( C_m \cdot \mathbb{1}[\text{engine } u \text{ maintained}] + C_f \cdot \mathbb{1}[\text{engine } u \text{ failed}] \right)$$

where $C_m = 20$ and $C_f = 100$. The normalised metric **cost per 1,000 operational cycles** enables fair comparison across policies with different maintenance frequencies.

### 8.2 False Discovery Rate (FDR)

The FDR quantifies the fraction of maintenance actions that were **premature** — executed on engines with substantial remaining life ($\mathrm{RUL} > 50$ cycles):

$$\mathrm{FDR} = \frac{FP}{TP + FP}, \qquad FP = \sum_t \mathbb{1}[a_t = 1 \land \mathrm{RUL}_t > 50]$$

This metric is of direct operational relevance: unnecessary early maintenance wastes scarce maintenance resources and reduces aircraft availability.

### 8.3 P(Maintenance | RUL)

The empirical probability of triggering maintenance, binned by the engine's RUL at the moment of decision, provides a qualitative view of each policy's **decision boundary**. An ideal policy should exhibit a monotonically increasing P(Maintenance | RUL) that rises sharply within the failure-risk zone (RUL $\leq 50$) and remains near zero at comfortable RUL levels.

---

## 9. Ethical Implications of RL in Aviation Maintenance

### 9.1 The Black-Box Problem and Accountability

Deep reinforcement learning agents, including the Double DQN and Dyna-Q systems presented in this work, are **inherently opaque**: their maintenance decisions emerge from non-linear transformations across hundreds of parameters, providing no human-interpretable justification. This opacity creates a fundamental accountability problem in aviation — one of the most heavily regulated safety domains globally.

Current regulatory frameworks administered by the Federal Aviation Administration (FAA) and the European Union Aviation Safety Agency (EASA) require that maintenance decisions be traceable to qualified human authority. FAA Advisory Circular AC 43.13-1B mandates that all maintenance actions on certified aircraft follow approved data, which implicitly presupposes human comprehension of the decision rationale. An autonomous RL agent that recommends deferring maintenance on a deteriorating engine — even if statistically optimal over a fleet — cannot currently satisfy this requirement unaided, because it cannot articulate *why* the sensor pattern observed at that moment is within acceptable limits.

The responsible deployment therefore requires **post-hoc explainability** mechanisms (e.g., SHAP attributions over sensor features, attention-weighted saliency maps) that allow a licensed aircraft maintenance engineer to inspect the features driving each recommendation. Without such tooling, the agent functions as an unsanctioned decision-maker rather than a certified decision-support system.

### 9.2 Human-in-the-Loop Requirements

The model presented in this project is best understood as a **decision-support tool** rather than an autonomous actuator. The operationally appropriate architecture positions the RL agent within a human-in-the-loop (HITL) system:

```
Sensor data → RL Agent (recommendation) → Human Engineer (review) → Maintenance decision
```

Under this paradigm, the agent reduces the cognitive load on the engineer by pre-screening thousands of sensor channels and surfacing engines at elevated risk, while preserving human authority over final action. This bifurcation is not merely procedural: it is a safety-critical design constraint. The $-100$ catastrophic failure penalty in the reward function reflects an economic cost, but the real-world consequence of an in-flight engine failure is measured in lives — a dimension the reward signal cannot capture.

HITL architecture also provides a mechanism for **value alignment**: feedback from maintenance engineers who disagree with agent recommendations can be collected as preference labels, enabling reward model refinement (Christiano et al., 2017) without retraining from scratch.

### 9.3 Distribution Shift and Operational Safety

The CMAPSS FD001 dataset represents a single flight-condition regime (sea-level; moderate degradation). Deployment against engines operating in Arctic conditions, high-altitude routes, or after non-standard repair events constitutes **covariate shift** — the test distribution differs from the training distribution in ways the agent cannot detect. Unlike a human engineer who would recognise an unusual sensor pattern and escalate, a DQN agent may produce confidently incorrect recommendations in novel operating regimes because Q-values are extrapolated from the training manifold without uncertainty quantification.

Mitigating this risk requires:
- **Epistemic uncertainty estimation** (e.g., Monte Carlo dropout, deep ensembles) to flag low-confidence states for human review
- **Continuous model monitoring** with statistical drift detection on incoming sensor streams
- **Conservative deployment policies** that default to preventive maintenance when the observed state falls outside a specified confidence region

### 9.4 Institutional Liability and Certification

No current regulatory framework provides a clear liability assignment when an AI system contributes to a maintenance decision that precedes an accident. The European Union's proposed AI Act (2024) classifies safety-critical infrastructure AI as high-risk, mandating conformity assessments, risk management systems, and human oversight provisions. Aviation maintenance systems trained on operational data additionally raise concerns under data governance regulations, as fleet sensor data may contain commercially sensitive information about aircraft utilisation patterns.

Certification pathways for ML-based maintenance tools are under active development within EASA's Artificial Intelligence Roadmap and the FAA's Runway to Safer Skies initiative, but no finalised standards exist as of the time of writing. Responsible deployment requires active engagement with regulatory authorities during design — not as a post-hoc compliance exercise.

### 9.5 Equity and Access

A subtler ethical dimension concerns the **distribution of benefits** from RL-based maintenance optimisation. Airlines with large modern fleets and sophisticated data infrastructure are best positioned to collect the sensor logs, labelling resources, and computational infrastructure required to train and validate such systems. Smaller regional carriers — which operate in many cases with older aircraft and thinner safety margins — may face precisely the conditions where predictive maintenance provides the greatest safety benefit, yet have the least capacity to adopt it. Industry standards bodies and civil aviation authorities bear a responsibility to ensure that safety-enhancing technology does not become an asymmetric advantage that widens the existing safety performance gap between large and small operators.

---

## 10. Summary

This project demonstrates that model-based reinforcement learning (Dyna-Q) and model-free reinforcement learning (Double DQN) can both learn substantially more cost-efficient maintenance policies than the Fixed-Interval baseline when evaluated on held-out CMAPSS FD001 engines. The Dyna-Q agent achieves comparable or superior sample efficiency through its $N = 50$ planning steps per real interaction, at the cost of additional world-model training overhead. The Double DQN agent benefits from the decoupled selection-evaluation mechanism that eliminates maximisation bias, producing more conservative and stable Q-value estimates in the presence of the large $-100$ failure penalty.

Both agents demonstrate reduced False Discovery Rates relative to the Fixed-Interval baseline, indicating a learned preference for later, more targeted maintenance interventions rather than time-scheduled ones. The P(Maintenance | RUL) curves show a qualitatively appropriate monotonic increase in maintenance probability as RUL approaches zero, consistent with economically rational risk-averse behaviour.

The ethical analysis underscores that these results, while technically promising, are not sufficient conditions for deployment. Responsible real-world application requires interpretability tooling, uncertainty quantification, regulatory engagement, and mandatory human oversight at the point of action — transforming the RL agent from an autonomous actor into a rigorously governed decision-support system.

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
