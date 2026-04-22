# Predictive Maintenance with Reinforcement Learning

Compares **Double DQN** and **Dyna-Q** for turbofan predictive maintenance on the NASA CMAPSS FD001 dataset. Includes Prioritised Experience Replay, TensorBoard logging, post-training evaluation metrics, and an interactive Streamlit dashboard.

## Project structure

```
pdm/
  data.py              – CMAPSS preprocessing, dead-sensor removal, sliding-window states, train/val split
  env.py               – TurboFanEnv (gymnasium.Env) with action-masking guard
  ddqn_agent.py        – DoubleDQNAgent, SumTree, PrioritizedReplayBuffer, MLP, Transition
  dyna_agent.py        – DynaQAgent (extends DoubleDQNAgent + joint world model)
  agents.py            – backward-compatibility re-export shim
train_compare.py       – Training pipeline (tqdm progress, TensorBoard, checkpoint save)
evaluate_metrics.py    – Post-training TCO / FDR / P(Maintenance|RUL) evaluation
sensitivity_analysis.py – Economic cost-ratio sensitivity sweep + domain comparison plots
plot_results.py        – Standalone cost-comparison plot
app.py                 – Streamlit live demo dashboard with entropy/certainty metric
METHODOLOGY_REPORT.md – Academic methodology and results write-up
```

## State representation

A 30-cycle trailing mean is applied to each of the **15 active sensor channels** (6 zero-variance dead sensors — sensor\_1, 5, 10, 16, 18, 19 — are removed at preprocessing time), and the raw cycle count is appended, yielding a **16-dimensional continuous state vector**. All active sensor channels are Min-Max scaled on the training split only (no data leakage).

## Expected data layout

Place the CMAPSS text files under a local `data/` directory:

```text
data/
  train_FD001.txt
  test_FD001.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Workflow

### 1 — Train agents

```bash
python train_compare.py \
  --data-dir data \
  --subset FD001 \
  --episodes 150 \
  --fixed-interval 150 \
  --seed 42 \
  --output-dir artifacts \
  --tensorboard-dir runs
```

**Outputs written to `artifacts/`:**

| File | Description |
|---|---|
| `training_history.csv` | Per-episode reward, loss, Q-value, epsilon |
| `cost_comparison.csv` | Per-engine cost for every policy |
| `cost_summary.csv` | Aggregate cost statistics |
| `training_curves.png` | Reward and loss learning curves |
| `total_cost_comparison.png` | Bar chart comparing policy costs |
| `checkpoint_DoubleDQN.pt` | Trained Double DQN weights |
| `checkpoint_DynaQ.pt` | Trained Dyna-Q weights + world model |

Monitor training live:

```bash
tensorboard --logdir runs
```

### 2 — Evaluate metrics

```bash
python evaluate_metrics.py \
  --data-dir data \
  --artifacts-dir artifacts \
  --output-dir artifacts \
  --fdr-threshold 50 \
  --fixed-interval 150
```

### 2b — Sensitivity analysis (cost-ratio sweep)

Shows how policy rankings change as the failure penalty varies from 50 to 1,000 (cost ratio 2.5:1 → 50:1), proving the agent learned the underlying MDP rather than memorising a fixed threshold.

```bash
python sensitivity_analysis.py \
  --data-dir data \
  --artifacts-dir artifacts \
  --output-dir artifacts
```

**Outputs written to `artifacts/`:**

| File | Description |
|---|---|
| `sensitivity_cost_curves.png` | Avg cost/engine vs. failure penalty — two panels (absolute + ratio scale) |
| `sensitivity_rul_at_maint.png` | Violin: RUL distribution at moment of maintenance trigger per policy |
| `sensitivity_summary.csv` | Full matrix: every (policy × failure cost) scenario |

**Additional outputs written to `artifacts/`:**

| File | Description |
|---|---|
| `eval_rollout.csv` | Step-level rollout records for all policies |
| `tco_per_engine.csv` | Per-engine Total Cost of Ownership breakdown |
| `tco_summary.csv` | Aggregate TCO with cost-per-1k-cycles |
| `maintenance_probability.csv` | P(Maintenance \| RUL) binned in 10-cycle intervals |
| `fdr_report.csv` | FDR confusion table (TP / FP / TN / FN, precision, recall) |
| `tco_comparison.png` | Stacked cost bar chart + engine longevity panel |
| `maintenance_probability.png` | P(Maintenance \| RUL) curves with failure-risk shading |

### 3 — Launch the Streamlit dashboard

Requires `checkpoint_DoubleDQN.pt` to exist in `artifacts/`.

```bash
streamlit run app.py
```

Features: live sensor feed, Q-value bar chart with softmax probabilities and Shannon entropy display, agent certainty progress bar, RUL gauge, cumulative cost chart, play/pause/reset controls, policy toggle (Double DQN / Dyna-Q / Fixed-Interval / Random).

### 4 — Standalone cost plot

```bash
python plot_results.py --metrics-file artifacts/cost_comparison.csv
```

## Key design choices

| Choice | Detail |
|---|---|
| **Dead-sensor removal** | 6 zero-variance channels (sensor\_1, 5, 10, 16, 18, 19) dropped before training; state dim 22 → 16 |
| **Proximity penalty** | Linear ramp from 0 at RUL=20 to −2.0 at RUL=1 added to safe-cycle reward; eliminates −100 reward cliff |
| **Prioritised Experience Replay** | SumTree (O(log n)), α = 0.6, β annealed 0.4 → 1.0 over 200 k steps; failure-boost ×5 for RUL < 30 |
| **Double DQN** | Decoupled action selection (online) / evaluation (target); Polyak soft-update τ = 0.005 every gradient step |
| **Dyna-Q world model** | Predicts (Δs, reward, done) jointly; three-term loss MSE(Δs) + MSE(r) + BCE(done); N = 100 planning steps |
| **Action masking** | `TurboFanEnv.step()` raises `RuntimeError` if called after episode termination |
| **Gradient clipping** | ‖∇‖₂ ≤ 5.0 on both Q-network and world model |
| **LR schedule** | Linear decay from 3 × 10⁻⁴ to 5 % of initial over training |
| **Hardware** | `torch.backends.cudnn.benchmark = True`; batch size 256 targets RTX 3060 Ti |

## Reward function

| Event | Reward |
|---|---|
| Safe operation (Continue, RUL > 0) | +1 |
| Safe operation, RUL ∈ (0, 20) | +1 minus linear proximity penalty (up to −2.0 at RUL=1) |
| Scheduled maintenance (Maintain) | −20 |
| Catastrophic failure (Continue, RUL = 0) | −100 |
