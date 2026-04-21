# Predictive Maintenance with Reinforcement Learning

Compares **Double DQN** and **Dyna-Q** for turbofan predictive maintenance on the NASA CMAPSS FD001 dataset. Includes Prioritised Experience Replay, TensorBoard logging, post-training evaluation metrics, and an interactive Streamlit dashboard.

## Project structure

```
pdm/
  data.py            – CMAPSS preprocessing, sliding-window states, train/val split
  env.py             – TurboFanEnv (gymnasium.Env)
  agents.py          – DoubleDQNAgent, DynaQAgent, SumTree, PrioritizedReplayBuffer
train_compare.py     – Training pipeline (tqdm progress, TensorBoard, checkpoint save)
evaluate_metrics.py  – Post-training TCO / FDR / P(Maintenance|RUL) evaluation
plot_results.py      – Standalone cost-comparison plot
app.py               – Streamlit live demo dashboard
METHODOLOGY_REPORT.md – Academic methodology and results write-up
```

## State representation

A 30-cycle trailing mean is applied to each of the 21 sensor channels, and the raw cycle count is appended, yielding a **22-dimensional continuous state vector**. All sensor channels are Min-Max scaled on the training split only (no data leakage).

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

Features: live sensor feed, Q-value bar chart, RUL gauge, cumulative cost chart, play/pause/reset controls, policy toggle (Double DQN / Fixed-Interval / Random).

### 4 — Standalone cost plot

```bash
python plot_results.py --metrics-file artifacts/cost_comparison.csv
```

## Key design choices

| Choice | Detail |
|---|---|
| **Prioritised Experience Replay** | SumTree (O(log n)), α = 0.6, β annealed 0.4 → 1.0 over 100 k steps; failure-boost ×5 for RUL < 20 |
| **Double DQN** | Decoupled action selection (online) / evaluation (target); target sync every 100 steps |
| **Dyna-Q world model** | Predicts state *residuals* Δs (not absolute s′) to avoid cycle-vs-sensor scale mismatch; N = 50 planning steps per real step |
| **Gradient clipping** | ‖∇‖₂ ≤ 10.0 on both Q-network and world model |
| **LR schedule** | Linear decay from 3 × 10⁻⁴ to 5 % of initial over training |
| **Hardware** | `torch.backends.cudnn.benchmark = True`; batch size 256 targets RTX 3060 Ti |

## Reward function

| Event | Reward |
|---|---|
| Safe operation (Continue, RUL > 0) | +1 |
| Scheduled maintenance (Maintain) | −20 |
| Catastrophic failure (Continue, RUL = 0) | −100 |
