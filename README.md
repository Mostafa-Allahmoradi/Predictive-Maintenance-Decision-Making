# Predictive Maintenance with Reinforcement Learning

This project compares Double DQN and Dyna-Q for turbofan predictive maintenance on the NASA CMAPSS FD001 dataset.

## Assumption used to reconcile the state definition

The project uses a 30-cycle sliding window, but compresses each window into a 21-sensor trailing mean and appends the current cycle count. That keeps the environment observation aligned with a 22-dimensional continuous state while still injecting temporal context.

## Expected data layout

Place the CMAPSS text files under a local `data/` directory:

```text
data/
  train_FD001.txt
  test_FD001.txt
```

## Run

```bash
pip install -r requirements.txt
python train_compare.py --data-dir data --subset FD001 --episodes 150 --fixed-interval 150
python plot_results.py --metrics-file artifacts/cost_comparison.csv
```

## Outputs

Running `train_compare.py` writes:

- `artifacts/training_history.csv`
- `artifacts/cost_comparison.csv`
- `artifacts/cost_summary.csv`
- `artifacts/training_curves.png`
- `artifacts/total_cost_comparison.png`
