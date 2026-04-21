"""Advanced evaluation metrics for the Predictive Maintenance RL project.

Metrics produced
----------------
1. Total Cost of Ownership (TCO)
   - Breakdown per policy: scheduled-maintenance cost + catastrophic-failure cost.
   - Cost-efficiency metric: cost per 1,000 cycles operated.

2. P(Maintenance | RUL) visualisation
   - For every policy, plots the empirical probability of choosing action=1
     binned by the engine's Remaining Useful Life at the moment of decision.
   - A vertical reference line marks the False-Discovery-Rate threshold.

3. False Discovery Rate (FDR)
   - Full signal-detection table: TP, FP (false discoveries), FN, TN.

     TP : agent maintains    AND  RUL ≤ threshold  (justified)
     FP : agent maintains    AND  RUL >  threshold  (premature  — the false discovery)
     TN : agent continues    AND  RUL >  threshold  (correct continuation)
     FN : agent continues    AND  episode ends in catastrophic failure  (missed danger)

     FDR = FP / (TP + FP)   — fraction of maintenance decisions that were premature.

Usage
-----
    python evaluate_metrics.py \\
        --data-dir data \\
        --artifacts-dir artifacts \\
        --output-dir artifacts \\
        --fixed-interval 150 \\
        --fdr-threshold 50

If no checkpoint is found in --artifacts-dir the RL slot is filled with an
untrained (random) agent and a clear warning is printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch

from pdm.agents import DoubleDQNAgent
from pdm.data import CMAPSSPreprocessor, split_episodes
from pdm.env import TurboFanEnv

# ──────────────────────────────────────────────────────────────────────────────
# Economic constants (must match TurboFanEnv defaults)
# ──────────────────────────────────────────────────────────────────────────────
MAINTENANCE_UNIT_COST: float = 20.0
FAILURE_UNIT_COST: float = 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Policy factories
# ──────────────────────────────────────────────────────────────────────────────

def random_policy(_state: np.ndarray) -> int:
    return int(np.random.randint(2))


def fixed_interval_policy(interval: int) -> Callable[[np.ndarray], int]:
    def _policy(state: np.ndarray) -> int:
        return 1 if int(round(float(state[-1]))) >= interval else 0
    return _policy


def greedy_rl_policy(agent: DoubleDQNAgent) -> Callable[[np.ndarray], int]:
    def _policy(state: np.ndarray) -> int:
        return agent.act(state, explore=False)
    return _policy


# ──────────────────────────────────────────────────────────────────────────────
# Per-step rollout
# ──────────────────────────────────────────────────────────────────────────────

def run_rollout(
    env: TurboFanEnv,
    policy_fn: Callable[[np.ndarray], int],
    label: str,
    seed: int = 0,
) -> pd.DataFrame:
    """Roll *policy_fn* over every episode in *env* and return per-step records.

    Each row captures the RUL and action **at the moment the decision was made**,
    which is the correct anchor for the P(maintenance|RUL) and FDR calculations.
    """
    rng = np.random.default_rng(seed)  # noqa: F841 — kept for reproducibility documentation
    records: list[dict] = []
    for engine_index, episode in enumerate(env.episodes):
        state, _ = env.reset(options={"engine_index": engine_index})
        terminated = False
        step = 0
        while not terminated:
            action = policy_fn(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            terminated = terminated or truncated
            # info["rul"] is the RUL at the decision point (before env advances index)
            records.append(
                {
                    "policy": label,
                    "engine_id": int(episode.engine_id),
                    "engine_index": engine_index,
                    "step": step,
                    "rul": int(info["rul"]),
                    "action": int(action),
                    "reward": float(reward),
                    "event": str(info["event"]),
                }
            )
            state = next_state
            step += 1
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Total Cost of Ownership
# ──────────────────────────────────────────────────────────────────────────────

def _per_engine_tco(rollout: pd.DataFrame) -> pd.DataFrame:
    """Build one row per (policy, engine) with cost breakdown."""
    rows: list[dict] = []
    for (policy, eng_id, eng_idx), grp in rollout.groupby(
        ["policy", "engine_id", "engine_index"], sort=False
    ):
        terminal_event = str(grp.iloc[-1]["event"])
        cycles = int((grp["event"] == "safe_operation").sum())
        m_cost = MAINTENANCE_UNIT_COST if terminal_event == "scheduled_maintenance" else 0.0
        f_cost = FAILURE_UNIT_COST if terminal_event == "catastrophic_failure" else 0.0
        rows.append(
            {
                "policy": policy,
                "engine_id": eng_id,
                "engine_index": eng_idx,
                "terminal_event": terminal_event,
                "cycles_operated": cycles,
                "maintenance_cost": m_cost,
                "failure_cost": f_cost,
                "total_engine_cost": m_cost + f_cost,
            }
        )
    return pd.DataFrame(rows)


def compute_tco_summary(per_engine: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-engine costs into a policy-level TCO summary."""

    def _failure_count(series: pd.Series) -> int:
        return int((series == "catastrophic_failure").sum())

    def _maintenance_count(series: pd.Series) -> int:
        return int((series == "scheduled_maintenance").sum())

    summary = (
        per_engine.groupby("policy", sort=False)
        .agg(
            n_engines=("engine_id", "count"),
            total_cycles=("cycles_operated", "sum"),
            avg_cycles_per_engine=("cycles_operated", "mean"),
            scheduled_maintenance_events=("terminal_event", _maintenance_count),
            catastrophic_failure_events=("terminal_event", _failure_count),
            total_maintenance_cost=("maintenance_cost", "sum"),
            total_failure_cost=("failure_cost", "sum"),
            total_raw_cost=("total_engine_cost", "sum"),
            avg_tco_per_engine=("total_engine_cost", "mean"),
        )
        .reset_index()
    )
    # Cost per 1,000 operational cycles — lower is more efficient
    summary["cost_per_1k_cycles"] = (
        summary["total_raw_cost"] / summary["total_cycles"].clip(lower=1) * 1_000
    ).round(2)
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# 2. P(Maintenance | RUL)  —  binned empirical probability
# ──────────────────────────────────────────────────────────────────────────────

_BIN_EDGES: list[int] = list(range(0, 201, 10)) + [10_000]
_BIN_LABELS: list[str] = [f"{lo}–{lo + 9}" for lo in range(0, 200, 10)] + ["200+"]


def _add_rul_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rul_bin"] = pd.cut(
        df["rul"].clip(lower=0),
        bins=_BIN_EDGES,
        labels=_BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    return df


def compute_maintenance_probability(rollout: pd.DataFrame) -> pd.DataFrame:
    """Return P(action=1 | RUL bin) for each policy."""
    rollout = _add_rul_bin(rollout)
    prob = (
        rollout.groupby(["policy", "rul_bin"], observed=True)["action"]
        .mean()
        .reset_index()
        .rename(columns={"action": "p_maintenance"})
    )
    return prob


# ──────────────────────────────────────────────────────────────────────────────
# 3. False Discovery Rate (signal-detection framing)
# ──────────────────────────────────────────────────────────────────────────────

def compute_fdr(rollout: pd.DataFrame, rul_threshold: int = 50) -> pd.DataFrame:
    """Produce a confusion-matrix style FDR report per policy.

    Decision threshold: RUL = *rul_threshold*.
    Positive label     : "engine is near failure"  ↔  RUL ≤ threshold.
    Predicted positive : agent chose action = 1 (maintain).

    ┌──────────────┬──────────────────────┬──────────────────────┐
    │              │  True: near-failure  │  True: safe (RUL>50) │
    ├──────────────┼──────────────────────┼──────────────────────┤
    │ Pred: maint  │  TP                  │  FP (false discovery)│
    │ Pred: cont.  │  FN (missed danger)  │  TN                  │
    └──────────────┴──────────────────────┴──────────────────────┘

    FN is approximated as the number of episodes ending in catastrophic failure
    (the agent never chose to maintain, so the engine was run to destruction).
    """
    records: list[dict] = []
    for policy, grp in rollout.groupby("policy", sort=False):
        maint = grp[grp["action"] == 1]
        cont = grp[grp["action"] == 0]

        tp = int((maint["rul"] <= rul_threshold).sum())
        fp = int((maint["rul"] > rul_threshold).sum())
        tn = int((cont["rul"] > rul_threshold).sum())
        fn_episodes = int((grp["event"] == "catastrophic_failure").sum())  # per-episode proxy

        total_maint = tp + fp
        fdr = fp / max(total_maint, 1)
        precision = tp / max(total_maint, 1)  # 1 – FDR
        recall = tp / max(tp + fn_episodes, 1)

        records.append(
            {
                "policy": policy,
                "TP (maint, RUL≤threshold)": tp,
                "FP (maint, RUL>threshold)": fp,
                "TN (cont,  RUL>threshold)": tn,
                "FN (catastrophic failures)": fn_episodes,
                "Total maintenance actions": total_maint,
                "FDR": round(fdr, 4),
                "Precision (1-FDR)": round(precision, 4),
                "Recall": round(recall, 4),
            }
        )
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
_POLICY_COLORS: dict[str, str] = {
    "Random": "#888888",
    "DoubleDQN": "#1f77b4",
    "DynaQ": "#ff7f0e",
}


def _policy_color(label: str) -> str | None:
    if label in _POLICY_COLORS:
        return _POLICY_COLORS[label]
    if label.startswith("FixedInterval"):
        return "#2ca02c"
    return None


def plot_maintenance_probability(
    prob_df: pd.DataFrame,
    output_path: Path,
    rul_threshold: int = 50,
) -> None:
    """Line chart: P(Maintenance | RUL bin) for each policy."""
    fig, ax = plt.subplots(figsize=(13, 6))

    for policy, grp in prob_df.groupby("policy", sort=False):
        grp = grp.sort_values("rul_bin")
        ax.plot(
            range(len(grp)),
            grp["p_maintenance"].values,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=str(policy),
            color=_policy_color(str(policy)),
        )

    # Mark FDR threshold bin
    threshold_bin_idx = min(int(rul_threshold / 10), len(_BIN_LABELS) - 1)
    ax.axvline(
        x=threshold_bin_idx,
        color="red",
        linestyle="--",
        linewidth=1.4,
        label=f"FDR threshold (RUL={rul_threshold})",
        zorder=3,
    )
    ax.fill_betweenx(
        [0, 1.05],
        0,
        threshold_bin_idx,
        alpha=0.06,
        color="red",
        label="Near-failure zone",
    )

    ax.set_xticks(range(len(_BIN_LABELS)))
    ax.set_xticklabels(_BIN_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Remaining Useful Life (RUL) at Decision Point", fontsize=11)
    ax.set_ylabel("P(Maintenance | RUL)", fontsize=11)
    ax.set_title("Probability of Maintenance Decision vs. Remaining Useful Life", fontsize=13)
    ax.set_ylim(-0.03, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_tco(tco_summary: pd.DataFrame, output_path: Path) -> None:
    """Two-panel figure: stacked cost bars + average engine longevity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    policies = tco_summary["policy"].tolist()
    x = np.arange(len(policies))
    width = 0.55

    # — Left panel: stacked cost breakdown —
    ax = axes[0]
    m_costs = tco_summary["total_maintenance_cost"].values.astype(float)
    f_costs = tco_summary["total_failure_cost"].values.astype(float)
    totals = m_costs + f_costs

    bars_m = ax.bar(x, m_costs, width, label=f"Scheduled maintenance (×{int(MAINTENANCE_UNIT_COST)})", color="#4c78a8")
    ax.bar(x, f_costs, width, bottom=m_costs, label=f"Catastrophic failure (×{int(FAILURE_UNIT_COST)})", color="#e45756")

    for xi, total in zip(x, totals):
        ax.text(xi, total + max(totals) * 0.01, f"{total:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha="right")
    ax.set_ylabel("Total Cost")
    ax.set_title("Total Cost of Ownership — Breakdown", fontsize=12)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)

    # — Right panel: average cycles operated —
    ax = axes[1]
    avg_cycles = tco_summary["avg_cycles_per_engine"].values.astype(float)
    bars_c = ax.bar(x, avg_cycles, width, color="#54a24b")
    for bar, val in zip(bars_c, avg_cycles):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(avg_cycles) * 0.01,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha="right")
    ax.set_ylabel("Avg Cycles Operated per Engine")
    ax.set_title("Engine Longevity by Policy", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Console report
# ──────────────────────────────────────────────────────────────────────────────

def _divider(char: str = "─", width: int = 72) -> str:
    return char * width


def print_report(
    tco_summary: pd.DataFrame,
    fdr_df: pd.DataFrame,
    rul_threshold: int,
) -> None:
    print("\n" + _divider("═"))
    print("  PREDICTIVE MAINTENANCE — ADVANCED EVALUATION REPORT")
    print(_divider("═"))

    print(f"\n{'── 1. Total Cost of Ownership ':─<72}")
    cols_tco = [
        "policy", "n_engines", "scheduled_maintenance_events", "catastrophic_failure_events",
        "total_maintenance_cost", "total_failure_cost", "total_raw_cost",
        "avg_tco_per_engine", "avg_cycles_per_engine", "cost_per_1k_cycles",
    ]
    print(tco_summary[cols_tco].to_string(index=False))

    print(f"\n{'── 2. False Discovery Rate (RUL threshold = ' + str(rul_threshold) + ') ':─<72}")
    print(fdr_df.to_string(index=False))
    print("\n  Interpretation:")
    print("  • FP  = premature maintenance (engine had > {t} cycles remaining)".format(t=rul_threshold))
    print("  • FN  = episodes ending in catastrophic failure (missed maintenance)")
    print("  • FDR = FP / (TP + FP)  — fraction of maintenance decisions that were premature")

    print("\n" + _divider("═") + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate predictive maintenance policies: TCO, P(maint|RUL), FDR."
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory with CMAPSS .txt files.")
    p.add_argument("--subset", default="FD001", help="CMAPSS subset, e.g. FD001.")
    p.add_argument("--window-size", type=int, default=30, help="Sliding-window size used during training.")
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing checkpoint_DoubleDQN.pt (produced by train_compare.py).",
    )
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Where to write output files.")
    p.add_argument("--fixed-interval", type=int, default=150, help="Fixed-interval maintenance cycle count.")
    p.add_argument("--fdr-threshold", type=int, default=50, help="RUL threshold for FDR calculation.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for data split and rollout.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    preprocessor = CMAPSSPreprocessor(
        args.data_dir, subset=args.subset, window_size=args.window_size
    )
    train_df, _ = preprocessor.load()
    episodes = preprocessor.build_episodes(train_df)
    _, eval_episodes = split_episodes(episodes, validation_fraction=0.2, seed=args.seed)
    env = TurboFanEnv(eval_episodes, seed=args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load RL agent ─────────────────────────────────────────────────────────
    rl_ckpt = args.artifacts_dir / "checkpoint_DoubleDQN.pt"
    if rl_ckpt.exists():
        rl_agent = DoubleDQNAgent.from_checkpoint(str(rl_ckpt), device=device)
        print(f"Loaded checkpoint: {rl_ckpt}")
    else:
        print(
            f"[WARNING] No checkpoint found at {rl_ckpt}.\n"
            "          Using an untrained agent (equivalent to random policy).\n"
            "          Run train_compare.py first to generate a trained checkpoint."
        )
        rl_agent = DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            epsilon_start=1.0,  # fully random until trained
        )

    # ── Policy registry ───────────────────────────────────────────────────────
    policies: dict[str, Callable[[np.ndarray], int]] = {
        "Random": random_policy,
        f"FixedInterval_{args.fixed_interval}": fixed_interval_policy(args.fixed_interval),
        "DoubleDQN": greedy_rl_policy(rl_agent),
    }

    # ── Rollouts ──────────────────────────────────────────────────────────────
    print(f"\nEvaluating {len(env.episodes)} engines per policy...")
    rollout_frames: list[pd.DataFrame] = []
    for label, policy_fn in policies.items():
        print(f"  Rolling out: {label}")
        rollout_frames.append(run_rollout(env, policy_fn, label, seed=args.seed))
    all_rollouts = pd.concat(rollout_frames, ignore_index=True)
    all_rollouts.to_csv(args.output_dir / "eval_rollout.csv", index=False)

    # ── 1. TCO ────────────────────────────────────────────────────────────────
    per_engine_df = _per_engine_tco(all_rollouts)
    per_engine_df.to_csv(args.output_dir / "tco_per_engine.csv", index=False)
    tco_summary = compute_tco_summary(per_engine_df)
    tco_summary.to_csv(args.output_dir / "tco_summary.csv", index=False)

    # ── 2. P(Maintenance | RUL) ───────────────────────────────────────────────
    prob_df = compute_maintenance_probability(all_rollouts)
    prob_df.to_csv(args.output_dir / "maintenance_probability.csv", index=False)

    # ── 3. FDR ────────────────────────────────────────────────────────────────
    fdr_df = compute_fdr(all_rollouts, rul_threshold=args.fdr_threshold)
    fdr_df.to_csv(args.output_dir / "fdr_report.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_maintenance_probability(
        prob_df,
        args.output_dir / "maintenance_probability.png",
        rul_threshold=args.fdr_threshold,
    )
    plot_tco(tco_summary, args.output_dir / "tco_comparison.png")

    # ── Console report ────────────────────────────────────────────────────────
    print_report(tco_summary, fdr_df, rul_threshold=args.fdr_threshold)


if __name__ == "__main__":
    main()
