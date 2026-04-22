"""
sensitivity_analysis.py
=======================
Economic sensitivity analysis for the Predictive Maintenance RL system.

The central question
--------------------
The RL agent was trained with a fixed cost ratio:
    failure_cost = 100,  maintenance_cost = 20  →  ratio 5:1

What happens when that assumption changes?
    • ratio  2.5:1  →  commercial drone, cheap hardware
    • ratio    5:1  →  **training scenario**  (industrial motor)
    • ratio   10:1  →  manufacturing line downtime
    • ratio   15:1  →  automotive powertrain
    • ratio   25:1  →  commercial aviation engine
    • ratio   50:1  →  safety-critical / medical device

Method
------
1.  Run one rollout per policy under the training environment (physical dynamics
    are fixed; the Q-network was frozen at checkpoint time).
2.  Record per-engine outcomes: #failures, #maintenance events, cycles operated,
    and the RUL at every maintenance trigger.
3.  Sweep failure_cost over FAILURE_COST_SWEEP while holding maintenance_cost = 20.
    For each scenario:
        total_cost = n_failures × failure_cost  +  n_maintenance × 20
4.  Plot how cost-ranking shifts across the sweep.  A policy that is optimal at
    one ratio but dominated at another has NOT learned the general cost structure —
    it has merely memorised a fixed threshold.

Outputs
-------
  artifacts/sensitivity_cost_curves.png   — avg cost/engine vs. failure penalty
  artifacts/sensitivity_rul_at_maint.png  — violin: RUL distribution at maint. trigger
  artifacts/sensitivity_summary.csv       — full numeric table for all scenarios

Usage
-----
    python sensitivity_analysis.py \\
        --data-dir data \\
        --artifacts-dir artifacts \\
        --output-dir artifacts \\
        --fixed-interval 150
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

from evaluate_metrics import (
    fixed_interval_policy,
    greedy_rl_policy,
    random_policy,
    run_rollout,
)
from pdm.ddqn_agent import DoubleDQNAgent
from pdm.data import CMAPSSPreprocessor, split_episodes
from pdm.env import TurboFanEnv
from pdm import config as CFG

# ──────────────────────────────────────────────────────────────────────────────
# Economic sweep parameters (read from .env via pdm.config)
# ──────────────────────────────────────────────────────────────────────────────

MAINTENANCE_UNIT_COST: float  = CFG.COST_MAINTENANCE_UNIT
TRAINING_FAILURE_COST: float  = CFG.COST_FAILURE_UNIT
FAILURE_COST_SWEEP: list[float] = CFG.SENS_FAILURE_COST_SWEEP

# Mapping failure cost → descriptive domain label (for annotation)
_DOMAIN_LABELS: dict[float, str] = {
    50.0:    "2.5:1\nDrone",
    100.0:   "5:1\nIndustrial\n(training)",
    200.0:   "10:1\nMfg. line",
    300.0:   "15:1\nAutomotive",
    500.0:   "25:1\nAviation",
    1_000.0: "50:1\nMedical",
}

# ──────────────────────────────────────────────────────────────────────────────
# Plot style
# ──────────────────────────────────────────────────────────────────────────────

_POLICY_COLORS: dict[str, str] = {
    "Random":      "#888888",
    "DoubleDQN":   "#1f77b4",
    "DynaQ":       "#ff7f0e",
    "FixedInterval": "#2ca02c",
}


def _policy_color(label: str) -> str:
    for key, color in _POLICY_COLORS.items():
        if label.startswith(key):
            return color
    return "#999999"


# ──────────────────────────────────────────────────────────────────────────────
# Per-engine event extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_per_engine_events(rollout: pd.DataFrame) -> pd.DataFrame:
    """Collapse each engine episode to one row: event counts + RUL at maintenance."""
    rows: list[dict] = []
    for (policy, eng_id, eng_idx), grp in rollout.groupby(
        ["policy", "engine_id", "engine_index"], sort=False
    ):
        terminal = str(grp.iloc[-1]["event"])
        cycles = int((grp["event"] == "safe_operation").sum())

        # All maintenance-trigger RULs (terminal and mid-episode for baselines)
        maint_rows = grp[grp["action"] == 1]
        rul_at_maint: list[int] = maint_rows["rul"].tolist()

        rows.append(
            {
                "policy": policy,
                "engine_id": eng_id,
                "engine_index": eng_idx,
                "terminal_event": terminal,
                "cycles_operated": cycles,
                "n_failures": 1 if terminal == "catastrophic_failure" else 0,
                "n_maintenance": 1 if terminal == "scheduled_maintenance" else 0,
                # Store the terminal-maintenance RUL as a scalar (NaN if not maintained)
                "rul_at_maintenance": int(grp.iloc[-1]["rul"])
                if terminal == "scheduled_maintenance"
                else float("nan"),
                # All mid-episode maintenance RULs (list serialised as json string;
                # used only for the violin plot, not the cost sweep)
                "_all_maint_ruls": rul_at_maint,
            }
        )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Cost computation
# ──────────────────────────────────────────────────────────────────────────────

def _cost_at(per_engine: pd.DataFrame, failure_cost: float) -> pd.DataFrame:
    """Compute per-policy average cost given a hypothetical failure_cost."""
    df = per_engine.copy()
    df["total_cost"] = (
        df["n_failures"] * failure_cost
        + df["n_maintenance"] * MAINTENANCE_UNIT_COST
    )
    summary = (
        df.groupby("policy", sort=False)
        .agg(
            avg_cost_per_engine=("total_cost", "mean"),
            total_cost=("total_cost", "sum"),
            failure_rate=("n_failures", "mean"),
            maintenance_rate=("n_maintenance", "mean"),
            avg_cycles=("cycles_operated", "mean"),
        )
        .reset_index()
    )
    summary["failure_cost_scenario"] = failure_cost
    summary["cost_ratio"] = round(failure_cost / MAINTENANCE_UNIT_COST, 1)
    return summary


def build_sensitivity_table(
    per_engine: pd.DataFrame,
    sweep: list[float] | None = None,
) -> pd.DataFrame:
    if sweep is None:
        sweep = FAILURE_COST_SWEEP
    return pd.concat(
        [_cost_at(per_engine, fc) for fc in sweep], ignore_index=True
    )


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 — Cost curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_cost_curves(
    sensitivity: pd.DataFrame,
    output_path: Path,
) -> None:
    """Two-panel figure: avg cost vs. failure penalty (absolute + ratio scale)."""
    policies = sorted(sensitivity["policy"].unique())
    failure_costs = sorted(sensitivity["failure_cost_scenario"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Sensitivity Analysis — Average Cost per Engine vs. Failure Penalty\n"
        "(Agent policy is frozen; only the economic accounting assumption changes)",
        fontsize=12,
        y=1.01,
    )

    for ax_idx, (ax, x_col, x_vals, x_label, x_fmt) in enumerate(
        [
            (axes[0], "failure_cost_scenario", failure_costs,
             "Failure Penalty (absolute $)", "${x:.0f}"),
            (axes[1], "cost_ratio",
             sorted(sensitivity["cost_ratio"].unique()),
             "Cost Ratio  (Failure ÷ Maintenance)", "{x:.1f}:1"),
        ]
    ):
        for policy in policies:
            pdata = (
                sensitivity[sensitivity["policy"] == policy]
                .sort_values(x_col)
            )
            ax.plot(
                pdata[x_col].values,
                pdata["avg_cost_per_engine"].values,
                marker="o",
                markersize=6,
                linewidth=2.2,
                label=str(policy),
                color=_policy_color(str(policy)),
                zorder=3,
            )

        # Vertical line at training assumption
        training_x = (
            TRAINING_FAILURE_COST
            if x_col == "failure_cost_scenario"
            else round(TRAINING_FAILURE_COST / MAINTENANCE_UNIT_COST, 1)
        )
        ymax = sensitivity.groupby(x_col)["avg_cost_per_engine"].max().max()
        ax.axvline(
            x=training_x,
            color="#e8c400",
            linestyle="--",
            linewidth=1.6,
            label=f"Training assumption ({training_x})",
            zorder=4,
        )
        # Shade the high-stakes region (above training cost)
        ax.fill_betweenx(
            [0, ymax * 1.15],
            training_x,
            x_vals[-1] * 1.05,
            alpha=0.06,
            color="#e63946",
            zorder=1,
        )
        ax.text(
            training_x * 1.02,
            ymax * 1.08,
            "High-stakes\nzone",
            fontsize=8,
            color="#e63946",
            va="top",
        )

        # Domain labels on x-ticks (absolute panel only)
        if x_col == "failure_cost_scenario":
            ax.set_xticks(failure_costs)
            ax.set_xticklabels(
                [_DOMAIN_LABELS.get(fc, f"${fc:.0f}") for fc in failure_costs],
                fontsize=8,
            )
        else:
            ax.set_xticks(sorted(sensitivity["cost_ratio"].unique()))
            ax.set_xticklabels(
                [f"{r:.1f}:1" for r in sorted(sensitivity["cost_ratio"].unique())],
                fontsize=9,
            )

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Avg Cost per Engine", fontsize=11)
        ax.set_ylim(bottom=0, top=ymax * 1.18)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))

    axes[0].set_title("Absolute Failure Penalty", fontsize=12)
    axes[1].set_title("Cost Ratio  (domain-agnostic KPI)", fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2 — RUL at maintenance (reveals implicit cost-ratio assumption)
# ──────────────────────────────────────────────────────────────────────────────

def plot_rul_at_maintenance(
    per_engine: pd.DataFrame,
    output_path: Path,
) -> None:
    """Violin + scatter: distribution of RUL at the moment maintenance is triggered.

    Interpretation guide
    --------------------
    • Tight cluster at LOW RUL  → agent is aggressive (runs engine close to failure).
      Implicitly trained for low cost-ratio.
    • Tight cluster at HIGH RUL → agent is conservative (expensive early maintenance).
      Implicitly trained for high cost-ratio.
    • Wide spread               → heuristic / un-learned threshold (random or rule-based).
    """
    # Explode the per-engine scalar `rul_at_maintenance` into per-row records
    # (also include any mid-episode maintenance from Fixed-Interval policies)
    all_maint: list[dict] = []
    for _, row in per_engine.iterrows():
        if not np.isnan(row["rul_at_maintenance"]):
            all_maint.append(
                {"policy": row["policy"], "rul": float(row["rul_at_maintenance"])}
            )
    if not all_maint:
        print("[WARNING] No maintenance events — skipping RUL violin plot.")
        return

    maint_df = pd.DataFrame(all_maint)
    policies = sorted(maint_df["policy"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    data_by_policy = [
        maint_df[maint_df["policy"] == p]["rul"].values for p in policies
    ]

    # Violin plot
    parts = ax.violinplot(
        data_by_policy,
        positions=range(len(policies)),
        showmedians=True,
        showextrema=True,
        widths=0.65,
    )
    for body, policy in zip(parts["bodies"], policies):
        body.set_facecolor(_policy_color(str(policy)))
        body.set_alpha(0.60)
    parts["cmedians"].set_color("#ffffff")
    parts["cmedians"].set_linewidth(2.5)
    for part_key in ("cmins", "cmaxes", "cbars"):
        parts[part_key].set_color("#aaaaaa")

    # Jittered scatter overlay
    rng = np.random.default_rng(0)
    for i, (vals, policy) in enumerate(zip(data_by_policy, policies)):
        jitter = rng.uniform(-0.09, 0.09, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=20,
            alpha=0.45,
            color=_policy_color(str(policy)),
            zorder=4,
        )
        # Annotate median text
        med = float(np.median(vals))
        ax.text(
            i,
            med + 2,
            f"med={med:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#ffffff",
            zorder=5,
        )

    # Reference lines
    ax.axhline(
        y=20,
        color="#e8c400",
        linestyle="--",
        linewidth=1.4,
        label="Proximity penalty onset (RUL = 20, training signal)",
        zorder=3,
    )
    ax.axhline(
        y=50,
        color="#e63946",
        linestyle=":",
        linewidth=1.3,
        label="FDR threshold (RUL = 50)",
        zorder=3,
    )

    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(
        [str(p) for p in policies], rotation=18, ha="right", fontsize=10
    )
    ax.set_ylabel("RUL Remaining at Maintenance Trigger (cycles)", fontsize=11)
    ax.set_title(
        "Implicit Maintenance Threshold — RUL at Decision Point\n"
        "(Tight cluster = consistent learned threshold; "
        "wide spread = rule-based / random)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Interpretation text box
    ax.text(
        0.99,
        0.01,
        "LOW RUL at trigger  →  agent tolerates risk  (low failure cost assumed)\n"
        "HIGH RUL at trigger →  agent overly cautious (high failure cost assumed)\n"
        "Narrow spread       →  RL learned a calibrated, consistent threshold",
        transform=ax.transAxes,
        fontsize=8,
        color="#aaaaaa",
        va="bottom",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#111118",
            alpha=0.80,
            edgecolor="#444",
        ),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Console report
# ──────────────────────────────────────────────────────────────────────────────

def _divider(char: str = "─", width: int = 76) -> str:
    return char * width


def print_sensitivity_report(sensitivity: pd.DataFrame) -> None:
    print("\n" + _divider("═"))
    print("  SENSITIVITY ANALYSIS — ECONOMIC COST-RATIO IMPACT")
    print(_divider("═"))
    print(
        f"\n  Training assumption: failure=${TRAINING_FAILURE_COST:.0f}, "
        f"maintenance=${MAINTENANCE_UNIT_COST:.0f}  "
        f"(ratio {TRAINING_FAILURE_COST / MAINTENANCE_UNIT_COST:.0f}:1)\n"
    )

    failure_costs = sorted(sensitivity["failure_cost_scenario"].unique())
    # Build pivot for ranking
    pivot = sensitivity.pivot_table(
        index="failure_cost_scenario",
        columns="policy",
        values="avg_cost_per_engine",
    )

    max_bar = 36

    for fc in failure_costs:
        ratio = fc / MAINTENANCE_UNIT_COST
        subset = sensitivity[sensitivity["failure_cost_scenario"] == fc].sort_values(
            "avg_cost_per_engine"
        )
        best = str(subset.iloc[0]["policy"])
        marker = "  ← TRAINING SCENARIO" if fc == TRAINING_FAILURE_COST else ""
        domain = _DOMAIN_LABELS.get(fc, "")
        print(
            f"  Failure cost ${fc:.0f}  (ratio {ratio:.1f}:1)"
            f"  [{domain.replace(chr(10), ' ')}]{marker}"
        )
        max_avg = float(subset["avg_cost_per_engine"].max())
        for _, row in subset.iterrows():
            pct = row["avg_cost_per_engine"] / max(max_avg, 1.0)
            bar = "█" * max(1, int(pct * max_bar))
            star = " ★" if str(row["policy"]) == best else ""
            print(
                f"    {str(row['policy']):26s}"
                f"  avg={row['avg_cost_per_engine']:7.1f}"
                f"  fail%={row['failure_rate']*100:5.1f}"
                f"  {bar}{star}"
            )
        print()

    # Crossover detection
    print(_divider())
    print("  CROSSOVER ANALYSIS — when does policy ranking change?")
    print(_divider())
    pivot["best"] = pivot.idxmin(axis=1)
    prev_best = None
    for fc, best in pivot["best"].items():
        ratio = fc / MAINTENANCE_UNIT_COST
        domain = _DOMAIN_LABELS.get(float(fc), "").replace("\n", " ")
        note = ""
        if best != prev_best:
            note = "  ← REGIME CHANGE" if prev_best is not None else "  (initial)"
        print(
            f"  ${fc:>6.0f}  (ratio {ratio:5.1f}:1)  "
            f"{str(best):26s}  {domain}{note}"
        )
        prev_best = best

    print("\n" + _divider("═") + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sensitivity analysis: cost-ratio impact on policy performance."
    )
    p.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Directory with CMAPSS .txt files.",
    )
    p.add_argument("--subset", default=CFG.DATA_SUBSET, help="CMAPSS subset (default: FD001).")
    p.add_argument(
        "--window-size", type=int, default=CFG.DATA_WINDOW_SIZE,
        help="Sliding-window size used during training.",
    )
    p.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts"),
        help="Directory containing trained checkpoints.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("artifacts"),
        help="Where to save output files.",
    )
    p.add_argument(
        "--fixed-interval", type=int, default=CFG.SENS_FIXED_INTERVAL,
        help="Fixed-interval maintenance cycle count.",
    )
    p.add_argument("--seed", type=int, default=CFG.SENS_SEED, help="Random seed.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    preprocessor = CMAPSSPreprocessor(
        args.data_dir, subset=args.subset, window_size=args.window_size
    )
    train_df, _ = preprocessor.load()
    episodes = preprocessor.build_episodes(train_df)
    _, eval_episodes = split_episodes(
        episodes, validation_fraction=0.2, seed=args.seed
    )
    env = TurboFanEnv(eval_episodes, seed=args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load checkpoints ──────────────────────────────────────────────────────
    policies: dict[str, Callable[[np.ndarray], int]] = {
        "Random": random_policy,
        f"FixedInterval_{args.fixed_interval}": fixed_interval_policy(
            args.fixed_interval
        ),
    }

    ddqn_ckpt = args.artifacts_dir / "checkpoint_DoubleDQN.pt"
    dynaq_ckpt = args.artifacts_dir / "checkpoint_DynaQ.pt"

    for ck_path, label in [
        (ddqn_ckpt, "DoubleDQN"),
        (dynaq_ckpt, "DynaQ"),
    ]:
        if ck_path.exists():
            agent = DoubleDQNAgent.from_checkpoint(str(ck_path), device=device)
            policies[label] = greedy_rl_policy(agent)
            print(f"Loaded: {ck_path}")
        else:
            print(f"[WARNING] Checkpoint not found: {ck_path} — skipping {label}.")

    # ── Rollouts (single pass — policy behaviour is fixed) ────────────────────
    print(f"\nRunning rollouts on {len(env.episodes)} engines per policy...")
    rollout_frames: list[pd.DataFrame] = []
    for label, policy_fn in policies.items():
        print(f"  {label}")
        rollout_frames.append(
            run_rollout(env, policy_fn, label, seed=args.seed)
        )
    all_rollouts = pd.concat(rollout_frames, ignore_index=True)

    per_engine = _extract_per_engine_events(all_rollouts)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    print("\nSweeping failure costs:", FAILURE_COST_SWEEP)
    sensitivity = build_sensitivity_table(per_engine)
    sensitivity.to_csv(args.output_dir / "sensitivity_summary.csv", index=False)
    print(f"Saved: {args.output_dir / 'sensitivity_summary.csv'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_cost_curves(sensitivity, args.output_dir / "sensitivity_cost_curves.png")
    plot_rul_at_maintenance(per_engine, args.output_dir / "sensitivity_rul_at_maint.png")

    # ── Console report ────────────────────────────────────────────────────────
    print_sensitivity_report(sensitivity)


if __name__ == "__main__":
    main()
