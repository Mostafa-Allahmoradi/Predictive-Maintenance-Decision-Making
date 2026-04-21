from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pdm.agents import DoubleDQNAgent, DynaQAgent, Transition
from pdm.data import CMAPSSPreprocessor, EngineEpisode, split_episodes
from pdm.env import TurboFanEnv


def train_agent(
    name: str,
    agent_factory: Callable[[int, int, str], DoubleDQNAgent],
    episodes: list[EngineEpisode],
    device: str,
    num_episodes: int,
    seed: int,
    writer: SummaryWriter | None = None,
) -> tuple[DoubleDQNAgent, pd.DataFrame]:
    env = TurboFanEnv(episodes=episodes, seed=seed)
    agent = agent_factory(env.observation_space.shape[0], env.action_space.n, device)
    history: list[dict[str, float | int | str]] = []
    global_step = 0

    progress = tqdm(range(num_episodes), desc=name, unit="ep", dynamic_ncols=True)
    for episode_index in progress:
        state, _ = env.reset()
        terminated = False
        episode_reward = 0.0
        last_info: dict[str, float | int | str] = {"cycles_survived": 0, "total_cost": 0.0, "event": ""}
        episode_loss_acc = 0.0
        episode_mean_q_acc = 0.0
        steps_with_update = 0

        while not terminated:
            action = agent.act(state, explore=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=float(done),
                rul=int(info["rul"]),
            )
            result = agent.observe(transition)
            state = next_state
            episode_reward += reward
            last_info = info
            global_step += 1

            if result is not None:
                episode_loss_acc += result["loss"]
                episode_mean_q_acc += result["mean_q"]
                steps_with_update += 1
                if writer is not None:
                    writer.add_scalar(f"{name}/step/Loss", result["loss"], global_step)
                    writer.add_scalar(f"{name}/step/MeanQ", result["mean_q"], global_step)

        ep_loss = episode_loss_acc / max(steps_with_update, 1)
        ep_mean_q = episode_mean_q_acc / max(steps_with_update, 1)
        current_lr = agent.optimizer.param_groups[0]["lr"]

        if writer is not None:
            ep_num = episode_index + 1
            writer.add_scalar(f"{name}/episode/Reward", episode_reward, ep_num)
            writer.add_scalar(f"{name}/episode/CyclesSurvived", int(last_info["cycles_survived"]), ep_num)
            writer.add_scalar(f"{name}/episode/Epsilon", agent.epsilon, ep_num)
            writer.add_scalar(f"{name}/episode/LearningRate", current_lr, ep_num)
            writer.add_scalar(f"{name}/episode/MeanLoss", ep_loss, ep_num)
            writer.add_scalar(f"{name}/episode/MeanQ", ep_mean_q, ep_num)

        progress.set_postfix(
            reward=f"{episode_reward:.1f}",
            loss=f"{ep_loss:.4f}",
            q=f"{ep_mean_q:.3f}",
            eps=f"{agent.epsilon:.3f}",
            lr=f"{current_lr:.2e}",
        )

        history.append(
            {
                "agent": name,
                "episode": episode_index + 1,
                "cumulative_reward": episode_reward,
                "cycles_survived": int(last_info["cycles_survived"]),
                "total_cost": float(last_info["total_cost"]),
                "event": str(last_info["event"]),
            }
        )

    return agent, pd.DataFrame(history)


def evaluate_policy(
    env: TurboFanEnv,
    policy_fn: Callable[[np.ndarray], int],
    label: str,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for engine_index, episode in enumerate(env.episodes):
        state, _ = env.reset(options={"engine_index": engine_index})
        terminated = False
        episode_reward = 0.0
        last_info: dict[str, float | int | str] = {"cycles_survived": 0, "total_cost": 0.0}

        while not terminated:
            action = policy_fn(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            last_info = info
            terminated = terminated or truncated

        records.append(
            {
                "policy": label,
                "engine_id": episode.engine_id,
                "cumulative_reward": episode_reward,
                "cycles_survived": int(last_info["cycles_survived"]),
                "total_cost": float(last_info["total_cost"]),
                "event": str(last_info["event"]),
            }
        )
    return pd.DataFrame(records)


def fixed_interval_policy_factory(interval: int) -> Callable[[np.ndarray], int]:
    def policy(state: np.ndarray) -> int:
        cycle = int(round(float(state[-1])))
        return 1 if cycle >= interval else 0

    return policy


def save_training_plots(history_df: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for agent_name, agent_history in history_df.groupby("agent"):
        axes[0].plot(agent_history["episode"], agent_history["cumulative_reward"], label=agent_name)
        axes[1].plot(agent_history["episode"], agent_history["cycles_survived"], label=agent_name)

    axes[0].set_title("Training Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].legend()
    axes[1].set_title("Training Engine Longevity")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Cycles Survived")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close(figure)


def save_cost_plot(cost_df: pd.DataFrame, output_dir: Path) -> None:
    summary = cost_df.groupby("policy", as_index=False)["total_cost"].mean()
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(summary["policy"], summary["total_cost"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axis.set_title("Average Total Cost by Policy")
    axis.set_ylabel("Total Cost")
    axis.set_xlabel("Policy")
    figure.tight_layout()
    figure.savefig(output_dir / "total_cost_comparison.png", dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agents for turbofan predictive maintenance.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing CMAPSS text files.")
    parser.add_argument("--subset", type=str, default="FD001", help="CMAPSS subset identifier.")
    parser.add_argument("--window-size", type=int, default=30, help="Sliding window size for state construction.")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes per agent.")
    parser.add_argument("--fixed-interval", type=int, default=150, help="Baseline maintenance interval in cycles.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Output directory for metrics and plots.")
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("runs"), help="Root directory for TensorBoard event files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = CMAPSSPreprocessor(args.data_dir, subset=args.subset, window_size=args.window_size)
    train_df, test_df = preprocessor.load()
    del test_df

    episodes = preprocessor.build_episodes(train_df)
    train_episodes, eval_episodes = split_episodes(episodes, validation_fraction=0.2, seed=args.seed)
    if not eval_episodes:
        raise RuntimeError("Evaluation split is empty. Add more engines or reduce the validation fraction.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Estimated learn-steps budget: episodes * avg engine life (~250 cycles).
    lr_decay_steps = args.episodes * 250

    double_dqn_factory = lambda state_dim, action_dim, runtime_device: DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=runtime_device,
        batch_size=256,
        learning_rate=3e-4,
        lr_decay_steps=lr_decay_steps,
    )
    dyna_q_factory = lambda state_dim, action_dim, runtime_device: DynaQAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=runtime_device,
        planning_steps=100,
        batch_size=256,
        learning_rate=3e-4,
        lr_decay_steps=lr_decay_steps,
    )

    tb_log_dir = str(args.tensorboard_dir / "pdm_experiment")
    writer: SummaryWriter = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs: {tb_log_dir}  (run: tensorboard --logdir {args.tensorboard_dir})")

    double_dqn_agent, double_dqn_history = train_agent(
        name="DoubleDQN",
        agent_factory=double_dqn_factory,
        episodes=train_episodes,
        device=device,
        num_episodes=args.episodes,
        seed=args.seed,
        writer=writer,
    )
    dyna_q_agent, dyna_q_history = train_agent(
        name="DynaQ",
        agent_factory=dyna_q_factory,
        episodes=train_episodes,
        device=device,
        num_episodes=args.episodes,
        seed=args.seed + 1,
        writer=writer,
    )
    writer.close()

    double_dqn_agent.save_checkpoint(output_dir / "checkpoint_DoubleDQN.pt")
    dyna_q_agent.save_checkpoint(output_dir / "checkpoint_DynaQ.pt")
    print(f"Checkpoints saved to {output_dir}/")

    history_df = pd.concat([double_dqn_history, dyna_q_history], ignore_index=True)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    save_training_plots(history_df, output_dir)

    evaluation_env = TurboFanEnv(eval_episodes, seed=args.seed)
    cost_frames = [
        evaluate_policy(evaluation_env, lambda state: double_dqn_agent.act(state, explore=False), "DoubleDQN"),
        evaluate_policy(evaluation_env, lambda state: dyna_q_agent.act(state, explore=False), "DynaQ"),
        evaluate_policy(evaluation_env, fixed_interval_policy_factory(args.fixed_interval), f"FixedInterval_{args.fixed_interval}"),
    ]
    cost_df = pd.concat(cost_frames, ignore_index=True)
    cost_df.to_csv(output_dir / "cost_comparison.csv", index=False)
    save_cost_plot(cost_df, output_dir)

    summary = cost_df.groupby("policy", as_index=False).agg(
        average_total_cost=("total_cost", "mean"),
        average_reward=("cumulative_reward", "mean"),
        average_cycles_survived=("cycles_survived", "mean"),
    )
    summary.to_csv(output_dir / "cost_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
