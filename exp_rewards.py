import numpy as np
import os

from env import Env2048
from ppo_agent.agent_ppo import AgentPPO
from ppo_agent.policy_mlp import PolicyMLP
from ppo_agent.policy_cnn import PolicyCNN
from stat_plotter.stat_plotter import StatsPlotter


def reward_merging(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Encourage merging.
    """

    reward = tot_merged
    return reward


def reward_empty_tiles(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Reward creating empty tiles.
    """

    old_empty = np.sum(old_grid == 0)
    new_empty = np.sum(new_grid == 0)
    reward = new_empty - old_empty  # Reward difference in empty cells
    return reward


def reward_win(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Encourage wins.
    """

    reward = 0
    if end and win:
        reward = 100
    return reward


def reward_max_tile(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Encourage reaching new tiles.
    """

    reward = 0
    old_max = np.max(old_grid)
    new_max = np.max(new_grid)

    if new_max > old_max:
        reward = new_max.item()
    return reward


def reward_new_tiles(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Encourage making new tiles.
    """

    reward = 0
    new_unique = np.unique(new_grid)
    old_unique = np.unique(old_grid)
    new_merged = np.setdiff1d(new_unique, old_unique)
    max_merged = np.max(new_merged) if new_merged.size > 0 else 0

    # reward tiles above 4 since 2 and 4 are automatically added
    if max_merged > 2:
        reward = max_merged
    return reward


def penalize_loss(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Discourage losses.
    """
    reward = 0
    if end and not win:
        reward = -100
    return reward


def penalize_no_op(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Discourage non-moves.
    """
    reward = 0
    if np.all(old_grid == new_grid):
        reward = -0.5
    return reward


def exp_rewards():
    exp_dir = "logs/grid_3_3_6/exp_rewards/"
    rewards = {
        "tot_merged": [reward_merging],
        "max_tile": [reward_max_tile],
        "new_tiles": [reward_new_tiles],
        "empty_tiles": [reward_empty_tiles],
        "win_loss": [reward_win, penalize_loss],
    }
    agent_files = {}

    os.makedirs(exp_dir, exist_ok=True)

    for reward_name, reward_list in rewards.items():
        # Lambda to sum the results of the reward functions
        reward_fn = lambda *args: sum(func(*args) for func in reward_list)
        env = Env2048(3, 3, 6, debug=True, reward_fn=reward_fn)
        agent = AgentPPO(
            env=env,
            policy=PolicyMLP,
            policy_hidden_dim=64,
            seed=1000,
            gamma=0.99,
            clip=0.2,
            num_updates=200,
            lr=1e-4,
            max_batch_moves=4096,
            max_eps_moves=512,
        )
        agent.learn(num_eps=10000)

        log_file = exp_dir + f"train_log_{reward_name}.csv"
        agent.log_statistics(log_file)

        agent_files[f"Reward Fn: {reward_name}"] = log_file

    plotter = StatsPlotter(agent_files)
    plotter.plot_metric(
        metric="eps_win", filt_width=750, compare=True, mode="save", save_path=exp_dir
    )
    plotter.plot_metric(
        metric="eps_end", filt_width=750, compare=True, mode="save", save_path=exp_dir
    )
    plotter.plot_metric(
        metric="eps_len", filt_width=750, compare=True, mode="save", save_path=exp_dir
    )
    plotter.plot_metric(
        metric="eps_score", filt_width=750, compare=True, mode="save", save_path=exp_dir
    )
    plotter.plot_metric(
        metric="eps_max_tile",
        filt_width=750,
        compare=True,
        mode="save",
        save_path=exp_dir,
    )
    plotter.plot_metric(
        metric="eps_rewards",
        filt_width=750,
        compare=True,
        mode="save",
        save_path=exp_dir,
    )


if __name__ == "__main__":
    exp_rewards()
