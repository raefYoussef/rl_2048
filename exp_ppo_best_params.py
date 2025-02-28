import os
import numpy as np
from itertools import combinations
import torch

from env2048.env2048 import Env2048
from agent_ppo.agent_ppo import AgentPPO
from agent_ppo.policy_mlp import PolicyMLP
from agent_ppo.policy_cnn import PolicyCNN
from stat_plotter.stat_plotter import StatsPlotter


def reward_merging(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Encourage merging.
    """

    reward = tot_merged
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


def exp_network():
    exp_dir = "logs/grid_3_3_6/exp_best/"
    # device = torch.device("cuda")
    device = torch.device("cpu")
    policy_dim = 64
    networks = {
        # "MLP": PolicyMLP,
        "CNN": PolicyCNN
    }
    rewards = {
        # "tot_merged": reward_merging,
        "new_tiles": reward_new_tiles,
    }
    agent_files = {}

    os.makedirs(exp_dir, exist_ok=True)

    for policy_name, policy_network in networks.items():
        for reward_name, reward_fn in rewards.items():
            # # Lambda to sum the results of the reward functions
            # reward_fn = lambda *args: sum(func(*args) for func in reward_list)

            env = Env2048(3, 3, 6, debug=True, reward_fn=reward_fn)
            agent = AgentPPO(
                env=env,
                policy=policy_network,
                policy_hidden_dim=64,
                seed=1000,
                gamma=0.99,
                clip=0.2,
                num_updates=300,
                lr=1e-5,
                target_kl=0.02,
                max_batch_moves=4096,
                max_eps_moves=512,
                actor_path=f"./models/ppo_actor_{policy_name}_{policy_dim}_{reward_name}.pth",
                critic_path=f"./models/ppo_critic_{policy_name}_{policy_dim}_{reward_name}.pth",
                device=device,
            )
            agent.learn(num_eps=100000)

            log_file = (
                exp_dir + f"train_log_{policy_name}_{policy_dim}_{reward_name}.csv"
            )
            agent.log_statistics(log_file)

            agent_files[
                f"{policy_name} Policy, Dim: {policy_dim}, Reward: {reward_name}"
            ] = log_file

    plotter = StatsPlotter(agent_files)
    plotter.plot_metric(
        metric="eps_win", filt_width=500, compare=True, mode="save", save_path=exp_dir
    )
    plotter.plot_metric(
        metric="eps_end", filt_width=500, compare=True, mode="save", save_path=exp_dir
    )
    plotter.plot_metric(
        metric="eps_len", filt_width=500, compare=True, mode="save", save_path=exp_dir
    )
    plotter.plot_metric(
        metric="eps_score", filt_width=500, compare=True, mode="save", save_path=exp_dir
    )
    plotter.plot_metric(
        metric="eps_max_tile",
        filt_width=500,
        compare=True,
        mode="save",
        save_path=exp_dir,
    )
    plotter.plot_metric(
        metric="eps_rewards",
        filt_width=500,
        compare=True,
        mode="save",
        save_path=exp_dir,
    )


if __name__ == "__main__":
    exp_network()
