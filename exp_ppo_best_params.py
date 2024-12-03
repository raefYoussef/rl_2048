import os
import numpy as np
from itertools import combinations
import torch

from env2048.env2048 import Env2048
from env2048.rewards import *
from agent_ppo.agent_ppo import AgentPPO
from agent_ppo.policy_mlp import PolicyMLP
from agent_ppo.policy_cnn import PolicyCNN
from stat_plotter.stat_plotter import StatsPlotter


def exp_best():
    exp_dir = "logs/grid_3_3_6/exp_best/"
    models_dir = "models/ppo/"
    device = torch.device("cpu")
    reward_fn = reward_merging_penalize_moved_tiles
    policy_dim = 64
    num_agents = 5

    networks = {
        "MLP": PolicyMLP,
    }
    grids = {
        # "3x3x6": [3, 3, 6],
        "4x4x11": [4, 4, 11],
    }
    agent_files = {}

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    for grid_name, grid_arr in grids.items():
        for policy_name, policy_network in networks.items():
            for agentIdx in range(num_agents):
                actor_file = (
                    models_dir
                    + f"ppo_actor_{grid_name}_{policy_name}_{policy_dim}_{agentIdx}.pth"
                )
                critic_file = (
                    models_dir
                    + f"ppo_critic_{grid_name}_{policy_name}_{policy_dim}_{agentIdx}.pth"
                )
                log_file = (
                    exp_dir
                    + f"train_log_{grid_name}_{policy_name}_{policy_dim}_{agentIdx}.csv"
                )

                env = Env2048(
                    grid_arr[0],
                    grid_arr[1],
                    grid_arr[2],
                    debug=False,
                    onehot_enc=True,
                    reward_fn=reward_fn,
                )
                agent = AgentPPO(
                    env=env,
                    seed=1000,
                    policy=policy_network,
                    policy_hidden_dim=64,
                    lr=1e-4,
                    gamma=0.85,
                    clip=0.2,
                    max_batch_moves=4096,
                    num_updates=150,
                    max_eps_moves=512,
                    # target_kl=0.05,
                    device=device,
                    actor_path=actor_file,
                    critic_path=critic_file,
                )
                agent.learn(num_eps=50000)
                agent.log_statistics(log_file)

                agent_files[f"Grid: {grid_name}, Agent: {agentIdx}"] = log_file

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
    exp_best()
