import os
import numpy as np
from itertools import combinations

from env2048.env2048 import Env2048
from env2048.rewards import *
from agent_ppo.agent_ppo import AgentPPO
from agent_ppo.policy_mlp import PolicyMLP
from agent_ppo.policy_cnn import PolicyCNN
from stat_plotter.stat_plotter import StatsPlotter


def exp_network():
    exp_dir = "logs/grid_3_3_6/exp_network/"
    networks = {"MLP": PolicyMLP, "CNN": PolicyCNN}

    rewards = {
        "tot_merged": [reward_merging],
        "max_tile": [reward_max_tile],
        "new_tiles": [reward_new_tiles],
    }
    agent_files = {}

    os.makedirs(exp_dir, exist_ok=True)

    for network_name, network in networks.items():
        for reward_name, reward_list in rewards.items():
            # Lambda to sum the results of the reward functions
            reward_fn = lambda *args: sum(func(*args) for func in reward_list)
            env = Env2048(3, 3, 6, debug=True, reward_fn=reward_fn, onehot_enc=True)
            agent = AgentPPO(
                env=env,
                policy=network,
                policy_hidden_dim=64,
                seed=1000,
                gamma=0.99,
                clip=0.2,
                num_updates=200,
                lr=1e-4,
                max_batch_moves=4096,
                max_eps_moves=512,
            )
            agent.learn(num_eps=70000)

            log_file = exp_dir + f"train_log_{network_name}_{reward_name}.csv"
            agent.log_statistics(log_file)

            agent_files[f"Network: {network_name}, Reward: {reward_name}"] = log_file

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


def exp_hidden():
    exp_dir = "logs/grid_3_3_6/exp_mlp_hidden/"
    sweep = [16, 32, 64, 128, 256]
    agent_files = {}

    os.makedirs(exp_dir, exist_ok=True)

    for hidden in sweep:
        env = Env2048(3, 3, 6, debug=True, reward_fn=reward_merging, onehot_enc=True)
        agent = AgentPPO(
            env=env,
            policy=PolicyMLP,
            policy_hidden_dim=hidden,
            seed=1000,
            gamma=0.99,
            clip=0.2,
            num_updates=200,
            lr=1e-4,
            max_batch_moves=4096,
            max_eps_moves=512,
        )
        agent.learn(num_eps=10000)

        log_file = exp_dir + f"train_log_{hidden}.csv"
        agent.log_statistics(log_file)

        agent_files[f"MLP Hidden Dim: {hidden}"] = log_file

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
    # exp_network()
    exp_hidden()
