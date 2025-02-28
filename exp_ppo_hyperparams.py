import os
import numpy as np
import torch

from env2048.env2048 import Env2048
from agent_ppo.agent_ppo import AgentPPO
from agent_ppo.policy_mlp import PolicyMLP
from agent_ppo.policy_cnn import PolicyCNN
from stat_plotter.stat_plotter import StatsPlotter


def exp_updates():
    exp_dir = "logs/grid_3_3_6/exp_num_updates/"
    num_updates = [5, 10, 50, 100, 500]
    agent_files = {}

    os.makedirs(exp_dir, exist_ok=True)

    for num_upd in num_updates:
        env = Env2048(3, 3, 6, debug=True)
        agent = AgentPPO(
            env=env,
            policy=PolicyMLP,
            policy_hidden_dim=64,
            seed=1000,
            gamma=0.99,
            clip=0.2,
            num_updates=num_upd,
            lr=1e-4,
            max_batch_moves=4096,
            max_eps_moves=512,
        )
        agent.learn(num_eps=10000)

        log_file = f"logs/grid_3_3_6/exp_num_updates/train_log_{num_upd}.csv"
        agent.log_statistics(log_file)

        agent_files[f"Num Updates: {num_upd}"] = log_file

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


def exp_gamma():
    exp_dir = "logs/grid_3_3_6/exp_gamma/"
    sweep = [1, 0.99, 0.95, 0.9, 0.8, 0.5]
    agent_files = {}

    os.makedirs(exp_dir, exist_ok=True)

    for gamma in sweep:
        env = Env2048(3, 3, 6, debug=True)
        agent = AgentPPO(
            env=env,
            policy=PolicyMLP,
            policy_hidden_dim=64,
            seed=1000,
            gamma=gamma,
            clip=0.2,
            num_updates=100,
            lr=1e-4,
            max_batch_moves=4096,
            max_eps_moves=512,
        )
        agent.learn(num_eps=10000)

        log_file = exp_dir + f"train_log_{gamma}.csv"
        agent.log_statistics(log_file)

        agent_files[f"Gamma: {gamma}"] = log_file

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


def exp_clipping():
    exp_dir = "logs/grid_3_3_6/exp_clipping/"
    sweep = [0.05, 0.1, 0.2, 0.3, 0.4]
    agent_files = {}

    os.makedirs(exp_dir, exist_ok=True)

    for clip in sweep:
        env = Env2048(3, 3, 6, debug=True)
        agent = AgentPPO(
            env=env,
            policy=PolicyMLP,
            policy_hidden_dim=64,
            seed=1000,
            gamma=0.99,
            clip=clip,
            num_updates=100,
            lr=1e-4,
            max_batch_moves=4096,
            max_eps_moves=512,
        )
        agent.learn(num_eps=10000)

        log_file = exp_dir + f"train_log_{clip}.csv"
        agent.log_statistics(log_file)

        agent_files[f"Clip: {clip}"] = log_file

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


def exp_batch():
    exp_dir = "logs/grid_3_3_6/exp_batch/"
    batch_sweep = [512, 1024, 2048, 4096, 8192]
    update_sweep = [25, 50, 100, 200, 400]
    agent_files = {}

    os.makedirs(exp_dir, exist_ok=True)

    for idx in range(len(batch_sweep)):
        batch = batch_sweep[idx]
        update = update_sweep[idx]

        env = Env2048(3, 3, 6, debug=True)
        agent = AgentPPO(
            env=env,
            policy=PolicyMLP,
            policy_hidden_dim=64,
            seed=1000,
            gamma=0.99,
            clip=0.2,
            num_updates=update,
            lr=1e-4,
            max_batch_moves=batch,
            max_eps_moves=512,
        )
        agent.learn(num_eps=10000)

        log_file = exp_dir + f"train_log_{batch}.csv"
        agent.log_statistics(log_file)

        agent_files[f"Batch: {batch}"] = log_file

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


def exp_lr():
    exp_dir = "logs/grid_3_3_6/exp_lr/"
    sweep = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    agent_files = {}

    os.makedirs(exp_dir, exist_ok=True)

    for lr in sweep:
        env = Env2048(3, 3, 6, debug=True)
        agent = AgentPPO(
            env=env,
            policy=PolicyMLP,
            policy_hidden_dim=64,
            seed=1000,
            gamma=0.99,
            clip=0.4,
            num_updates=100,
            lr=lr,
            max_batch_moves=4096,
            max_eps_moves=512,
        )
        agent.learn(num_eps=10000)

        log_file = exp_dir + f"train_log_{lr}.csv"
        agent.log_statistics(log_file)

        agent_files[f"lr: {lr}"] = log_file

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


def exp_kl():
    exp_dir = "logs/grid_3_3_6/exp_kl/"
    sweep = [None, 0.01, 0.02, 0.03, 0.04, 0.05]
    agent_files = {}

    os.makedirs(exp_dir, exist_ok=True)

    for kl in sweep:
        if kl:
            num_updates = 1000
        else:
            num_updates = 100

        env = Env2048(3, 3, 6, debug=True)
        agent = AgentPPO(
            env=env,
            policy=PolicyMLP,
            policy_hidden_dim=64,
            seed=1000,
            gamma=0.99,
            clip=0.4,
            num_updates=num_updates,
            lr=1e-4,
            target_kl=kl,
            max_batch_moves=4096,
            max_eps_moves=512,
        )
        agent.learn(num_eps=10000)

        log_file = exp_dir + f"train_log_{kl}.csv"
        agent.log_statistics(log_file)

        agent_files[f"KL Thresh: {kl}"] = log_file

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


def exp_grad_norm():
    exp_dir = "logs/grid_3_3_6/exp_grad_norm/"
    sweep = [None, 0.1, 0.3, 0.5, 0.7, 0.9]
    agent_files = {}

    os.makedirs(exp_dir, exist_ok=True)

    for max_grad in sweep:
        env = Env2048(3, 3, 6, debug=True)
        agent = AgentPPO(
            env=env,
            policy=PolicyMLP,
            policy_hidden_dim=64,
            seed=1000,
            gamma=0.99,
            clip=0.4,
            num_updates=100,
            lr=1e-4,
            max_grad_norm=max_grad,
            max_batch_moves=4096,
            max_eps_moves=512,
        )
        agent.learn(num_eps=10000)

        log_file = exp_dir + f"train_log_{max_grad}.csv"
        agent.log_statistics(log_file)

        agent_files[f"Max Grad: {max_grad}"] = log_file

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
    # exp_updates()
    # exp_gamma()
    # exp_clipping()
    # exp_batch()
    # exp_lr()
    exp_kl()
    exp_grad_norm()
