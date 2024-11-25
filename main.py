import numpy as np
import os 

from env import Env2048
from ppo_agent.agent_ppo import AgentPPO
from ppo_agent.policy_mlp import PolicyMLP
from ppo_agent.policy_cnn import PolicyCNN
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
    plotter.plot_metric(metric="eps_win", filt_width=750, compare=True, mode="save", save_path=exp_dir)
    plotter.plot_metric(metric="eps_end", filt_width=750, compare=True, mode="save", save_path=exp_dir)
    plotter.plot_metric(metric="eps_len", filt_width=750, compare=True, mode="save", save_path=exp_dir)
    plotter.plot_metric(metric="eps_score", filt_width=750, compare=True, mode="save", save_path=exp_dir)
    plotter.plot_metric(metric="eps_max_tile", filt_width=750, compare=True, mode="save", save_path=exp_dir)
    plotter.plot_metric(metric="eps_rewards", filt_width=750, compare=True, mode="save", save_path=exp_dir)
    
def exp_gamma():
    exp_dir = "logs/grid_3_3_6/exp_gamma/"
    sweep = [1, .99, .95, .9, .8, .5, 0]
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
    plotter.plot_metric(metric="eps_win", filt_width=750, compare=True, mode="save", save_path=exp_dir)
    plotter.plot_metric(metric="eps_end", filt_width=750, compare=True, mode="save", save_path=exp_dir)
    plotter.plot_metric(metric="eps_len", filt_width=750, compare=True, mode="save", save_path=exp_dir)
    plotter.plot_metric(metric="eps_score", filt_width=750, compare=True, mode="save", save_path=exp_dir)
    plotter.plot_metric(metric="eps_max_tile", filt_width=750, compare=True, mode="save", save_path=exp_dir)
    plotter.plot_metric(metric="eps_rewards", filt_width=750, compare=True, mode="save", save_path=exp_dir)
    
    

if __name__ == "__main__":
    exp_updates()
    # exp_gamma()