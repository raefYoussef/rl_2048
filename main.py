from env import Env2048
from ppo_agent.agent_ppo import AgentPPO
from ppo_agent.policy_mlp import PolicyMLP
from ppo_agent.policy_cnn import PolicyCNN
from stat_plotter.stat_plotter import StatsPlotter
import numpy as np


def main():
    env = Env2048(3, 3, 6, debug=False)
    agent = AgentPPO(
        env=env,
        policy=PolicyCNN,
        policy_hidden_dim=64,
        seed=1000,
        gamma=0.99,
        clip=0.2,
        num_updates=100,
        lr=1e-4,
        max_batch_moves=4096,
        max_eps_moves=512,
    )
    agent.learn(num_eps=10000)
    agent.log_statistics("train_log.csv")


def compare_agent():
    # Hidden Size Exp
    agent_files = {
        # "Agent_18": "./logs/grid_3_3_6/train_log_18.csv",  # MLP, new tile, 32
        # "Agent_17": "./logs/grid_3_3_6/train_log_17.csv",  # MLP, new tile, 256
        # "Agent_19": "./logs/grid_3_3_6/train_log_19.csv",  # MLP, new tile, 128
        # "Agent_14": "./logs/grid_3_3_6/train_log_14.csv",
        "Agent_15": "./logs/grid_3_3_6/train_log_15.csv",  # CNN, delta score
        "Agent_16": "./logs/grid_3_3_6/train_log_16.csv",  # CNN, new tile
    }

    plotter = StatsPlotter(agent_files)
    plotter.plot_metric(metric="eps_win", filt_width=750, compare=True)


if __name__ == "__main__":
    main()
    # compare_agent()
