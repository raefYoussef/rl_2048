import os
import numpy as np

from env2048.env2048 import Env2048
from env2048.rewards import *
from agent_ppo.agent_ppo import AgentPPO
from agent_ppo.policy_mlp import PolicyMLP
from agent_ppo.policy_cnn import PolicyCNN
from stat_plotter.stat_plotter import StatsPlotter


def exp_rewards():
    exp_dir = "logs/grid_3_3_6/exp_rewards/"
    rewards = {
        ##--------------------------------------
        ## Base Rewards
        ##--------------------------------------
        "tot_merged": [reward_merging],
        # "max_tile": [reward_max_tile],
        # "new_tiles": [reward_new_tiles],
        # "empty_tiles": [reward_empty_tiles],
        ##--------------------------------------
        ## Base Rewards + Augmentation
        ##--------------------------------------
        "tot_merged_mov_tiles": [
            lambda *args: (reward_merging(*args) + 0.1 * penalize_moved_tiles(*args))
        ],
        "tot_merged_nop": [
            lambda *args: (reward_merging(*args) + 0.1 * penalize_nop(*args))
        ],
        "tot_merged_dir": [
            lambda *args: (reward_merging(*args) + 0.01 * reward_directions(*args))
        ],
        "tot_merged_nop_mov_tiles": [
            lambda *args: (
                reward_merging(*args)
                + 0.1 * penalize_nop(*args)
                + 0.1 * penalize_moved_tiles(*args)
            )
        ],
        "tot_merged_dir_nop": [
            lambda *args: (
                reward_merging(*args)
                + 0.01 * reward_directions(*args)
                + 0.1 * penalize_nop(*args)
            )
        ],
        "tot_merged_dir_mov_tiles": [
            lambda *args: (
                reward_merging(*args)
                + 0.01 * reward_directions(*args)
                + 0.1 * penalize_moved_tiles(*args)
            )
        ],
        "tot_merged_dir_nop_mov_tiles": [
            lambda *args: (
                reward_merging(*args)
                + 0.01 * reward_directions(*args)
                + 0.1 * penalize_nop(*args)
                + 0.1 * penalize_moved_tiles(*args)
            )
        ],
        # "max_tile_nop": [reward_max_tile, penalize_nop],
        # "new_tiles_nop": [reward_new_tiles, penalize_nop],
        # "empty_tiles_nop": [reward_empty_tiles, penalize_nop],
        ##--------------------------------------
        ## Ineffective rewards (moderate performance)
        ##--------------------------------------
        # "win_nop": [reward_win, penalize_nop],
        # "win_loss_nop": [reward_win, penalize_loss, penalize_nop],
        ##--------------------------------------
        ## Bad rewards (teaches agent to play longer games, not win them)
        ##--------------------------------------
        # "win": [reward_win],
        # "win_loss": [reward_win, penalize_loss],
        # "score": [reward_score],
        # "reward_directions": [reward_directions]
    }
    agent_files = {}

    os.makedirs(exp_dir, exist_ok=True)

    for reward_name, reward_list in rewards.items():
        log_file = exp_dir + f"train_log_{reward_name}.csv"

        # Lambda to sum the results of the reward functions
        reward_fn = lambda *args: sum(func(*args) for func in reward_list)
        env = Env2048(3, 3, 6, debug=True, reward_fn=reward_fn, onehot_enc=True)
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
        agent.log_statistics(log_file)

        agent_files[f"Reward Fn: {reward_name}"] = log_file

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
    exp_rewards()
