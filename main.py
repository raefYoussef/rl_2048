from env import Env2048
from ppo_agent.agent_ppo import AgentPPO
from ppo_agent.policy_mlp import PolicyMLP
from ppo_agent.policy_cnn import PolicyCNN
import numpy as np


def main():
    env = Env2048(3, 3, 6, debug=False)
    agent = AgentPPO(
        env=env,
        policy=PolicyMLP,
        policy_hidden_dim=64,
        seed=1000,
        gamma=0.99,
        clip=0.2,
        num_updates=100,
        lr=1e-4,
        max_batch_moves=4096,
    )
    agent.learn(num_eps=10000)
    agent.log_statistics("train_log.csv")


if __name__ == "__main__":
    main()
