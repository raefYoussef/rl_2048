from env import Env2048
from ppo_agent.agent_ppo import AgentPPO
from ppo_agent.policy_mlp import PolicyMLP
import numpy as np


def main():
    env = Env2048(4, 4, 7)
    agent = AgentPPO(
        env=env, policy=PolicyMLP, seed=1000, gamma=0.99, num_updates=5, lr=1e-5
    )
    agent.learn(num_eps=10000)
    agent.log_statistics("train_log.csv")


if __name__ == "__main__":
    main()
