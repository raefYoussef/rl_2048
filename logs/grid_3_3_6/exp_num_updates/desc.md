1. Reward delta score
    1. Params
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
        max_eps_moves=1024,
    )
    agent.learn(num_eps=10000)