Reward Functions
1. Reward for new tile, and gamma of .9, num of updates is 100
   1. 13.54% win rate
2. Same as #1, increase gamma to .99, change num of updates to 20
   1. 19.32% win rate
3. Same as #2, increase num updates to 50
   1. Better performance, 29.09%. Shows sign of better performance with more epochs
4. Same as #3, gamma = .99, increase num updates to 100
   1. Squeezed better performance, 30.36%. 
5. Same as #4, gamma = .999
   1. worse performance, 28.25%.
6. Same as #5, lr 1e-4 -> 1e-3
   1. Oscillates and did not converge
7. Same as #5, lr 1e-4 -> 1e-5, updates 100 -> 1000
   1. Too slow and did not converge
8. Replaced Policy with CNN, reward is based on max merged tile for a move above 4
   1. policy=PolicyCNN,
        seed=1000,
        gamma=0.99,
        num_updates=100,
        lr=1e-4,
        max_batch_moves=4094,
   2. Each iteration takes ~11-14 seconds so 3x the amount. But we get a better performance 33.69%
9. Changes Reward to add penalty for non moves, otherwise it adds reward for max merged
   1.  agent = AgentPPO(
        env=env,
        policy=PolicyMLP,
        seed=1000,
        gamma=0.99,
        clip=0.2,
        num_updates=100,
        lr=1e-4,
        max_batch_moves=4094,
    )
   2. Win = 9.67%
10. Same as #4 but decrease hidden dim to 32 and add penalty for non-moves
    1.  Perf 12.53% 
    2.  The penalty hurts the performance a lot
    3.  agent = AgentPPO(
        env=env,
        policy=PolicyMLP,
        policy_hidden_dim=32,
        seed=1000,
        gamma=0.99,
        clip=0.2,
        num_updates=100,
        lr=1e-4,
        max_batch_moves=4094,
    )
11. Same as #4 but increase hidden dim to 128
    1.  Perf 30.84% 
    2.  agent = AgentPPO(
        env=env,
        policy=PolicyMLP,
        policy_hidden_dim=128,
        seed=1000,
        gamma=0.99,
        clip=0.2,
        num_updates=100,
        lr=1e-4,
        max_batch_moves=4094,
    )
12. Decrease clip to .1
    1.  19.35%
    2.  agent = AgentPPO(
        env=env,
        policy=PolicyMLP,
        policy_hidden_dim=64,
        seed=1000,
        gamma=0.99,
        clip=0.1,
        num_updates=100,
        lr=1e-4,
        max_batch_moves=4094,
    )
13. Increase Clip to .3
    1.  24%
14. Reward for new max tile, game end gives you -max for loss
    1.  33.04%
    2.  agent = AgentPPO(
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
15. CNN Policy, reward is delta score
   1. Average win toward the end was 31.01%
   2. env = Env2048(3, 3, 6, debug=False)
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
16. CNN Policy, reward is new tile
    1. 19.84%
    2. env = Env2048(3, 3, 6, debug=False)
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
17. MLP, new tile reward, 256 hidden
    1.  22.44%
    2.  env = Env2048(3, 3, 6, debug=False)
    agent = AgentPPO(
        env=env,
        policy=PolicyMLP,
        policy_hidden_dim=256,
        seed=1000,
        gamma=0.99,
        clip=0.2,
        num_updates=100,
        lr=1e-4,
        max_batch_moves=4096,
        max_eps_moves=512,
    )
18. MLP, new tile reward, 32 hidden
    1.  22.44%
    2.  env = Env2048(3, 3, 6, debug=False)
    agent = AgentPPO(
        env=env,
        policy=PolicyMLP,
        policy_hidden_dim=32,
        seed=1000,
        gamma=0.99,
        clip=0.2,
        num_updates=100,
        lr=1e-4,
        max_batch_moves=4096,
        max_eps_moves=512,
    )

