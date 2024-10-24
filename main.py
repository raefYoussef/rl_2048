from env import Env2048
import numpy as np

def main():
    env = Env2048(3, 4, 64)
    rng = np.random.default_rng(seed=100)
    end = False
    while not end:
        move = rng.integers(0, 3)
        state, reward, score, end, win = env.move(move)
    env.log_history("history.csv")

if __name__ == "__main__":
    main()
