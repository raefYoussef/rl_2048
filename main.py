from env import Env2048
import numpy as np


def main():
    env = Env2048(3, 4, 6)
    rng = np.random.default_rng(seed=100)
    end = False
    while not end:
        action = rng.integers(0, 4)  # doesn't seem to include the upper bound
        state, reward, score, end, win = env.step(action)
    env.log_history("history.csv")
    env.print_history(env.history)


if __name__ == "__main__":
    main()
