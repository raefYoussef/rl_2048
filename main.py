from env import Env2048


def main():
    env = Env2048(3, 4, 5)
    state = env.get_state()
    print(state)

if __name__ == "__main__":
    main()
