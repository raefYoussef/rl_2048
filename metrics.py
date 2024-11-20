import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class Metrics:
    def __init__(self):
        self.wins = []
        self.rewards = []
        self.scores = []
        self.steps = []
        self.high_tile = []

    def add_episode(self, win=0, reward=0, score=0, steps=0, high_tile=0):
        self.wins.append(win)   # expected only 0 or 1
        self.rewards.append(reward)
        self.scores.append(score)
        self.steps.append(steps)
        self.high_tile.append(high_tile)

    def plot_winrate(self):
        # TODO might want to not plot the 'normal' data (won't be nice to look at)
        plot_episodes(self.wins, "Episode Wins", "Win")

    def plot_rewards(self):
        # TODO Not sure what we want here do we want a cumulative reward for the whole episode? 
        # or do we want to be looking at something else?
        plot_episodes(self.rewards, "Total Episode Rewards", "Reward")

    def plot_scores(self):
        plot_episodes(self.scores, "Scores", "Score")

    def plot_steps(self):
        plot_episodes(self.steps, "Game Duration", "Moves Taken")

    def plot_hightile(self):
        # TODO might want to not plot the 'average' data (won't )
        plot_episodes(self.high_tile, "Highest Tile", "Highest Tile (2^X)")

    def save_to_file(self, filename="metrics_data.csv"):
        df = pd.DataFrame({"wins":self.wins, "rewards":self.rewards, "scores":self.scores, "steps":self.steps, "high_tile":self.high_tile})
        df.to_csv(filename)

    def read_from_file(self, filename="metrics_data.csv"):
        df = pd.read_csv(filename)
        self.wins = df["wins"].to_list()
        self.rewards = df["rewards"].to_list()
        self.scores = df["scores"].to_list()
        self.steps = df["steps"].to_list()
        self.high_tile = df["high_tile"].to_list()

def plot_episodes(data, title="Result", y_label="", average_over=100):
    # can be run after each episode, or just once at the end
    np_data = np.array(data)
    plt.figure(1)
    plt.clf()
    plt.plot(np_data)
    if len(np_data) > 1:
        # plot multiple-episode averages too
        means = np.zeros_like(np_data, dtype=np.float64)
        for index in range(len(np_data)):
            if index >= average_over:
                mean = np.mean(np_data[index-average_over:index])
            else:
                mean = np.mean(np_data[:index])
            means[index] = mean
        plt.plot(means)
        plt.title(f"{title}: ={means[-1]}")
    else:
        plt.title(f"{title}")
    plt.xlabel('Episode')
    plt.ylabel(y_label)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.display(plt.gcf())