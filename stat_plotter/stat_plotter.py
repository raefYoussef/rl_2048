import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # Switch backend
import matplotlib.pyplot as plt
import os


class StatsPlotter:
    def __init__(self, agent_files):
        """
        Initialize the plotter with a dictionary of agent names and their file paths.

        Inputs:
            agent_files:    Dictionary [keys: agent names, values: file paths]
        """
        self.agent_data = {
            agent_name: pd.read_csv(file_path)
            for agent_name, file_path in agent_files.items()
        }

    def apply_filter(self, data, width):
        """
        Applies a moving average filter to the data.

        Inputs:
            data:   Pandas Series of data to filter.
            width:  Width of the moving average filter. If None, no filtering is applied.
        
        Outputs:
            Filtered data as Pandas Series.
        """
        return data.rolling(window=width, min_periods=1).mean() if width else data

    def save_figure(self, metric, agent_name=None, save_path="metrics", compare=False):
        """
        Saves the current plot to a specified directory.

        Inputs:
            metric:      The metric being plotted.
            agent_name:  Name of the agent (None for comparison plots).
            save_path:   Directory where the plot will be saved.
            compare:     Whether it's a comparison plot (True) or individual plot (False).
        """
        os.makedirs(save_path, exist_ok=True)
        filename = f"{metric}_{'comparison' if compare else agent_name}.png"
        plt.savefig(os.path.join(save_path, filename))
        plt.close()

    def plot_metric(self, metric, agents=None, compare=False, filt_width=None, mode="plot", save_path="metrics"):
        """
        Plot a specific metric for one or multiple agents, with optional moving average filtering.

        Inputs:
            metric:     The column name of the metric to plot.
            agents:     List of agent names to plot. If None, all agents are included.
            compare:    Whether to overlay plots for comparison (True) or plot separately (False).
            filt_width: Width of the moving average filter. If None, no filtering is applied.
            mode:       "plot" to only plot, "save" to only save, "save_and_plot" to do both.
            save_path:  Directory where plots will be saved.
        """
        if agents is None:
            agents = self.agent_data.keys()

        if compare:
            plt.figure(figsize=(10, 6))
            for agent in agents:
                if agent in self.agent_data:
                    data = self.agent_data[agent][metric]
                    filtered_data = self.apply_filter(data, filt_width)
                    plt.plot(filtered_data, label=agent)
            plt.title(f"Comparison of {metric} (Filter Width: {filt_width})")
            plt.xlabel("Episode")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)

            if mode in ["save", "save_and_plot"]:
                self.save_figure(metric, save_path=save_path, compare=True)
            if mode in ["plot", "save_and_plot"]:
                plt.show()
        else:
            for agent in agents:
                if agent in self.agent_data:
                    data = self.agent_data[agent][metric]
                    filtered_data = self.apply_filter(data, filt_width)
                    plt.figure(figsize=(10, 6))
                    plt.plot(
                        filtered_data,
                        label=f"{agent} (Filtered)" if filt_width else agent,
                    )
                    plt.title(f"{metric} for {agent} (Filter Width: {filt_width})")
                    plt.xlabel("Episode")
                    plt.ylabel(metric)
                    plt.legend()
                    plt.grid(True)

                    if mode in ["save", "save_and_plot"]:
                        self.save_figure(metric, agent_name=agent, save_path=save_path)
                    if mode in ["plot", "save_and_plot"]:
                        plt.show()
