import pandas as pd
import matplotlib.pyplot as plt


class StatsPlotter:
    def __init__(self, agent_files):
        """
        Initialize the plotter with a dictionary of agent names and their file paths.

        Inputs:
            agent_files:    Dictionary [keys:agent names, values: file paths]
        """
        self.agent_data = {
            agent_name: pd.read_csv(file_path)
            for agent_name, file_path in agent_files.items()
        }

    def plot_metric(self, metric, agents=None, compare=False, filt_width=None):
        """
        Plot a specific metric for one or multiple agents, with optional moving average filtering.

            metric:     The column name of the metric to plot.
            agents:     List of agent names to plot. If None, all agents are included.
            compare:    Whether to overlay plots for comparison (True) or plot separately (False).
            filt_width: Width of the moving average filter. If None, no filtering is applied.
        """
        if agents is None:
            agents = self.agent_data.keys()

        def apply_filter(data, width):
            """Applies a moving average filter."""
            return data.rolling(window=width, min_periods=1).mean() if width else data

        if compare:
            plt.figure(figsize=(10, 6))
            for agent in agents:
                if agent in self.agent_data:
                    data = self.agent_data[agent][metric]
                    filtered_data = apply_filter(data, filt_width)
                    plt.plot(filtered_data, label=agent)
            plt.title(f"Comparison of {metric} (Filter Width: {filt_width})")
            plt.xlabel("Episode")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            for agent in agents:
                if agent in self.agent_data:
                    data = self.agent_data[agent][metric]
                    filtered_data = apply_filter(data, filt_width)
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
                    plt.show()
