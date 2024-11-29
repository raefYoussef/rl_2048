import numpy as np


def reward_merging(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Encourage merging.
    """

    reward = tot_merged
    return reward


def reward_empty_tiles(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Reward creating empty tiles.
    """

    old_empty = np.sum(old_grid == 0)
    new_empty = np.sum(new_grid == 0)
    reward = new_empty - old_empty  # Reward difference in empty cells
    return reward


def reward_win(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Encourage wins. Uses constant reward.
    """

    reward = 0
    if end and win:
        reward = 1
    return reward


def reward_max_tile(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Encourage reaching new tiles.
    """

    reward = 0
    old_max = np.max(old_grid)
    new_max = np.max(new_grid)

    if new_max > old_max:
        reward = new_max.item()
    return reward


def reward_new_tiles(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Encourage making new tiles.
    """

    reward = 0
    new_unique = np.unique(new_grid)
    old_unique = np.unique(old_grid)
    new_merged = np.setdiff1d(new_unique, old_unique)
    max_merged = np.max(new_merged) if new_merged.size > 0 else 0

    # reward tiles above 4 since 2 and 4 are automatically added
    if max_merged > 2:
        reward = max_merged
    return reward


def penalize_loss(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Discourage losses. Use constant penalty.
    """
    reward = 0
    if end and not win:
        reward = -1
    return reward


def penalize_non_moves(old_grid, new_grid, score, tot_merged, end, win) -> float:
    """
    Discourage non-moves.
    """
    reward = 0
    if np.all(old_grid == new_grid):
        reward = -1e-2
    return reward


def reward_practical_strategies(old_grid, new_grid, score, tot_merged, end, win):
    # Ensure the largest tile is in a corner
    max_tile = np.max(new_grid)
    corners = [new_grid[0, 0], new_grid[0, -1], new_grid[-1, 0], new_grid[-1, -1]]
    R_corner = 1.0 if max_tile in corners else 0.0

    # Penalize differences between adjacent tiles (Cluster reward)
    R_cluster = 0.0
    for i in range(new_grid.shape[0]):
        for j in range(new_grid.shape[1]):
            if i > 0:
                R_cluster -= abs(
                    new_grid[i, j] - new_grid[i - 1, j]
                )  # Vertical difference
            if j > 0:
                R_cluster -= abs(
                    new_grid[i, j] - new_grid[i, j - 1]
                )  # Horizontal difference

    # Monotonicity reward
    def is_monotonic(array):
        return all(array[i] >= array[i + 1] for i in range(len(array) - 1)) or all(
            array[i] <= array[i + 1] for i in range(len(array) - 1)
        )

    R_monotonic = 0.0
    for row in new_grid:
        if is_monotonic(row):
            R_monotonic += 1
    for col in new_grid.T:
        if is_monotonic(col):
            R_monotonic += 1

    # Weighted sum
    alpha, beta, gamma = 1.0, 0.1, 0.1
    R_total = alpha * R_corner + beta * R_cluster + gamma * R_monotonic

    return R_total
