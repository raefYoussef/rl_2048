import numpy as np


def reward_merging(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
) -> float:
    """
    Encourage merging.
    """

    reward = tot_merged
    return reward


def reward_empty_tiles(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
) -> float:
    """
    Encourages creating empty tiles.
    """

    old_empty = np.sum(old_grid == 0)
    new_empty = np.sum(new_grid == 0)
    reward = new_empty - old_empty  # Reward difference in empty cells
    return reward


def reward_win(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
) -> float:
    """
    Encourage wins. Uses constant reward.
    """

    reward = 0
    if end and win:
        reward = 1
    return reward


def reward_max_tile(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
) -> float:
    """
    Encourage reaching new tiles.
    """

    reward = 0
    old_max = np.max(old_grid)
    new_max = np.max(new_grid)

    if new_max > old_max:
        reward = new_max.item()
    return reward


def reward_new_tiles(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
) -> float:
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


def reward_board_val(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
) -> float:
    """
    Rewards the current board value (sum of tiles)
    """

    reward = np.sum(new_grid)
    return reward


def reward_score(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
) -> float:
    """
    Rewards the current score
    """

    reward = score
    return reward


def reward_directions(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
) -> float:
    """
    Encourage moving in desired directions (Down or Left)
    """

    if action == 3:  # Down
        reward = 1
    elif action == 0:  # Left
        reward = 1
    else:
        reward = -1  # Penalize moving right or up

    return reward


def penalize_loss(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
) -> float:
    """
    Discourage losses. Use constant penalty.
    """
    reward = 0
    if end and not win:
        reward = -1
    return reward


def penalize_nop(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
) -> float:
    """
    Discourage non-moves.
    """
    reward = 0
    if np.all(old_grid == new_grid):
        reward += -1
    return reward


def penalize_moves(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
) -> float:
    """
    Discourage each move that doesn't end the game in a win.
    """
    reward = 0
    if not (end and win):
        reward += -1
    return reward


def penalize_moved_tiles(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
) -> float:
    """
    Penalty for each moved tile
    """
    existing_tiles = old_grid != 0
    moved_tiles = old_grid[existing_tiles] != new_grid[existing_tiles]
    reward = -np.sum(moved_tiles)
    return reward


def reward_practical_strategies(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
):
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


def reward_merging_penalize_moved_tiles(
    old_grid, action, new_grid, tot_merged, score, end, win, num_moves
) -> float:
    """
    Reward merging and penalize unnecessary tile movement
    """
    reward = 0
    reward += reward_merging(
        old_grid, action, new_grid, tot_merged, score, end, win, num_moves
    )
    reward += 0.1 * penalize_moved_tiles(
        old_grid, action, new_grid, tot_merged, score, end, win, num_moves
    )
    return reward
