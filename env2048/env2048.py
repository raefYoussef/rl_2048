from typing import Optional
from typing import Tuple, List
import numpy as np
import numpy.typing as npt
import pandas as pd
from prettytable import PrettyTable
import os
import time


class Env2048:
    """Env2048 is a class for playing 2048"""

    def __init__(
        self,
        n_rows=4,
        n_cols=4,
        max_tile=11,
        reward_fn=None,
        init_state=None,
        debug=False,
    ) -> None:
        """
        Env2048(n_rows=4, n_cols=4, max_tile=11, init_state=np.array([]))

        Init a 2048 Game environment

        Inputs:
            n_rows:         Number of grid rows
            n_cols:         Number of grid columns
            max_tile:       Maximum allowable tile. Game is won upon reaching this tile value.
            reward_fn:      Custom reward function. Has to adhere to interface.
            init_state:     A 'grid' from a previous run. If provided it will act as board starting position.
            debug:          Init RNG seed.
        """
        self.max_tile = max_tile
        self.min_tile = 1  # 2^1 = 2
        self.num_actions = 4

        if reward_fn:
            self.reward_fn = reward_fn
        else:
            self.reward_fn = self._default_reward_fn

        if debug:
            self.rng = np.random.default_rng(seed=100)
        else:
            self.rng = np.random.default_rng(seed=None)

        if init_state is None:
            self.nrows = n_rows
            self.ncols = n_cols
        else:
            self.nrows = init_state.shape[0]
            self.ncols = init_state.shape[1]

        # reset game board
        self.reset(init_state=init_state)

    def reset(self, init_state=None) -> None:
        """
        reset(init_state=None)

        Resets Environment and History

        Inputs:
            init_state: Start state (optional)

        Outputs:
            None
        """

        self.grid = np.zeros((self.nrows, self.ncols), dtype=np.int_)

        # init grid randomly
        if init_state is None:
            empty_cells = list(zip(*np.where(self.grid == 0)))
            starting_ind = self.rng.choice(len(empty_cells), 2, replace=False)
            self.grid[empty_cells[starting_ind[0]]] = self.min_tile
            self.grid[empty_cells[starting_ind[1]]] = self.min_tile
        else:  # init state is provided
            self.grid = init_state

        # reset vars
        self.score = 0
        self.end = False
        self.win = False

        # reset game history
        self.history = {
            "state": [self.grid.flatten()],
            "action": [],
            "score": [],
            "reward": [],
            "end": [],
            "win": [],
        }
        return self.grid

    def get_action_dim(self) -> int:
        """
        get_action_dim()

        Get Number of Actions (i.e. 4, up/down/left/right)

        Outputs:
            dim:   Number of Possible Actions
        """
        return self.num_actions

    def get_state_dim(self) -> int:
        """
        get_state_dim()

        Get State Size (grid size, e.g. 16 for regular board)

        Outputs:
            dim:   Number of Board size
        """
        dim = self.nrows * self.ncols
        return dim

    def get_grid_dim(self) -> List[int]:
        """
        get_grid_dim()

        Get State Size (grid size, e.g. 16 for regular board)

        Outputs:
            dim:   Number of Board size
        """
        return [self.nrows, self.ncols]

    def get_state(self) -> npt.NDArray[np.int_]:
        """
        get_state()

        Get State (i.e. grid)

        Outputs:
            Grid:   Current grid
        """
        return self.grid

    def get_score(self) -> int:
        """
        get_score()

        Get Score

        Outputs:
            Score:   Current score
        """
        return self.score

    def get_max_tile(self) -> int:
        """
        get_max_tile()

        Get Maximum Tile

        Outputs:
            tile:   Max tile
        """
        return np.max(self.grid)

    def step(
        self, action: int
    ) -> Tuple[npt.NDArray[np.int_], float, float, bool, bool]:
        """
        step(action)

        Execute move/action

        Inputs:
            action:     Direction to move (0: L, 1: U, 2: R, 3: D)(int)

        Outputs:
            state:      New state (int)
            reward:     Reward of action (float)
            score:      Current score (float)
            end:        Game end flag (bool)
            win:        Game ended in a win (bool)
        """

        # As long game is not over
        if not self.end:
            # execute move
            if action >= 0 and action < self.num_actions:
                rot_grid = np.rot90(self.grid, action)
                shift_grid, tot_merged = self._shift_left(rot_grid)
                new_grid = np.rot90(shift_grid, -action)
            else:  # invalid move
                raise ValueError(f"Invalid move, Action range is 0-{self.poss_moves-1}")

            # check end of game
            end, win = self._check_end(new_grid)

            # update vars
            self.score += tot_merged
            self.end = end
            self.win = win

            # calculate reward
            reward = self.reward_fn(
                self.grid, new_grid, self.score, tot_merged, end, win
            )

            # add new tile
            if (not end) and (not np.array_equal(self.grid, new_grid)):
                new_grid = self._add_tile(new_grid)

            # update vars
            self.grid = new_grid

            # update history
            self.history["state"].append(self.grid.flatten())
            self.history["action"].append(action)
            self.history["score"].append(self.score)
            self.history["reward"].append(reward)
            self.history["end"].append(end)
            self.history["win"].append(win)

            # return outputs
            ret_vals = (
                self.grid,
                self.history["reward"][-1],
                self.history["score"][-1],
                self.history["end"][-1],
                self.history["win"][-1],
            )

        # game is over
        else:
            # no more rewards
            ret_vals = (
                self.grid,
                0,
                self.history["score"][-1],
                self.history["end"][-1],
                self.history["win"][-1],
            )

        return ret_vals

    def log_history(self, filename):
        """
        log_history(filename)

        Logs game history to a csv file

        Inputs:
            filename:   CSV file path
        """
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.history.items()]))
        df.to_csv(filename)

    def _shift_left(
        self, grid: npt.NDArray[np.int_]
    ) -> Tuple[npt.NDArray[np.int_], int]:
        """
        _shift_left(grid)

        Execute Left Move on a Grid (used on rotated grid for other moves)

        Inputs:
            grid:       Current grid

        Outputs:
            new_grid:   New grid
            tot_merged: Sum of created new tiles
        """
        new_grid = np.zeros_like(grid)
        grid_rows = new_grid.shape[0]
        tot_merged = 0

        for row_idx in range(grid_rows):
            row = grid[row_idx]

            # move non-zero values to the left
            non_zero = row[row != 0]
            merged = []
            skip = False

            # merge tiles
            for entry_idx in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                if (entry_idx < len(non_zero) - 1) and (
                    non_zero[entry_idx] == non_zero[entry_idx + 1]
                ):
                    merged.append(non_zero[entry_idx] + 1)
                    tot_merged += merged[-1]
                    skip = True
                else:
                    merged.append(non_zero[entry_idx])

            # fill the row in the new grid
            new_grid[row_idx, : len(merged)] = merged

        return (new_grid, tot_merged)

    def _check_end(self, grid: npt.NDArray[np.int_]) -> Tuple[bool, bool]:
        """
        _check_end(grid)

        Check if game is over

        Inputs:
            grid:   Current grid

        Outputs:
            end:    T: game over, F: there are possible moves
            win:    T: win, F: loss (valid when end is true)
        """

        # reached win tile
        if np.any(grid >= self.max_tile):
            # game is over, player won
            return (True, True)
        # empty board
        elif np.any(grid == 0):
            # there's an empty space, game is not over
            return (False, False)
        # full board
        else:
            # check for possible move horizontally
            for row in grid:
                for idx in range(len(row) - 1):
                    # there's a possible move, game is not over
                    if row[idx] == row[idx + 1]:
                        return (False, False)

            # check for possible move vertically
            for col in grid.T:
                for idx in range(len(col) - 1):
                    # there's a possible move, game is not over
                    if col[idx] == col[idx + 1]:
                        return (False, False)

            # no possible moves left, game is over, player lost
            return (True, False)

    def _add_tile(self, grid: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """
        _add_tile(grid)

        Add a tile (2 or 4) randomly to grid

        Inputs:
            grid:       Current grid

        Outputs:
            new_grid:   New grid with added tile
        """
        # find all empty cells (value 0)
        empty_cells = list(zip(*np.where(grid == 0)))

        # no empty cells to add a new tile
        if not empty_cells:
            return grid

        # select one of the empty cells
        row, col = empty_cells[self.rng.choice(len(empty_cells))]

        # add a 2 with 90% prob and 4 with 10% prob
        new_tile = self.min_tile if self.rng.random() < 0.9 else (self.min_tile + 1)
        grid[row, col] = new_tile

        return grid

    # TODO: need to review
    # probaly something we need to investigate and experiment with when we have our model up
    # our model may also influence how we want this to look
    def _default_reward_fn(
        self,
        old_grid: npt.NDArray[np.int_],
        new_grid: npt.NDArray[np.int_],
        score: float,
        tot_merged: float,
        end: bool,
        win: bool,
    ) -> float:
        """
        _default_reward_fn(grid, score, tot_merged, end, win)

        Reward Function

        Inputs:
            old_grid:   grid prior to move
            new_grid:   grid after move
            score:      most recent score
            tot_merged: most recent total of merged tiles
            end:        flag to mark game end
            win:        flag to mark a win/loss (valid when end is true)

        Outputs:
            reward:     reward for move
        """
        reward = 0.0

        # # game over rewards
        # # rationale: winning the game is the ultimate goal,
        # #            it should be rewarded heavier than the greatest merging reward.
        # # concern: the loss penalty might be too large
        # max_merge_reward = (self.get_state_dim() / 2) * (self.max_tile - 1)
        # if end:
        #     if win:
        #         # reward += max_merge_reward
        #         reward += 0
        #     else:
        #         # reward += -max_merge_reward
        #         reward += -np.max(new_grid).item()

        # additional reward is based on total merged tiles
        # rationale: higher total encourages merging
        # concern: this might cause the agent to prioritize high scores over winning
        #          maybe we only want to count the number of tiles merged?
        reward += tot_merged

        # # additional reward is based on reaching a new max tile
        # # rationale: this should encourage the agent to reach new max tiles vs merging lower ones
        # # concern: this might encourage short term merging strategy over long term
        # old_max = np.max(old_grid)
        # new_max = np.max(new_grid)

        # if new_max > old_max:
        #     reward += new_max.item()

        # # Reward for merging tiles
        # new_unique = np.unique(new_grid)
        # old_unique = np.unique(old_grid)
        # new_merged = np.setdiff1d(new_unique, old_unique)
        # max_merged = np.max(new_merged) if new_merged.size > 0 else 0
        # if max_merged > 2:
        #     reward += max_merged

        # # additional reward is based on number of empty states
        # # rationale: higher number of empty states encourages merging
        # # concern: this is potentially captured in score/tot_merged
        # old_empty = np.sum(old_grid == 0).item()
        # new_empty = np.sum(new_grid == 0).item()
        # reward += new_empty - old_empty

        # # additional small negative penalty to discourage non-moves
        # if reward == 0 and np.all(old_grid == new_grid):
        #     reward += -0.1

        # # penalty to discourage excessive moves
        # max_tot_merge = np.sum(np.arange(self.max_tile + 1))
        # if reward == 0:
        #     reward += -(max_tot_merge / 1e3)

        return reward

    def print_state(
        self, flat_state, rows, cols, score, last_action=None, prev_reward=None
    ):
        # expects the 'flat_state' to be valid in regard to the row and col provided
        action = ""
        if last_action is not None:
            action = "Previous Action: "
            if last_action == 0:
                action += "L, "
            elif last_action == 1:
                action += "U, "
            elif last_action == 2:
                action += "R, "
            else:  # 3
                action += "D, "
        reward = f"Reward: {prev_reward}, " if prev_reward else ""
        t = PrettyTable(header=False, padding_width=2)
        print(f"{action}{reward}Score: {score}")
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(flat_state[i * cols + j])
            t.add_row(row, divider=True)
        print(t)

    def print_history(self, history):
        # expects a valid 'history'
        for row_i in range(len(history["state"])):
            # uncomment if you want it to clear between each state (also uncomment the sleep)
            # os.system("cls" if os.name == "nt" else "clear")
            # time.sleep(1)  # sleep 1 second before writing the next state
            # could instead add a pause if we wanted
            flat_state = history["state"][row_i]
            rows = self.nrows
            cols = self.ncols
            if row_i == 0:
                score = 0
                last_action = None
                prev_reward = None
            else:
                score = history["score"][row_i - 1]
                last_action = history["action"][row_i - 1]
                prev_reward = history["reward"][row_i - 1]
            self.print_state(flat_state, rows, cols, score, last_action, prev_reward)
