import typing
import numpy as np
import numpy.typing as npt
import pandas as pd


class Env2048:
    """Env2048 is a class for playing 2048"""

    def __init__(
        self, n_rows=4, n_cols=4, max_tile=2048, init_state=None, debug=False
    ) -> None:
        """
        Env2048(n_rows=4, n_cols=4, max_tile=11, init_state=np.array([]))

        Init a 2048 Game environment

        Inputs:
            n_rows:         Number of grid rows
            n_cols:         Number of grid columns
            max_tile:       Maximum allowable tile. Game is won upon reaching this tile value.
            init_state:     A 'grid' from a previous run. If provided it will act as board starting position.
            debug:          Init RNG seed.
        """
        self.max_tile = max_tile
        self.min_tile = 2
        self.actions = {
            0: [-1, 0],  # U
            1: [1, 0],  # D
            2: [0, -1],  # L
            3: [0, 1],
        }  # R
        self.rotation = {0: 1, 1: 3, 2: 0, 3: 2}
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
        self.grid = np.zeros((n_rows, n_cols), dtype=int)
        self.score = 0
        self.end = False
        self.win = False
        self.history = None

        # reset game board
        self.reset(init_state=init_state)

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

    def reset(self, init_state=None) -> None:
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

        # reset game history # TODO probably don't need score or win here.
        self.history = {
            "state": [self.grid.flatten()],
            "action": [],
            "score": [],
            "reward": [],
            "end": [],
            "win": [],
        }

    # TODO how do we handle moves that cannot be made? (can only move in a way that would actually move tiles)
    # so a move is invalid if we would get the same 'new_grid' after taking that action
    # might just allow them as a possible action and return a reward of -1 or something?
    def move(
        self, action: int
    ) -> typing.Tuple[npt.NDArray[np.int_], float, float, bool, bool]:
        """
        move(action)

        Execute move/action

        Inputs:
            action:     Direction to move (0: U, 1: D, 2: L, 3: R)(int)

        Outputs:
            state:      New state (int)
            reward:     Reward of action (float)
            end:        Game end flag (bool)
        """

        # As long game is not over
        if not self.end:
            # execute move
            if action >= 0 and action < len(self.actions):
                rot_grid = np.rot90(self.grid, self.rotation[action])
                shift_grid, tot_merged = self._shift_left(rot_grid)
                new_grid = np.rot90(shift_grid, -self.rotation[action])
            else:  # invalid move
                raise ValueError(f"Invalid move, Action range is 0-{self.poss_moves-1}")

            # check end of game
            end, win = self._check_end(new_grid)

            # update vars
            self.score += tot_merged
            self.end = end
            self.win = win

            # calculate reward
            reward = self._calc_reward(
                self.grid, new_grid, self.score, tot_merged, end, win
            )

            # add new tile
            if (not end) and (not np.array_equal(self.grid, new_grid)):
                new_grid = self._add_tile(new_grid)

            # update vars
            self.grid = new_grid

            # update history # so 6 lists?
            self.history["state"].append(self.grid.flatten())
            self.history["action"].append(action)
            self.history["score"].append(self.score)
            self.history["reward"].append(reward)
            self.history["end"].append(end)
            self.history["win"].append(win)

            # return outputs
            ret_vals = (
                self.history["state"][-1],
                self.history["reward"][-1],
                self.history["score"][-1],
                self.history["end"][-1],
                self.history["win"][-1],
            )

        # game is over
        else:
            # no more rewards
            ret_vals = (
                self.history["state"][-1],
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
    ) -> typing.Tuple[npt.NDArray[np.int_], int]:
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
                    merged.append(non_zero[entry_idx] * 2)
                    tot_merged += merged[-1]
                    skip = True
                else:
                    merged.append(non_zero[entry_idx])

            # fill the row in the new grid
            new_grid[row_idx, : len(merged)] = merged

        return (new_grid, tot_merged)

    # no real reason to pass in the grid right? (in any of these as it is a member)
    def _check_end(self, grid: npt.NDArray[np.int_]) -> typing.Tuple[bool, bool]:
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
        new_tile = 2 if self.rng.random() < 0.9 else 4
        grid[row, col] = new_tile

        return grid

    # TODO: need to review
    # probaly something we need to investigate and experiment with when we have our model up
    # our model may also influence how we want this to look
    def _calc_reward(
        self,
        old_grid: npt.NDArray[np.int_],
        new_grid: npt.NDArray[np.int_],
        score: float,
        tot_merged: float,
        end: bool,
        win: bool,
    ) -> float:
        """
        _calc_reward(grid, score, tot_merged, end, win)

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

        reward = 0

        # game over rewards
        # rationale: winning the game is the ultimate goal, it should be rewarded heavily.
        #            Moreover, it add/subtracts the score because not all wins/losses are equal.
        # concern: the score might get large and dominate the reward function
        if end:
            if win:
                reward = 0.25 * score + 5000
            else:
                reward = 0.25 * score - 5000

        # additional reward is based on total merged tiles
        # rationale: higher total encourages merging
        # concern: this might cause the agent to prioritize high scores over winning
        reward += tot_merged

        # additional reward is based on reaching a new max tile
        # rationale: this should encourage the agent to reach new max tiles vs merging lower ones
        # concern: this might encourage short term merging strategy over long term
        old_max = np.max(old_grid)
        new_max = np.max(new_grid)

        if new_max > old_max:
            reward += new_max

        # # additional reward is based on number of empty states
        # # rationale: higher number of empty states encourages merging
        # # concern: this is potentially captured in score/tot_merged
        # num_empty = np.sum(new_grid == 0)
        # reward += num_empty

        return reward
