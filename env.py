import typing 
import numpy as np
import numpy.typing as npt
import pandas as pd

class Env2048():
    """ Env2048 is a class for playing 2048"""

    def __init__(self, n_rows=4, n_cols=4, max_tile=2048, init_state=None, debug=False) -> None:
        """ 
            Env2048(n_rows=4, n_cols=4, max_tile=11, init_state=np.array([]))

            Init a 2048 Game environment
            
            Inputs:
                n_rows:         Number of grid rows
                n_cols:         Number of grid columns
                max_tile:       Maximum allowable tile. Game is won upon reaching this tile value. 
                init_state:     Init state. If provided it will act as board starting position.
                debug:          Init RNG seed.
        """
        self.max_tile = max_tile
        self.min_tile = 2
        self.poss_moves = 4
        if debug:
            self.rng = np.random.default_rng(seed=100)
        else:
            self.rng = np.random.default_rng(seed=None)
                
        # init grid (dimensions are constant per env)
        if init_state is None:
            self.nrows = n_rows
            self.ncols = n_cols
        else:
            self.nrows = self.grid.shape[0]
            self.ncols = self.grid.shape[1]
        
        self.grid = None
        self.score = 0
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
            self.grid = np.zeros(shape=(self.nrows, self.ncols), dtype=int)

            # init two random tiles
            pos1 = (self.rng.integers(0, self.nrows), self.rng.integers(0, self.ncols))
            while True:
                pos2 = (self.rng.integers(0, self.nrows), self.rng.integers(0, self.ncols))
                if pos2 != pos1:
                    break
            
            self.grid[pos1] = self.min_tile
            self.grid[pos2] = self.min_tile

        # init state is provided
        else:
            self.grid = init_state

        # reset game history
        self.history = {"state": [self.grid.flatten()], "action": [], "score": [], "reward": [], "end": [], "win": []}


    def move(self, action: int) -> typing.Tuple[npt.NDArray[np.int_], float, float, bool, bool]:
        """ 
            move(action)

            Execute move/action 
            
            Inputs:
                action:     Direction to move (0: L, 1: U, 2: R, 3: D)(int)

            Outputs:
                state:      New state (int)
                reward:     Reward of action (float)
                end:        Game end flag (bool)
        """

        # execute move
        if action >= 0 and action < self.poss_moves:
            rot_grid = np.rot90(self.grid, action)
            shift_grid, tot_merged = self._shift_left(rot_grid)
            new_grid = np.rot90(shift_grid, -action)
        # invalid move
        else:
            raise ValueError(f"Invalid move, Action range is 0-{self.poss_moves-1}")
        
        # check end of game 
        end, win = self._check_end(new_grid)

        # add new tile
        if (not end) and (not np.array_equal(self.grid, new_grid)):
            new_grid = self._add_tile(new_grid)

        # update vars
        self.grid = new_grid
        self.score += tot_merged

        # update history
        self.history["state"].append(self.grid.flatten())
        self.history["action"].append(action)
        self.history["score"].append(self.score)
        self.history["reward"].append(tot_merged)   # TODO: need to review
        self.history["end"].append(end)
        self.history["win"].append(win)

        # return outputs
        ret_vals = (self.history["state"][-1], self.history["reward"][-1], self.history["score"][-1], self.history["end"][-1], self.history["win"][-1])
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


    def _shift_left(self, grid: npt.NDArray[np.int_]) -> typing.Tuple[npt.NDArray[np.int_], int]:
        """ 
            _shift_left(grid)

            Execute Left Move on a Grid
            
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
                if (entry_idx < len(non_zero) - 1) and (non_zero[entry_idx] == non_zero[entry_idx + 1]):
                    merged.append(non_zero[entry_idx] * 2)
                    tot_merged += merged[-1]
                    skip = True
                else:
                    merged.append(non_zero[entry_idx])

            # fill the row in the new grid
            new_grid[row_idx, :len(merged)] = merged
        
        return (new_grid, tot_merged)
    

    def _check_end(self, grid: npt.NDArray[np.int_]) -> typing.Tuple[bool, bool]:
        """ 
            _check_end(grid)

            Check if game is over
            
            Inputs:
                grid:   Current grid

            Outputs:
                end:    T: game over, F: there are possible moves
                win:    T: win, F: loss (check when end is true)
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
                for idx in range(len(row)-1):
                    # there's a possible move, game is not over
                    if row[idx] == row[idx+1]:
                        return (False, False)
            
            # check for possible move vertically
            for col in grid.T:
                for idx in range(len(col)-1):
                    # there's a possible move, game is not over
                    if col[idx] == col[idx+1]:
                        return (False, False)
                    
            # no possible moves left, game is over, player lost
            return (True, False)
                    
            
    def _add_tile(self, grid: npt.NDArray[np.int_]) ->  npt.NDArray[np.int_]:
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
