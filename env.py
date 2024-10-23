import typing 
import numpy as np
import numpy.typing as npt


class Env2048():
    """ Env2048 is a class for playing 2048"""

    def __init__(self, n_rows=4, n_cols=4, max_tile=11, init_state=np.array([])) -> None:
        """ 
            Env2048(n_rows=4, n_cols=4, max_tile=11, init_state=np.array([]))

            Init a 2048 Game environment
            
            Inputs:
                n_rows:         Number of grid rows
                n_cols:         Number of grid columns
                max_tile:       Maximum allowable tile. Game is won upon reaching this tile value. 
                init_state:     Init state. If provided it will act as board starting position.
        """
        self.max_tile = max_tile
        self.min_tile = 2
        self.poss_moves = 4
        self.player_score = 0
        self.player_moves = 0

        rng = np.random.default_rng(seed=100)

        # init state is provided
        if init_state.size != 0:
            self.grid = init_state
            self.nrows = self.grid.shape[0]
            self.ncols = self.grid.shape[1]
        # init grid randomly
        else:
            self.nrows = n_rows
            self.ncols = n_cols
            self.grid = np.zeros(shape=(self.nrows, self.ncols), dtype=int)

            # init two random tiles
            pos1 = (rng.integers(0, self.nrows), rng.integers(0, self.ncols))
            while True:
                pos2 = (rng.integers(0, self.nrows), rng.integers(0, self.ncols))
                if pos2 != pos1:
                    break
            
            self.grid[pos1] = self.min_tile
            self.grid[pos2] = self.min_tile

        
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
        return self.player_score
    

    def execute_action(self, action: int) -> typing.Tuple[int, float, bool]:
        """ 
            execute_action(action)

            Execute action 
            
            Inputs:
                action:     Direction to move (0: U, 1: R, 2: D, 3: L)(int)

            Outputs:
                state:      New state (int)
                reward:     Reward of action (float)
                end:        Game end flag (bool)
        """