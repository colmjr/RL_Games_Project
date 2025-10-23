import numpy as np
import gymnasium as gym
class MinesweeperEnv(gym.Env):
    #Custom Minesweeper Environment for Reinforcement Learning.

    def __init__(self,gridheight,gridwidth,mine_multiple):#mine multiple is the fraction of total cells that are mines
        self.gridwidth=gridwidth
        self.gridheight=gridheight
        self.mine_multiple=mine_multiple
        self.action_space = gym.spaces.Discrete(self.gridwidth * self.gridheight)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.gridheight, self.gridwidth), dtype=np.int16)
        self.state = np.zeros((self.gridheight, self.gridwidth), dtype=np.int16)
        self.mines = set()

    def reset(self):
        #Reset the environment to the initial state.
        self.state = np.zeros((self.gridheight, self.gridwidth), dtype=np.int16)
        self.mines = set()
        self.revealed = set()
        self.done = False
        #Randomly place mines
        num_mines = (self.gridwidth * self.gridheight) // self.mine_multiple
        while len(self.mines) < num_mines:
            mine = (np.random.randint(0, self.gridheight), np.random.randint(0, self.gridwidth))
            self.mines.add(mine)
        return self.state, {}

    def step(self, action):
        self.x=action["x"]//self.gridwidth
        self.y=action["y"]//self.gridheight
        if (self.x, self.y) in self.mines:
            terminated=True
            reward = -1  # Hit a mine
        else:
            self.revealed.add((self.x, self.y))
            reward = 0.05  # Safe cell revealed
        if len(self.revealed) == (self.gridwidth * self.gridheight) - len(self.mines):
            terminated=True
            reward = 1  # All safe cells revealed
        return self.state, reward, self.done, False, {}

    def render(self, mode='human'):
        """Render the current state of the environment."""
        display_grid = np.full((self.gridheight, self.gridwidth), '.', dtype=str)
        for (r, c) in self.revealed:
            display_grid[r, c] = '0'  # Placeholder for revealed cells
        for (r, c) in self.mines:
            if (r, c) in self.revealed:
                display_grid[r,c] = '*'  # Mine revealed
        print("\n".join(" ".join(row) for row in display_grid))