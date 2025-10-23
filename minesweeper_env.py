import numpy as np
import gymnasium as gym
class MinesweeperEnv(gym.Env):
    #Custom Minesweeper Environment for Reinforcement Learning.

    def __init__(self,gridheight,gridwidth,mine_multiple):#mine multiple is the fraction of total cells that are mines, ex minemultiple=6
        self.gridwidth=gridwidth
        self.gridheight=gridheight
        self.mine_multiple=mine_multiple
        self.action_space = gym.spaces.MultiDiscrete([self.gridwidth, self.gridheight])
        self.observation_space = gym.spaces.Box(low=-1, high=8, shape=(self.gridheight, self.gridwidth), dtype=np.int16)
        #0 is not revealed, 0-8 is possible with the adjacent mines
        self.state = np.full((self.gridheight, self.gridwidth), -1, dtype=np.int16)
        self.mines = set()

    def reset(self):
        #Reset the environment to the initial state.
        self.state = np.full((self.gridheight, self.gridwidth), -1, dtype=np.int16)
        self.mines = set()
        self.revealed = set()
        reward=0
        #Randomly place mines
        num_mines = (self.gridwidth * self.gridheight) // self.mine_multiple
        while len(self.mines) < num_mines:
            mine = (np.random.randint(0, self.gridwidth), np.random.randint(0, self.gridheight))
            self.mines.add(mine)
        return self.state, {}
    def mine_count_check(self,x,y):
        count=0
        for i in range(-1,2):
            for j in range(-1,2):
                if (x+i,y+j) in self.mines:
                    count+=1
        return count
    def step(self, action):
        self.x=action["x"]
        self.y=action["y"]
        terminated=False
        reward=0
        if self.revealed.__contains__((self.x,self.y)):
            reward=-0.05  #penalty for revealing an already revealed cell
            return self.state, reward, self.done, False, {}
        elif (self.x, self.y) in self.mines:
            terminated=True
            reward = -1  # Hit a mine
        else:
            self.revealed.add((self.x, self.y))
            reward = 0.05  # Safe cell revealed
            self.state[self.y,self.x]=self.mine_count_check(self.x,self.y)
        if len(self.revealed) == (self.gridwidth * self.gridheight) - len(self.mines) and not terminated and (self.x,self.y) not in self.mines:
            reward = 1  # All safe cells revealed
            terminated = True
            self.state[self.y,self.x]=self.mine_count_check(self.x,self.y)
        self.done=terminated
        return self.state, reward, self.done, False, {}

    def render(self, mode='human'):
        display_grid = np.full((self.gridheight, self.gridwidth), '.', dtype=str)
        for (r, c) in self.revealed:
            display_grid[r, c] = str(self.state[r,c])  # Show revealed count
        for (r, c) in self.mines:
            if (r, c) in self.revealed:
                display_grid[r,c] = '*'  # Mine revealed
        print("\n".join(" ".join(row) for row in display_grid))