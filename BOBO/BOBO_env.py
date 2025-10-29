"""Custom two-player environment for the game of BOBO using PettingZoo."""
from copy import copy
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv

MOVES = {
    0: {"name": "save", "cost": -1, "type": "gain"},
    1: {"name": "defend", "cost": 0, "type": "defense"},
    2: {"name": "storm_defend", "cost": 0, "type": "defense"},
    3: {"name": "all_defend", "cost": 2, "type": "defense"},
    4: {"name": "sword", "cost": 1, "type": "attack"},
    5: {"name": "double_sword", "cost": 2, "type": "attack"},
    6: {"name": "storm", "cost": 3, "type": "attack"},
    7: {"name": "bomb", "cost": 5, "type": "attack"},
    8: {"name": "deflect", "cost": 1, "type": "defense"}
}

WIN_RULES = {
    # save
    (4, 0): "player1", (5, 0): "player1", (6, 0): "player1", (7, 0): "player1",
    (0, 4): "player2", (0, 5): "player2", (0, 6): "player2", (0, 7): "player2",
    # defend
    (6, 1): "player1", (7, 1): "player1",
    (1, 6): "player2", (1, 7): "player2",
    # storm_defend
    (4, 2): "player1", (5, 2): "player1", (7, 2): "player1",
    (2, 4): "player2", (2, 5): "player2", (2, 7): "player2",
    # sword
    (5, 4): "player1", (6, 4): "player1", (7, 4): "player1", (8, 4): "player1",
    (4, 5): "player2", (4, 6): "player2", (4, 7): "player2", (4, 8): "player2",
    # double_sword
    (6, 5): "player1", (7, 5): "player1",
    (5, 6): "player2", (5, 7): "player2",
    # storm
    (7, 6): "player1",
    (6, 7): "player2",
    # deflect
    (5, 8): "player1", (6, 8): "player1", (7, 8): "player1",
    (8, 5): "player2", (8, 6): "player2", (8, 7): "player2",
}


class CustomEnvironment(ParallelEnv):
    """A custom two-player environment in the game of BOBO."""
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, maxsteps):
        """Initialize the CustomEnvironment with maxsteps per episode."""
        self.move1 = None
        self.point1 = None
        self.move2 = None
        self.point2 = None
        self.timestep = None
        self.maxsteps = maxsteps
        self.no_attack_rounds = 0
        self.possible_agents = ["player1", "player2"]
        self.action_spaces = {a: Discrete(len(MOVES)) for a in self.possible_agents}
        self.observation_spaces = {a: MultiDiscrete([len(MOVES), 20, 20]) for a in self.possible_agents}

    def apply_move(self, action, player_points):
        """Applies the move and updates player points."""
        move = MOVES[action]
        if move["cost"] == -1:
            return action, player_points + 1
        elif player_points >= move["cost"]:
            return action, player_points - move["cost"]
        else:
            return -1, player_points

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.move1 = 1
        self.point1 = 0
        self.move2 = 1
        self.point2 = 0
        self.no_attack_rounds = 0
        observations = {
            a: (self.move1, self.point1, self.point2)
            for a in self.agents
        }
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        """Takes a step in the environment based on the actions of both players."""
        self.move1, self.point1 = self.apply_move(actions["player1"], self.point1)
        self.move2, self.point2 = self.apply_move(actions["player2"], self.point2)
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        # Track consecutive rounds without an attack from either player.
        if self.move1 in MOVES and (MOVES[self.move1]["type"] == "sword" or MOVES[self.move1]["type"] == "double_sword" or MOVES[self.move1]["type"] == "storm" or MOVES[self.move1]["type"]=="bomb"):
            self.no_attack_rounds += 1
        winner = WIN_RULES.get((self.move1, self.move2))
        if winner == "player1":
            rewards = {"player1": 1, "player2": -1}
            terminations = {a: True for a in self.agents}
            print(f"P1Win")
            self.point1 = 0
            self.point2 = 0
            self.no_attack_rounds = 0
        elif winner == "player2":
            rewards = {"player1": -1, "player2": 1}
            terminations = {a: True for a in self.agents}
            print(f"P2Win")
            self.point1 = 0
            self.point2 = 0
            self.no_attack_rounds = 0
        elif self.move1 == -1:
            rewards = {"player1": -1, "player2": 1}
            terminations = {a: True for a in self.agents}
            print(f"P1Invalid")
            self.point1 = 0
            self.point2 = 0
            self.no_attack_rounds = 0
        elif self.move2 == -1:
            rewards = {"player1": 0.05, "player2": -0.05}
            terminations = {a: True for a in self.agents}
            print(f"P2Invalid")
            self.point1 = 0
            self.point2 = 0
            self.no_attack_rounds = 0
        elif self.attack_rounds >= 1:
            rewards["player1"]+=self.attack_rounds*0.08 #reward for attacking for player 1(PPO)
            self.no_attack_rounds = 0 
        else:
            truncations = {a: False for a in self.agents}
        self.timestep += 1
        if self.timestep > self.maxsteps:
            truncations = {"player1": True, "player2": True}
        observations = {a: (self.move1, self.move2, self.point1, self.point2) for a in self.agents}
        infos = {a: {"p1_move": self.move1, "p2_move": self.move2,
                     "p1_points": self.point1, "p2_points": self.point2,
                     "winner": winner if any(terminations.values()) else None
                     } for a in self.agents
                 }
        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the current state of the environment."""
        print(f"P1({self.point1} pts): {MOVES[self.move1]['name']}"
              f"P2({self.point2} pts): {MOVES[self.move2]['name']}")

    def observation_space(self, agent):
        """Returns the observation space for the given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Returns the action space for the given agent."""
        return self.action_spaces[agent]
