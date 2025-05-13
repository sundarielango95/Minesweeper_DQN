import gymnasium as gym
import numpy as np
from gym import spaces
import random
from env_config import ENV_CONFIG, RISK_PROFILES
from termcolor import colored
from gymnasium import Env

class CognitiveMinesweeperEnv(Env):
    def __init__(self):
        super(CognitiveMinesweeperEnv, self).__init__()

        self.grid_size = ENV_CONFIG['grid_size']
        self.quadrant_size = ENV_CONFIG['quadrant_size']
        self.total_tiles = self.grid_size * self.grid_size
        self.max_steps = ENV_CONFIG['max_steps']
        self.remap_interval = ENV_CONFIG['remap_interval']

        self.action_space = spaces.MultiDiscrete([self.total_tiles, 2])  # tile_index, action_type (0=click, 1=flag)
        self.observation_space = spaces.Box(low=0, high=4, shape=(self.total_tiles,), dtype=np.int32)

        self.episode_counter = 0
        self.current_step = 0
        self.moves_log = []

        self.quadrant_profiles = ENV_CONFIG['quadrant_profiles']
        self.quadrant_mapping = []  # Will be a shuffled list of quadrant IDs [0, 1, 2, 3]

        self.state = np.zeros(self.total_tiles, dtype=np.int32)
        self.tile_types = np.zeros(self.total_tiles, dtype=np.int32)  # 1=safe, 2=low-risk, 3=danger

        self._remap_quadrants()
        self.reset()

    def _remap_quadrants(self):
        self.quadrant_mapping = random.sample(range(4), 4)  # Random order of quadrant IDs

    def _assign_tile_types(self):
        self.tile_types = np.zeros(self.total_tiles, dtype=np.int32)
        for q_id in range(4):
            profile = self.quadrant_profiles[self.quadrant_mapping[q_id]]
            danger_pct, low_pct, safe_pct = RISK_PROFILES[profile]
            indices = self._get_quadrant_indices(q_id)
            random.shuffle(indices)
            n = len(indices)
            danger_n = int(n * danger_pct)
            low_n = int(n * low_pct)
            safe_n = n - danger_n - low_n

            for i in range(danger_n):
                self.tile_types[indices[i]] = 3
            for i in range(danger_n, danger_n + low_n):
                self.tile_types[indices[i]] = 2
            for i in range(danger_n + low_n, n):
                self.tile_types[indices[i]] = 1

    def _get_quadrant_indices(self, quadrant_id):
        rows = (0, 5) if quadrant_id in [0, 1] else (5, 10)
        cols = (0, 5) if quadrant_id in [0, 2] else (5, 10)
        return [r * self.grid_size + c for r in range(*rows) for c in range(*cols)]

    def reset(self):
        if self.episode_counter % self.remap_interval == 0:
            self._remap_quadrants()
        self._assign_tile_types()
        self.state = np.zeros(self.total_tiles, dtype=np.int32)
        self.current_step = 0
        self.episode_counter += 1
        self.moves_log = []
        return self.state.copy()

    def step(self, action):
        index, act_type = action
        reward = 0
        done = False

        if self.state[index] != 0:
            return self.state.copy(), -1, False, {}

        if act_type == 0:  # Click
            tile_value = self.tile_types[index]
            if tile_value == 1:
                reward = 10
                self.state[index] = 1
            elif tile_value == 2:
                reward = -10
                self.state[index] = 2
            elif tile_value == 3:
                reward = -50
                self.state[index] = 3
        elif act_type == 1:  # Flag
            if self.tile_types[index] == 3:
                reward = 5
            else:
                reward = -5
            self.state[index] = 4

        self.moves_log.append((self.current_step, index, act_type, reward))

        self.current_step += 1
        done = np.all(self.state != 0) or self.current_step >= self.max_steps

        return self.state.copy(), reward, done, {}

    def render(self, mode='human'):
        grid = self.state.reshape((self.grid_size, self.grid_size))
        symbol_map = {
            0: ('.', 'white'),   # unrevealed
            1: ('S', 'green'),   # safe
            2: ('L', 'yellow'),  # low-risk
            3: ('D', 'red'),     # danger
            4: ('F', 'cyan')     # flagged
        }
        for row in grid:
            print(' '.join(colored(symbol_map[val][0], symbol_map[val][1]) for val in row))
        print("\nMoves Log:")
        for step, idx, act, reward in self.moves_log:
            r, c = divmod(idx, self.grid_size)
            action_str = "Click" if act == 0 else "Flag"
            print(f"Step {step}: ({r}, {c}) -> {action_str} -> Reward: {reward}")

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)