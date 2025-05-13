# Cognitive Minesweeper Gymnasium Environment

A custom reinforcement learning environment built with Gymnasium, simulating a Minesweeper-like game designed to explore decision-making under spatially varying risk and uncertainty.Project DescriptionThe Cognitive Minesweeper environment (CognitiveMinesweeperEnv) provides a grid-based task where an agent (or a human player) must make sequential decisions (clicking or flagging tiles) to maximize cumulative reward. Unlike traditional Minesweeper where risk is derived solely from adjacent mines, this environment introduces varying risk profiles across different spatial quadrants of the grid.The environment is designed to be challenging for RL agents learning risk-averse or risk-seeking behaviors based on spatially defined probabilities. It is also being explored as a potential behavioral task for cognitive science research, offering a structured alternative or complement to tasks like the Iowa Gambling Test for studying decision-making under uncertainty in different populations (e.g., neurological conditions like Parkinson's disease).

# Key Features

1. Gymnasium Interface: Implements the standard gymnasium.Env API, making it compatible with popular reinforcement learning libraries.
2. Grid-Based Gameplay: A configurable grid (default 10x10) with tiles having hidden types (Safe, Low-Risk, Danger).
3. Spatially Varying Risk: The grid is divided into quadrants, each assigned a distinct risk profile (probability distribution of tile types). These profiles can remap periodically across episodes.
4. Action Space: Agents can choose any tile and perform one of two actions:Click (Reveal): Uncovers the tile, yielding rewards or penalties based on its type.Flag: Marks a tile as potentially dangerous, yielding rewards or penalties based on whether it was correctly identified as Danger.
5. Reward System: Defined rewards and penalties encourage learning to identify and manage risks (e.g., positive for safe clicks and correct flags, negative for revealing danger or incorrect flags).
6. Configurable: Game parameters like grid size, risk profiles, max steps, and remapping intervals are controlled via an external env_config.py.
7. Text Rendering: Includes a basic text-based render() method for visualization (using termcolor).
8. Potential for Graphical UI: Designed to be compatible with a separate graphical interface layer (e.g., built with Pygame) for human play.

# Motivation

This environment was developed to:Provide a novel benchmark for training RL agents on tasks requiring nuanced risk assessment and strategic decision-making in a spatial context.Serve as a flexible platform for cognitive scientists to study human learning and decision-making under spatially contingent risk, potentially offering insights similar to or distinct from tasks like the Iowa Gambling Test.

# Installation
Clone this repository:git clone https://github.com/sundarielango95/Minesweeper_DQN.git
cd Minesweeper_DQN
Install the required libraries:pip install gymnasium numpy termcolor pygame # pygame is needed if you use the included UI script
Create an env_config.py file in the root directory of the project. This file should define the ENV_CONFIG and RISK_PROFILES dictionaries. See the example structure below.env_config.py Example# env_config.py

ENV_CONFIG = {
    'grid_size': 10,
    'quadrant_size': 5, # Assuming a grid_size of 10 is divided into 2x2 quadrants of 5x5 tiles
    'max_steps': 100,   # Maximum number of actions per episode
    'remap_interval': 5, # Remap quadrant profiles every X episodes
    'quadrant_profiles': ['standard', 'cautious', 'aggressive', 'random'] # List of profile names to use
}

RISK_PROFILES = {
    'standard': (0.20, 0.30, 0.50), # (Danger %, Low-Risk %, Safe %)
    'cautious': (0.10, 0.20, 0.70),
    'aggressive': (0.40, 0.30, 0.30),
    'random': (0.25, 0.25, 0.50)
    # Add more profiles as needed
}
Basic Usage (for RL Agent)

import gymnasium as gym
from cognitive_minesweeper_env import CognitiveMinesweeperEnv
from env_config import ENV_CONFIG # Make sure this file exists

# Instantiate the environment
env = CognitiveMinesweeperEnv() # Or register with gym.make if preferred

# Start an episode
observation = env.reset()

total_reward = 0

done = False

info = {} # Gymnasium reset might return info

# Run an episode
while not done:
    # Example: Choose a random action (replace with your agent's logic)
    action = env.action_space.sample()

    # Step the environment
    observation, reward, done, info = env.step(action)
    total_reward += reward

    # Optional: Render the environment (text-based)
    # env.render()
    # print(f"Step: {env.current_step}, Action: {action}, Reward: {reward}, Done: {done}, Total Reward: {total_reward}")

print(f"\nEpisode finished with total reward: {total_reward}")

# Close the environment (important for resources if using rendering, etc.)
env.close()

# Human Playable Game (Pygame UI)

A separate script (play_minesweeper.py in this repo) demonstrates how to wrap this environment in a graphical Pygame UI, allowing human users to play the game. Refer to that script for details on running the human-playable version.

# Environment Details
1. Observation Space: A Box(low=0, high=4, shape=(grid_size * grid_size,)) numpy array representing the state of each tile from the player's perspective:0: Unrevealed1: Safe (Revealed)2: Low-Risk (Revealed)3: Danger (Revealed)4: Flagged
2. Action Space: A MultiDiscrete([grid_size * grid_size, 2]) representing [tile_index, action_type].tile_index: The flat index of the chosen tile (0 to total_tiles - 1).action_type: 0 for Click, 1 for Flag.
3. Reward Function:Clicking Safe (1): Positive reward (e.g., +10)Clicking Low-Risk (2): Negative penalty (e.g., -10)Clicking Danger (3): Large negative penalty (e.g., -50) - typically ends the game or episodeFlagging Danger (3): Small positive reward (e.g., +5)Flagging Safe or Low-Risk (1 or 2): Negative penalty (e.g., -5)
4. Attempting action on an already revealed/flagged tile: Small penalty (e.g., -1) - does not consume a step or change tile state.
5. Episode Termination: An episode ends when:A Danger tile is clicked.The maximum number of steps is reached.(Optional future) All non-Danger tiles are successfully revealed.

# Potential Future Work

1. Refine reward function and tile probabilities based on empirical testing.
2. Implement different information conditions (e.g., classic Minesweeper neighbor counts).Develop more sophisticated graphical UI features (images, animations).
3. Integrate logging for cognitive experiments (response times, click sequences).
4. Benchmark various RL algorithms on the environment.Conduct studies to validate the environment as a cognitive assessment tool.
