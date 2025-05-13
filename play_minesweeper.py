# You'll need to have pygame installed: pip install pygame
import pygame
import numpy as np
# Assuming your environment is in cognitive_minesweeper_env.py
from cognitive_minesweeper_env import CognitiveMinesweeperEnv
# Assuming your config is in env_config.py
from env_config import ENV_CONFIG

# --- Pygame Setup ---
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
DARK_GRAY = (120, 120, 120)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255) # For text

# Get grid size from environment config
GRID_SIZE = ENV_CONFIG.get('grid_size', 10) # Default to 10 if not found
TOTAL_TILES = GRID_SIZE * GRID_SIZE

# Window dimensions
TILE_SIZE = 60
GRID_PIXEL_SIZE = GRID_SIZE * TILE_SIZE
INFO_PANEL_HEIGHT = 100 # Area for score, steps, messages
SCREEN_WIDTH = GRID_PIXEL_SIZE
SCREEN_HEIGHT = GRID_PIXEL_SIZE + INFO_PANEL_HEIGHT

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Cognitive Minesweeper")

# --- Font Setup ---
font = pygame.font.Font(None, 36) # Default font, size 36

# --- Game Variables ---
env = CognitiveMinesweeperEnv()
observation = env.reset()
done = False
total_reward = 0

# Map environment state values to colors and symbols for drawing
STATE_COLORS = {
    0: DARK_GRAY, # Unrevealed
    1: GREEN,     # Safe
    2: YELLOW,    # Low-risk
    3: RED,       # Danger
    4: CYAN       # Flagged
}

STATE_SYMBOLS = {
    0: '',  # Unrevealed (blank or covering image)
    1: 'S', # Safe
    2: 'L', # Low-risk
    3: 'D', # Danger
    4: 'F'  # Flagged
}


# --- Helper Functions ---

def get_tile_coords(index):
    """Converts a flat index to (row, col)"""
    row = index // GRID_SIZE
    col = index % GRID_SIZE
    return row, col

def get_tile_index(row, col):
     """Converts (row, col) to a flat index"""
     if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
         return row * GRID_SIZE + col
     return -1 # Invalid index

def get_tile_rect(index):
    """Gets the Pygame Rect object for a tile index"""
    row, col = get_tile_coords(index)
    return pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)

def draw_grid(observation, total_reward, current_step, done):
    """Draws the game grid and info panel"""
    screen.fill(BLACK) # Background

    # Draw tiles
    for i in range(TOTAL_TILES):
        row, col = get_tile_coords(i)
        tile_state = observation[i]
        rect = get_tile_rect(i)

        # Draw tile background
        color = STATE_COLORS.get(tile_state, GRAY)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, BLACK, rect, 1) # Border

        # Draw symbol if revealed or flagged
        symbol = STATE_SYMBOLS.get(tile_state, '')
        if symbol:
            text_surf = font.render(symbol, True, BLACK)
            text_rect = text_surf.get_rect(center=rect.center)
            screen.blit(text_surf, text_rect)

    # Draw info panel
    info_panel_rect = pygame.Rect(0, GRID_PIXEL_SIZE, SCREEN_WIDTH, INFO_PANEL_HEIGHT)
    pygame.draw.rect(screen, GRAY, info_panel_rect)

    score_text = font.render(f"Score: {total_reward}", True, BLUE)
    screen.blit(score_text, (10, GRID_PIXEL_SIZE + 10))

    step_text = font.render(f"Steps: {current_step}/{env.max_steps}", True, BLUE)
    screen.blit(step_text, (SCREEN_WIDTH // 2 - step_text.get_width() // 2, GRID_PIXEL_SIZE + 10)) # Center horizontally

    # Display game over message
    if done:
        message = "Game Over!"
        # You could add Win/Loss condition messages here based on final state/reward
        message_color = RED
        if total_reward > 0 and not np.any(observation == 3): # Example: Win if positive score and no Danger revealed
             message = "You Cleared It!"
             message_color = GREEN

        game_over_text = font.render(message, True, message_color)
        screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, GRID_PIXEL_SIZE + 50))

        # Add restart instruction
        restart_text = font.render("Press R to Restart", True, BLACK)
        screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, GRID_PIXEL_SIZE + 80))


    pygame.display.flip() # Update the full screen

# --- Main Game Loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if not done: # Only process clicks if game is not over
                # Check if the click was on the grid area
                if event.pos[1] < GRID_PIXEL_SIZE:
                    col = event.pos[0] // TILE_SIZE
                    row = event.pos[1] // TILE_SIZE
                    clicked_index = get_tile_index(row, col)

                    if clicked_index != -1:
                        action_type = 0 # 0 for click
                        if event.button == 3: # Right mouse button
                            action_type = 1 # 1 for flag

                        # Take the step in the environment
                        observation, reward, done, info = env.step([clicked_index, action_type])
                        total_reward += reward # Accumulate reward

        if event.type == pygame.KEYDOWN:
            if done and event.key == pygame.K_r: # Press R to restart when game is done
                observation = env.reset()
                done = False
                total_reward = 0
                print("Game Restarted!") # Optional: Print to console

    # Draw the current state
    draw_grid(observation, total_reward, env.current_step, done)

# Quit Pygame
pygame.quit()