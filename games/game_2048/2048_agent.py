import time
import os
import pyautogui
import argparse
import numpy as np
from tools.utils import encode_image, log_output
from tools.serving.api_providers import anthropic_completion, openai_completion, gemini_completion
import subprocess
import multiprocessing
# System prompt for LLM
system_prompt = (  
    "You are an expert AI agent specialized in playing the 2048 game with advanced strategic reasoning. "  
    "Your primary goal is to achieve the highest possible tile value by strategically merging tiles. "  
    "Prioritize keeping the highest-value tile in a corner while maintaining an open board for flexibility. "  
    "Avoid moves that result in an early game over by ensuring sufficient space for future moves. "  
      
    "### 2048 Game Rules ### "  
    "1. The game is played on a 4×4 grid, where numbered tiles slide in one of four directions: 'up', 'down', 'left', or 'right'. "  
    "2. Tiles with the same number that collide in the chosen direction will merge into a new tile with double the value. "  
    "3. A move is only valid if at least one tile can slide or merge; otherwise, the move is ignored. "  
    "4. After every move, a new tile (usually 2 or 4) appears at a random empty spot on the board. "  
    "5. The game ends when no valid moves are possible (i.e., the board is full, and no merges can be made). "  
      
    "### Strategy Guidelines ### "  
    "1. Always aim to keep the highest tile in one of the corners (preferably bottom-left or bottom-right). "  
    "2. Prioritize moves that merge tiles to create higher values while preserving board flexibility. "  
    "3. Avoid disrupting tile alignment unless necessary to create merges. "  
    "4. Do not make moves that trap high-value tiles or severely limit future options. "  
      
    "### Decision Output ### "  
    "Analyze the game state from the provided image and determine the optimal move. "  
    "Output only a single word: 'up', 'right', 'left', or 'down'. "  
    "Do not provide explanations—only return the move decision."  
)  

# WIDTH, HEIGHT = 800, 700
# WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))

def capture_screenshot():
    """
    Captures the full screen, saves it, and returns the image path.
    """
    os.makedirs("cache", exist_ok=True)
    screenshot_path = "cache/2048_screenshot.png"

    # Capture full screen
    screen_width, screen_height = pyautogui.size()
    
    region = (0, 0, screen_width, screen_height)
    screenshot = pyautogui.screenshot(region=region)
    screenshot.save(screenshot_path)

    return screenshot_path

def get_best_move(system_prompt, api_provider, model_name):
    """
    Takes a screenshot, sends it to the LLM, and gets the best move.
    """
    screenshot_path = capture_screenshot()
    base64_image = encode_image(screenshot_path)

    move_prompt = (
        "Analyze the 2048 game state from the image and return the best move: 'up', 'right', 'left', or 'down'.\n"
        "Prioritize merging high-value tiles and keeping the board open.\n"
        "Output ONLY the move as a single word."
    )

    start_time = time.time()

    if api_provider == "anthropic":
        move = anthropic_completion(system_prompt, model_name, base64_image, move_prompt)
    elif api_provider == "openai":
        move = openai_completion(system_prompt, model_name, base64_image, move_prompt)
    elif api_provider == "gemini":
        move = gemini_completion(system_prompt, model_name, base64_image, move_prompt)
    else:
        raise NotImplementedError(f"API provider '{api_provider}' is not supported.")

    latency = time.time() - start_time
    print(f"[INFO] LLM Response Latency: {latency:.2f}s")

    return move.strip().lower()

def main():
    """
    Runs a single AI worker for 2048 in a loop without concurrency.
    """
    parser = argparse.ArgumentParser(description="2048 LLM AI Agent (Single Worker)")
    parser.add_argument("--api_provider", type=str, default="openai",
                        help="API provider to use (anthropic, openai, gemini).")
    parser.add_argument("--model_name", type=str, default="gpt-4-turbo",
                        help="Model name.")
    parser.add_argument("--loop_interval", type=float, default=0.5,
                        help="Time in seconds between moves.")

    args = parser.parse_args()

    print(f"Starting 2048 AI Agent...")
    print(f"API Provider: {args.api_provider}, Model Name: {args.model_name}")

    try:
        while True:
            move = get_best_move(system_prompt, args.api_provider, args.model_name)
            # Map arrow keys to WASD

            if move in ["up", "right", "left", "down"]:

                pyautogui.press(move)
                print(f"Executed move: {move}")
            else:
                print(f"Invalid move received: {move}")

            time.sleep(args.loop_interval)  # Delay before next move

    except KeyboardInterrupt:
        print("\nGame interrupted by user. Exiting...")

if __name__ == "__main__":
    main()
