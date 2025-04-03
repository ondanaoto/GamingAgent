import time
import numpy as np
import argparse
from collections import deque, Counter
import shutil

import os
import json
import re
import pyautogui

from games.ace_attorney.workers import ace_attorney_worker, perform_move, ace_evidence_worker
from tools.utils import str2bool
from collections import Counter

CACHE_DIR = "cache/ace_attorney"

# System prompt remains constant
system_prompt = (
    "You are an expert AI agent specialized in playing Ace Attorney games. Your goal is to solve cases by gathering evidence, cross-examining witnesses, and presenting the correct evidence at the right time to prove your client's innocence. "
)

def main():
    # Delete existing cache directory if it exists and create a new one
    # if os.path.exists(CACHE_DIR):
    #     shutil.rmtree(CACHE_DIR)
    # os.makedirs(CACHE_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Ace Attorney AI Agent")
    parser.add_argument("--api_provider", type=str, default="anthropic", help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="claude-3-7-sonnet-20250219", help="LLM model name.")
    parser.add_argument("--modality", type=str, default="vision-text", choices=["text-only", "vision-text"],
                        help="modality used.")
    parser.add_argument("--thinking", type=str, default="False", help="Whether to use deep thinking.")
    parser.add_argument("--episode_name", type=str, default="The First Turnabout", 
                        help="Name of the current episode being played.")
    args = parser.parse_args()

    prev_response = ""

    try:
        start_time = time.time()

        thinking_bool = str2bool(args.thinking)

        # evidence_result = ace_evidence_worker(
        #     system_prompt,
        #     args.api_provider,
        #     args.model_name,
        #     prev_response,
        #     thinking=thinking_bool,
        #     modality=args.modality,
        #     episode_name=args.episode_name,
        # )

        while True:
            # Direct call to ace_attorney_worker
            result = ace_attorney_worker(
                system_prompt,
                args.api_provider,
                args.model_name,
                prev_response,
                thinking=thinking_bool,
                modality=args.modality,
                episode_name=args.episode_name,
            )

            # Process the result
            if result:
                # Extract game state, move and thought from the result
                game_state = result["game_state"]
                move = result["move"].strip().lower()
                thought = result["thought"]
                    
                # Print current game state
                print(f"Current Game State: {game_state}")
                
                # Perform the move using the imported function
                perform_move(move)
                
                # Update previous response with game state, move and thought
                prev_response = f"game_state: {game_state}\nmove: {move}\nthought: {thought}"

            print("[debug] previous response:")
            print(prev_response)
            elapsed_time = time.time() - start_time

            time.sleep(3)
            print(f"[INFO] Move executed in {elapsed_time:.2f} seconds.")
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()