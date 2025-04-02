import time
import numpy as np
import argparse
from collections import deque, Counter
import shutil

import os
import json
import re
import pyautogui

from games.ace_attorney.workers import ace_attorney_worker, perform_move
from tools.utils import str2bool
from collections import Counter

CACHE_DIR = "cache/ace_attorney"

# System prompt remains constant
system_prompt = (
    "You are an expert AI agent specialized in playing Ace Attorney games. Your goal is to solve cases by gathering evidence, cross-examining witnesses, and presenting the correct evidence at the right time to prove your client's innocence. "
)

def main():
    # Delete existing cache directory if it exists and create a new one
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Ace Attorney AI Agent")
    parser.add_argument("--api_provider", type=str, default="openai", help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-11-20", help="LLM model name.")
    parser.add_argument("--modality", type=str, default="vision-text", choices=["text-only", "vision-text"],
                        help="modality used.")
    parser.add_argument("--thinking", type=str, default=True, help="Whether to use deep thinking.")
    parser.add_argument("--episode_name", type=str, default="The First Turnabout", 
                        help="Name of the current episode being played.")
    args = parser.parse_args()

    prev_responses = deque(maxlen=1)

    try:
        while True:
            start_time = time.time()

            # Direct call to ace_attorney_worker
            result = ace_attorney_worker(
                system_prompt,
                args.api_provider,
                args.model_name,
                "\n".join(prev_responses),
                thinking=str2bool(args.thinking),
                modality=args.modality,
                episode_name=args.episode_name
            )
            
            print("Worker finished execution...")
            print(result)

            # Process the result
            if result:
                for move_thought in result:
                    move = move_thought["move"].strip().lower()
                    thought = move_thought["thought"]
                    
                    # Perform the move using the imported function
                    perform_move(move)
                    
                    # Update previous responses
                    latest_response = "step executed:\n" + f"move: {move}, thought: {thought}" + "\n"
                    prev_responses.append(latest_response)

            print("[debug] previous message:")
            print("\n".join(prev_responses))
            elapsed_time = time.time() - start_time
            time.sleep(1)
            print(f"[INFO] Move executed in {elapsed_time:.2f} seconds.")
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()