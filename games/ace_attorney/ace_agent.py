import time
import numpy as np
import concurrent.futures
import argparse
from collections import deque, Counter
import shutil

import os
import json
import re
import pyautogui

from games.ace_attorney.workers import ace_attorney_worker, perform_move, ace_evidence_worker, short_term_memory_worker
from tools.utils import str2bool
from collections import Counter

CACHE_DIR = "cache/ace_attorney"

def majority_vote_move(moves_list, prev_move=None):
    """
    Returns the majority-voted move from moves_list.
    If there's a tie for the top count, and if prev_move is among those tied moves,
    prev_move is chosen. Otherwise, pick the first move from the tie.
    """
    if not moves_list:
        return None

    c = Counter(moves_list)
    
    # c.most_common() -> list of (move, count) sorted by count descending, then by move
    counts = c.most_common()
    top_count = counts[0][1]  # highest vote count

    tie_moves = [m for m, cnt in counts if cnt == top_count]

    if len(tie_moves) > 1 and prev_move:
        if prev_move in tie_moves:
            return prev_move
        else:
            return tie_moves[0]
    else:
        return tie_moves[0]

# System prompt remains constant
system_prompt = (
    "You are an expert AI agent specialized in playing Ace Attorney games. Your goal is to solve cases by gathering evidence, cross-examining witnesses, and presenting the correct evidence at the right time to prove your client's innocence. "
)

def main():
    parser = argparse.ArgumentParser(description="Ace Attorney AI Agent")
    parser.add_argument("--api_provider", type=str, default="anthropic", help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="claude-3-7-sonnet-20250219", help="LLM model name.")
    parser.add_argument("--modality", type=str, default="vision-text", choices=["text-only", "vision-text"],
                        help="modality used.")
    parser.add_argument("--thinking", type=str, default="False", help="Whether to use deep thinking.")
    parser.add_argument("--episode_name", type=str, default="The First Turnabout", 
                        help="Name of the current episode being played.")
    parser.add_argument("--num_threads", type=int, default=3, help="Number of parallel threads to launch.")
    args = parser.parse_args()

    prev_response = ""

    # Delete existing cache directory if it exists and create a new one
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    thinking_bool = str2bool(args.thinking)

    print("--------------------------------Start Evidence Worker--------------------------------")
    evidence_result = ace_evidence_worker(
        system_prompt,
        args.api_provider,
        args.model_name,
        prev_response,
        thinking=thinking_bool,
        modality=args.modality,
        episode_name = args.episode_name
    )

    try:
        while True:
            start_time = time.time()


            # Self-consistency launch with 1-second interval between threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                futures = []
                for i in range(args.num_threads):
                    futures.append(
                        executor.submit(
                            ace_attorney_worker,
                            system_prompt,
                            args.api_provider,
                            args.model_name,
                            prev_response,
                            thinking=thinking_bool,
                            modality=args.modality,
                            episode_name=args.episode_name,
                        )
                    )
                    if i < args.num_threads - 1:  # Don't sleep after the last thread
                        time.sleep(2)  # Add 1-second interval between launching threads
                
                # Wait until all threads finish
                concurrent.futures.wait(futures)
                results = [f.result() for f in futures]
            
            print("\n=== Thread Results Summary ===")
            for i, result in enumerate(results, 1):
                if result and "move" in result and "thought" in result and "game_state" in result:
                    print(f"Thread {i}:")
                    print(f"  Move: {result['move'].strip().lower()}")
                    print(f"  Thought: {result['thought']}")
                    print(f"  Game State: {result['game_state']}")
                else:
                    print(f"Thread {i}: Invalid result")
            print("===========================\n")

            # Collect all moves and thoughts from the results
            moves = []
            thoughts = []
            game_states = []
            
            for result in results:
                if result and "move" in result and "thought" in result and "game_state" in result:
                    moves.append(result["move"].strip().lower())
                    thoughts.append(result["thought"])
                    game_states.append(result["game_state"])

            if not moves:
                print("[WARNING] No valid moves found in results")
                continue

            # Print vote counts
            move_counts = Counter(moves)
            print("\n=== Move Votes ===")
            for move, count in move_counts.most_common():
                print(f"{move}: {count} votes")
            print("================\n")

            # Perform majority vote on moves
            chosen_move = majority_vote_move(moves)
            
            # Find the thought associated with the chosen move
            chosen_thought = thoughts[moves.index(chosen_move)]
            chosen_game_state = game_states[moves.index(chosen_move)]

            # Print current game state
            print(f"Current Game State: {chosen_game_state}")
            
            # Perform the chosen move
            perform_move(chosen_move)
            
            # Update previous response with game state, move and thought
            prev_response = f"game_state: {chosen_game_state}\nmove: {chosen_move}\nthought: {chosen_thought}"

            # Update short-term memory with the chosen response
            short_term_memory_worker(
                system_prompt,
                args.api_provider,
                args.model_name,
                prev_response,
                thinking=thinking_bool,
                modality=args.modality,
                episode_name=args.episode_name
            )

            print("[debug] previous response:")
            print(prev_response)
            elapsed_time = time.time() - start_time
            time.sleep(3)
            print(f"[INFO] Move executed in {elapsed_time:.2f} seconds.")
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()