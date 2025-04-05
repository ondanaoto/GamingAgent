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

from games.ace_attorney.workers import (
    ace_attorney_worker, 
    perform_move, 
    ace_evidence_worker, 
    short_term_memory_worker,
    vision_only_reasoning_worker,
    long_term_memory_worker,
    memory_retrieval_worker,
    vision_only_ace_attorney_worker
)
from tools.utils import str2bool, encode_image, log_output, get_annotate_img, capture_game_window, log_game_event
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
    parser.add_argument("--modality", type=str, default="vision-text", 
                       choices=["text-only", "vision-text", "vision-only"],
                       help="modality used.")
    parser.add_argument("--thinking", type=str, default="False", help="Whether to use deep thinking.")
    parser.add_argument("--episode_name", type=str, default="The_First_Turnabout", 
                       help="Name of the current episode being played.")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of parallel threads to launch.")
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
    decision_state = None

    try:
        while True:
            start_time = time.time()

            # Choose the appropriate worker based on modality
            if args.modality == "vision-only":
                worker_func = vision_only_ace_attorney_worker
            else:
                worker_func = ace_attorney_worker

            # Self-consistency launch with 1-second interval between threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                futures = []
                for i in range(args.num_threads):
                    futures.append(
                        executor.submit(
                            worker_func,
                            system_prompt,
                            args.api_provider,
                            args.model_name,
                            prev_response,
                            thinking=thinking_bool,
                            modality=args.modality,
                            episode_name=args.episode_name,
                            decision_state=decision_state
                        )
                    )
                    if i < args.num_threads - 1:  # Don't sleep after the last thread
                        time.sleep(2)  # Add 1-second interval between launching threads
                
                # Wait until all threads finish
                concurrent.futures.wait(futures)
                results = [f.result() for f in futures]
            
            print("\n" + "="*70)
            print("=== Analysis Results ===")
            print("="*70)
            for i, result in enumerate(results, 1):
                if result and "move" in result and "thought" in result and "game_state" in result:
                    print(f"\nThread {i} Analysis:")
                    print(f"├── Game State: {result['game_state']}")
                    print(f"├── Move: {result['move'].strip().lower()}")
                    print(f"├── Thought Process:")
                    print(f"│   ├── Primary Reasoning: {result['thought']}")
                    if "dialog" in result and result["dialog"]:
                        print(f"│   ├── Dialog Context: {result['dialog']['name']}: {result['dialog']['text']}")
                    if "evidence" in result and result["evidence"]:
                        print(f"│   ├── Evidence Context: {result['evidence']['name']}: {result['evidence']['description']}")
                    if "scene" in result and result["scene"]:
                        print(f"│   └── Scene Context: {result['scene'][:200]}...")
                else:
                    print(f"\nThread {i}: Invalid result")
            print("\n" + "="*70)

            # Collect all moves and thoughts from the results
            moves = []
            thoughts = []
            game_states = []
            dialogs = []
            evidences = []
            scenes = []
            
            for result in results:
                if result and "move" in result and "thought" in result and "game_state" in result:
                    moves.append(result["move"].strip().lower())
                    thoughts.append(result["thought"])
                    game_states.append(result["game_state"])
                    dialogs.append(result.get("dialog", {}))
                    evidences.append(result.get("evidence", {}))
                    scenes.append(result.get("scene", ""))

            if not moves:
                print("[WARNING] No valid moves found in results")
                continue

            # Print vote counts with reasoning
            move_counts = Counter(moves)
            print("\n=== Move Analysis ===")
            for move, count in move_counts.most_common():
                print(f"├── Move: {move}")
                print(f"│   ├── Votes: {count}")
                # Find all thoughts associated with this move
                move_indices = [i for i, m in enumerate(moves) if m == move]
                print(f"│   ├── Supporting Thoughts:")
                for idx in move_indices:
                    print(f"│   │   ├── Thought: {thoughts[idx]}")
                    if dialogs[idx]:
                        print(f"│   │   ├── Dialog: {dialogs[idx]['name']}: {dialogs[idx]['text']}")
                    if evidences[idx]:
                        print(f"│   │   ├── Evidence: {evidences[idx]['name']}: {evidences[idx]['description']}")
                    print(f"│   │   └── Scene: {scenes[idx][:150]}...")
            print("└──" + "─"*66)

            # Perform majority vote on moves
            chosen_move = majority_vote_move(moves)
            chosen_idx = moves.index(chosen_move)
            chosen_thought = thoughts[chosen_idx]
            chosen_game_state = game_states[chosen_idx]
            chosen_dialog = dialogs[chosen_idx]
            chosen_evidence = evidences[chosen_idx]
            chosen_scene = scenes[chosen_idx]

            print("\n=== Final Decision ===")
            print(f"├── Game State: {chosen_game_state}")
            print(f"├── Chosen Move: {chosen_move}")
            print(f"├── Decision Reasoning:")
            print(f"│   ├── Primary Thought: {chosen_thought}")
            if chosen_dialog:
                print(f"│   ├── Dialog Context: {chosen_dialog['name']}: {chosen_dialog['text']}")
            if chosen_evidence:
                print(f"│   ├── Evidence Context: {chosen_evidence['name']}: {chosen_evidence['description']}")
            print(f"│   └── Scene Context: {chosen_scene[:200]}...")
            print(f"└── Execution Status: Pending")
            print("="*70 + "\n")
            
            # Log the final decision
            log_game_event(f"Final Decision - State: {chosen_game_state}, Move: {chosen_move}, Thought: {chosen_thought}, Dialog: {chosen_dialog}, Evidence: {chosen_evidence}, Scene: {chosen_scene[:150]}...")

            # Perform the chosen move
            perform_move(chosen_move)
            
            # Update previous response with game state, move, thought and scene
            prev_response = f"game_state: {chosen_game_state}\nmove: {chosen_move}\nthought: {chosen_thought}\nscene: {chosen_scene}"

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

            if chosen_move == "z" and decision_state and decision_state.get("has_options"):
                decision_state = None  # 👈 RESET after confirming choice
            else:
                # Keep the state if returned by worker
                for result in results:
                    if "decision_state" in result:
                        decision_state = result["decision_state"]


            elapsed_time = time.time() - start_time
            time.sleep(4)
            print(f"[INFO] Move executed in {elapsed_time:.2f} seconds\n")
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()