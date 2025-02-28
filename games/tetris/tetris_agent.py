import time
import numpy as np
import concurrent.futures
import argparse

from games.tetris.workers import worker_tetris

system_prompt = (
    "You are an expert AI agent specialized in playing Tetris gameplay, search for and execute optimal moves given each game state. Prioritize line clearing over speed."
)

def main():
    """
    Spawns a number of short-term and/or long-term Tetris workers based on user-defined parameters.
    Each worker will analyze the Tetris board and choose moves accordingly.
    """
    parser = argparse.ArgumentParser(
        description="Tetris gameplay agent with configurable concurrent workers."
    )
    parser.add_argument("--api_provider", type=str, default="anthropic",
                        help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="claude-3-5-sonnet-20241022",
                        help="Model name.")
    parser.add_argument("--concurrency_interval", type=float, default=1,
                        help="Interval in seconds between starting workers.")
    parser.add_argument("--api_response_latency_estimate", type=float, default=4.0,
                        help="Estimated API response latency in seconds.")
    parser.add_argument("--short_control_time", type=float, default=1,
                        help="Short worker control time.")
    parser.add_argument("--long_control_time", type=float, default=2,
                        help="Long worker control time.")
    parser.add_argument("--policy", type=str, default="alternate", 
                        choices=["mixed", "alternate", "long", "short"],
                        help="Worker policy: 'long' or 'short'. "
                             "In 'long' or 'short' modes only those workers are enabled, "
                             "or use 'mixed'/'alternate' to combine them.")

    args = parser.parse_args()

    # Calculate how many threads to spawn
    num_threads = int(args.api_response_latency_estimate / args.concurrency_interval)
    
    # Create an offset list
    offsets = [i * args.concurrency_interval for i in range(num_threads)]

    print(f"Starting with {num_threads} threads using policy '{args.policy}'...")
    print(f"API Provider: {args.api_provider}, Model Name: {args.model_name}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            if args.policy == "mixed":
                # Mixed: always spin up one long worker and one short worker in each iteration
                if i % 2 == 0:
                    executor.submit(
                        worker_tetris, i, offsets[i], system_prompt,
                        args.api_provider, args.model_name
                    )
                executor.submit(
                    worker_tetris, i, offsets[i], system_prompt,
                    args.api_provider, args.model_name
                )
            elif args.policy == "alternate":
                # Alternate: even threads -> long worker, odd threads -> short worker
                if i % 2 == 0:
                    executor.submit(
                        worker_tetris, i, offsets[i], system_prompt,
                        args.api_provider, args.model_name
                    )
                else:
                    executor.submit(
                        worker_tetris, i, offsets[i], system_prompt,
                        args.api_provider, args.model_name
                    )
            elif args.policy == "long":
                executor.submit(
                    worker_tetris, i, offsets[i], system_prompt,
                    args.api_provider, args.model_name
                )
            elif args.policy == "short":
                executor.submit(
                    worker_tetris, i, offsets[i], system_prompt,
                    args.api_provider, args.model_name
                )

        try:
            # Keep the main thread alive so the workers can run
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\nMain thread interrupted. Exiting all threads...")

if __name__ == "__main__":
    main()