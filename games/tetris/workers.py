import time
import os
import pyautogui
import numpy as np

from tools.utils import encode_image, log_output, extract_python_code
from tools.serving.api_providers import anthropic_completion, openai_completion, gemini_completion

def worker_tetris(
    thread_id,
    offset,
    system_prompt,
    api_provider,
    model_name,
    plan_seconds,
):
    """
    A single Tetris worker that plans moves for 'plan_seconds'.
    1) Sleeps 'offset' seconds before starting (to stagger starts).
    2) Continuously:
        - Captures a screenshot
        - Calls the LLM with a Tetris prompt that includes 'plan_seconds'
        - Extracts the Python code from the LLM output
        - Executes the code with `exec()`
    """
    all_response_time = []

    time.sleep(offset)
    print(f"[Thread {thread_id}] Starting after {offset}s delay... (Plan: {plan_seconds} seconds)")

    tetris_prompt = f"""
Analyze the current Tetris board state and generate PyAutoGUI code to control Tetris 
for the next {plan_seconds} second(s). You can move left/right, rotate pieces, soft drop, 
hard drop, or hold a piece if that's supported. Focus on clearing lines and avoiding 
stacking that would cause a top-out.

### General Tetris Controls (example keybinds):
- left arrow: move piece left
- right arrow: move piece right
- up arrow: rotate piece clockwise
- down arrow: accelerated drop

### Strategies and Caveats:
1. Avoid creating holes and keep the stack flat.
2. If you see a chance to clear lines, use a well-timed drop or rotate.
3. Plan for your next piece as well, but do not top out.
4. The entire sequence of key presses should be feasible within {plan_seconds} second(s).

### Output Format:
- Output ONLY the Python code for PyAutoGUI commands, e.g. `pyautogui.press("left")`.
- Include brief comments for each action.
- Do not print anything else besides these Python commands.
    """

    try:
        while True:
            # Capture the screen
            screen_width, screen_height = pyautogui.size()
            region = (0, 0, screen_width, screen_height)
            screenshot = pyautogui.screenshot(region=region)

            # Create a unique folder for this thread's cache
            thread_folder = f"cache/tetris/thread_{thread_id}"
            os.makedirs(thread_folder, exist_ok=True)

            screenshot_path = os.path.join(thread_folder, "screenshot.png")
            screenshot.save(screenshot_path)

            # Encode the screenshot
            base64_image = encode_image(screenshot_path)

            start_time = time.time()
            if api_provider == "anthropic":
                generated_code_str = anthropic_completion(system_prompt, model_name, base64_image, tetris_prompt)
            elif api_provider == "openai":
                generated_code_str = openai_completion(system_prompt, model_name, base64_image, tetris_prompt)
            elif api_provider == "gemini":
                generated_code_str = gemini_completion(system_prompt, model_name, base64_image, tetris_prompt)
            else:
                raise NotImplementedError(f"API provider: {api_provider} is not supported.")

            end_time = time.time()
            latency = end_time - start_time
            all_response_time.append(latency)

            print(f"[Thread {thread_id}] Request latency: {latency:.2f}s")
            avg_latency = np.mean(all_response_time)
            print(f"[Thread {thread_id}] Latencies: {all_response_time}")
            print(f"[Thread {thread_id}] Average latency: {avg_latency:.2f}s\n")

            print(f"[Thread {thread_id}] --- API output ---\n{generated_code_str}\n")

            # Extract Python code for execution
            clean_code = extract_python_code(generated_code_str)
            log_output(thread_id, f"[Thread {thread_id}] Python code to be executed:\n{clean_code}\n")
            print(f"[Thread {thread_id}] Python code to be executed:\n{clean_code}\n")

            try:
                exec(clean_code)
            except Exception as e:
                print(f"[Thread {thread_id}] Error executing code: {e}")

    except KeyboardInterrupt:
        print(f"[Thread {thread_id}] Interrupted by user. Exiting...")
