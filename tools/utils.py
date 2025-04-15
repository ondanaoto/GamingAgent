import time
import os
import pyautogui
import base64
import anthropic
import numpy as np
import concurrent.futures
import re
import cv2
import sys
import platform
from PIL import Image

import numpy as np
import json
import argparse
from pathlib import Path
import time
def encode_image(image_path):
    """
    Read a file from disk and return its contents as a base64-encoded string.
    """
    time.sleep(0.5)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image file at path: {image_path}")

    # Proceed with base64 encoding
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def log_output(thread_id, log_text, game, alias=None, mode="w"):
    """
    Logs output.
    """
    # alias has to be string
    if alias:
        assert isinstance(alias, str), f"Expected {str}, got {type(alias)}"
        thread_folder = f"cache/{game}/thread_{thread_id}/{alias}"
    else:
        thread_folder = f"cache/{game}/thread_{thread_id}"
    os.makedirs(thread_folder, exist_ok=True)
    
    log_path = os.path.join(thread_folder, "output.log")
    with open(log_path, mode, encoding="utf-8") as log_file:
        log_file.write(log_text + "\n\n")

def extract_python_code(content):
    """
    Extracts Python code from the assistant's response.
    - Detects code enclosed in triple backticks (```python ... ```)
    - If no triple backticks are found, returns the raw content.
    """
    match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return content.strip()

def extract_planning_prompt(generated_str): 
    """ Searches for a segment in generated_str of the form:
    ```planning prompt
    {text to be extracted}
    ```

    and returns the text inside. If it doesn't find it, returns an empty string.
    """
    pattern = r"```planning prompt\s*(.*?)\s*```"
    match = re.search(pattern, generated_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_patch_table(generated_str): 
    pattern = r"```game patch table\s*(.*?)\s*```"
    match = re.search(pattern, generated_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_game_table(generated_str): 
    pattern = r"```game table\s*(.*?)\s*```"
    match = re.search(pattern, generated_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def read_log_to_string(log_path):
    """
    Reads the log file and returns its content as a string.
    """
    assert os.path.exists(log_path), "Log file {log_path} does not exist."

    with open(log_path, "r", encoding="utf-8") as file:
        log_content = file.read()
    
    return log_content

def find_iteration_dirs(base_path):
    """
    Returns a list of tuples (iteration_number, iteration_path), sorted by iteration_number.
    Only includes directories matching the pattern iter_#.
    """
    print("Traversing root directory...")
    iteration_dirs = []

    # Identify all threads (subdirectories)
    for item in os.listdir(base_path):
        full_path = os.path.join(base_path, item)
        for iter_dir in os.listdir(full_path):
            iter_match = re.match(r"iter_(\d+)$", iter_dir)  # Extract number from iter_#
            if iter_match:
                print(iter_dir)
                iter_num = int(iter_match.group(1))  # Convert the extracted number to int
                iter_full_path = os.path.join(full_path, iter_dir)
                iteration_dirs.append((iter_num, iter_full_path))
            else:
                continue

    # Sort by iteration number (ascending)
    iteration_dirs.sort(key=lambda x: x[0])

    return iteration_dirs

def build_iteration_content(iteration_dirs, memory_size):
    """
    Given a sorted list of (iter_num, path) for all Tetris iterations,
    select only the latest `memory_size` iterations.
    For each iteration, locate the PNG and LOG file,
    extract base64 image and generated code, and build a single string
    that includes 'generated code for STEP <iter_num>:' blocks.
    Returns a tuple (steps_content, list_image_base64).
    """
    print("building iteration content...")
    total_iterations = len(iteration_dirs)
    # Only look at the last 'memory_size' iterations
    relevant_iterations = iteration_dirs[-memory_size:] if total_iterations > memory_size else iteration_dirs
    steps_content = []
    list_image_base64 = []

    for (iter_num, iter_path) in relevant_iterations:
        png_file = None
        log_file = None

        # Identify .png and .log inside the iter_# directory
        for f in os.listdir(iter_path):
            f_lower = f.lower()
            if f_lower.endswith(".png"):
                png_file = os.path.join(iter_path, f)
            elif f_lower.endswith(".log"):
                log_file = os.path.join(iter_path, f)

        # Encode the image if available
        image_base64 = ""
        if png_file and os.path.isfile(png_file):
            image_base64 = encode_image(png_file)
            # We'll keep track of the *last* iteration's image for submission
            list_image_base64.append(image_base64)

        # Extract generated code from the log if available
        code_snippets = ""
        if log_file and os.path.isfile(log_file):
            with open(log_file, "r", encoding="utf-8") as lf:
                log_content = lf.read()
                code_snippets = extract_python_code(log_content).strip()

        # Build the block for this iteration
        block = f"Generated code for STEP {iter_num}:\n{code_snippets}"
        steps_content.append(block)

    # Join all iteration blocks into one string
    return steps_content, list_image_base64

def preprocess_image(image_path, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0, cache_dir=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image '{image_path}' not found or unreadable.")
        exit()
    
    # Crop image to remove left, right, top, and bottom sides
    height, width = image.shape[:2]
    new_x_start = crop_left
    new_x_end = width - crop_right
    new_y_start = crop_top
    new_y_end = height - crop_bottom
    cropped_image = image[new_y_start:new_y_end, new_x_start:new_x_end]
    
    # Save cropped image
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cropped_debug_path = os.path.join(cache_dir, "cropped_debug.png")
    else:
        cropped_debug_path = "cropped_debug.png"
    cv2.imwrite(cropped_debug_path, cropped_image)
    
    return image, cropped_image, new_x_start, new_y_start

def generate_grid(image, grid_rows, grid_cols):
    height, width = image.shape[:2]
    cell_width = width // grid_cols
    cell_height = height // grid_rows
    
    vertical_lines = [i * cell_width for i in range(grid_cols + 1)]
    horizontal_lines = [i * cell_height for i in range(grid_rows + 1)]
    
    return vertical_lines, horizontal_lines
def annotate_with_grid(image, vertical_lines, horizontal_lines, x_offset, y_offset, alpha=0.5, enable_digit_label = True, thickness = 1, black = False, font_size=0.4):
    """Annotates the image with semi-transparent gray grid cell numbers."""
    grid_annotations = []
    
    # Create a copy of the image to overlay transparent text
    overlay = image.copy()

    for row in range(len(horizontal_lines) - 1):
        for col in range(len(vertical_lines) - 1):
            x = (vertical_lines[col] + vertical_lines[col + 1]) // 2
            y = (horizontal_lines[row] + horizontal_lines[row + 1]) // 2
            cell_id = row * (len(vertical_lines) - 1) + col + 1
            grid_annotations.append({'id': cell_id, 'x': x + x_offset, 'y': y + y_offset})
            
            if enable_digit_label:
                # Draw semi-transparent text on the overlay
                text = str(cell_id)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = font_size
                thickness = thickness
                if black:
                    text_color = (0, 0, 0)
                else:
                    text_color = (255, 255, 255)  # Gray color
            
                cv2.putText(overlay, text, (x - 10, y + 10), font, font_scale, text_color, thickness, cv2.LINE_AA)
            
            # Draw green grid rectangle
            if black:
                cv2.rectangle(image, (vertical_lines[col], horizontal_lines[row]), 
                            (vertical_lines[col + 1], horizontal_lines[row + 1]), (0, 0, 0), thickness)
            else:
                cv2.rectangle(image, (vertical_lines[col], horizontal_lines[row]), 
                            (vertical_lines[col + 1], horizontal_lines[row + 1]), (0, 255, 0), thickness)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image, grid_annotations


def save_grid_annotations(grid_annotations, cache_dir=None):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        output_file = os.path.join(cache_dir, 'grid_annotations.json')
    else:
        output_file = 'grid_annotations.json'
    
    with open(output_file, 'w') as file:
        json.dump(grid_annotations, file, indent=4)
    return output_file

def get_annotate_img(image_path, crop_left=50, crop_right=50, crop_top=50, crop_bottom=50, grid_rows=9, grid_cols=9, output_image='annotated_grid.png', cache_dir=None, enable_digit_label=True, thickness=1, black=False, font_size=0.4):
    original_image, cropped_image, x_offset, y_offset = preprocess_image(image_path, crop_left, crop_right, crop_top, crop_bottom, cache_dir)
    vertical_lines, horizontal_lines = generate_grid(cropped_image, grid_rows, grid_cols)
    annotated_cropped_image, grid_annotations = annotate_with_grid(cropped_image, vertical_lines, horizontal_lines, x_offset, y_offset, enable_digit_label=enable_digit_label, thickness=thickness, black=black, font_size=font_size)
    grid_annotation_path = save_grid_annotations(grid_annotations, cache_dir)
    
    # Place the annotated cropped image back onto the original image
    original_image[y_offset:y_offset + annotated_cropped_image.shape[0], x_offset:x_offset + annotated_cropped_image.shape[1]] = annotated_cropped_image
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        output_image_path = os.path.join(cache_dir, output_image)
        annotated_cropped_image_path = os.path.join(cache_dir, 'annotated_cropped_image.png')
    else:
        output_image_path = output_image
        annotated_cropped_image_path = 'annotated_cropped_image.png'
    
    cv2.imwrite(output_image_path, original_image)
    cv2.imwrite(annotated_cropped_image_path, annotated_cropped_image)

    return output_image_path, grid_annotation_path, annotated_cropped_image_path

def get_annotate_patched_img(image_path, 
        crop_left=50, crop_right=50, crop_top=50, crop_bottom=50, 
        grid_rows=9, grid_cols=9, 
        x_dim=5,          # number of cells (columns) in each patch
        y_dim=5,          # number of cells (rows) in each patch
        output_image='annotated_grid.png', cache_dir=None):
    
    # Generate grid annotations
    original_image, cropped_image, x_offset, y_offset = preprocess_image(
        image_path, crop_left, crop_right, crop_top, crop_bottom, cache_dir
    )
    vertical_lines, horizontal_lines = generate_grid(cropped_image, grid_rows, grid_cols)
    annotated_cropped_image, grid_annotations = annotate_with_grid(
        cropped_image, vertical_lines, horizontal_lines, x_offset, y_offset
    )
    grid_annotation_path = save_grid_annotations(grid_annotations, cache_dir)
    
    # Place the annotated cropped image back onto the original image
    original_image[y_offset:y_offset + annotated_cropped_image.shape[0],
                   x_offset:x_offset + annotated_cropped_image.shape[1]] = annotated_cropped_image
    
    # Set up paths for saving
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        output_image_path = os.path.join(cache_dir, output_image)
        annotated_cropped_image_path = os.path.join(cache_dir, 'annotated_cropped_image.png')
    else:
        output_image_path = output_image
        annotated_cropped_image_path = 'annotated_cropped_image.png'
    
    # Scale images
    scale_factor = 5
    original_image_scaled = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    annotated_cropped_image_scaled = cv2.resize(annotated_cropped_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    # Save the scaled images
    cv2.imwrite(output_image_path, original_image_scaled)
    cv2.imwrite(annotated_cropped_image_path, annotated_cropped_image_scaled)

    # Generate grid patches
    grid_patches = generate_patches_from_cells(
        grid_annotations, grid_rows, grid_cols, x_dim, y_dim
    )

    # Crop, scale, and save each patch image
    annotated_cropped_patch_image_paths = []
    for patch in grid_patches:
        patch_num = patch["patch_number"]

        # Adjust coordinates relative to the cropped image
        x1_cropped = patch["x1"] - x_offset
        y1_cropped = patch["y1"] - y_offset
        x2_cropped = patch["x2"] - x_offset
        y2_cropped = patch["y2"] - y_offset

        # Extract the sub-image from the annotated cropped image
        sub_image = annotated_cropped_image[y1_cropped : y2_cropped+1, x1_cropped : x2_cropped+1]

        # Scale the patch image by the same factor
        sub_image_scaled = cv2.resize(sub_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        patch_filename = f"patch_{patch_num}.png"
        if cache_dir:
            patch_path = os.path.join(cache_dir, patch_filename)
        else:
            patch_path = patch_filename

        # Save the scaled patch image
        cv2.imwrite(patch_path, sub_image_scaled)
        annotated_cropped_patch_image_paths.append(patch_path)

    return output_image_path, grid_annotation_path, annotated_cropped_patch_image_paths


def scale_png_to_512x512(input_path, output_path):
    """
    Scales a PNG image to 512x512 pixels using OpenCV.

    Args:
        input_path (str): Path to the original PNG image.
        output_path (str): Where to save the resized 512x512 image.
    """
    # Read the input image
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read the image at {input_path}")

    # Resize to 512x512
    resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    # Save the result
    cv2.imwrite(output_path, resized_image)
    print(f"Saved scaled image to {output_path}")

def str2bool(value):
    """
    Converts a string to a boolean.
    Accepts: 'true', 'false' (case-insensitive)
    Raises an error for invalid values.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif value.lower() in ('false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def get_platform():
    """
    Detects the current operating system platform.
    Returns: 'windows', 'mac', or 'other'
    """
    system = platform.system().lower()
    if 'windows' in system:
        return 'windows'
    elif 'darwin' in system:
        return 'mac'
    else:
        return 'other'

def find_game_window():
    """
    Find the Phoenix Wright game window using case-insensitive substring matching.
    Returns the window handle if found, None otherwise.
    """
    import win32gui
    
    def window_enum_callback(hwnd, result):
        window_title = win32gui.GetWindowText(hwnd).lower()
        if "phoenix wright" in window_title:
            result.append(hwnd)
            
    result = []
    win32gui.EnumWindows(window_enum_callback, result)
    return result[0] if result else None

def calculate_scale_multiplier(width, height, target_min=768, target_max=1500):
    """
    Calculate the scale multiplier to fit within target dimensions while maintaining aspect ratio.
    """
    # Calculate multipliers that would reach each target
    width_multiplier = target_max / width
    height_multiplier = target_max / height
    min_multiplier = target_min / min(width, height)
    
    # Use the smallest multiplier that satisfies both conditions
    scale = min(width_multiplier, height_multiplier, min_multiplier)
    
    # Ensure we don't make the image too large
    if scale < 1:
        scale = 1
        
    return scale

def capture_game_window(image_name, window_name, cache_dir):
    """
    Captures a screenshot of the specified game window.
    Args:
        image_name (str): Name of the output image file (e.g., 'current_screen.png')
        window_name (str): Name of the window to capture (e.g., 'Phoenix Wright: Ace Attorney Trilogy')
        cache_dir (str): Directory to save the screenshot
    Returns: Path to the saved screenshot
    """
    platform_type = get_platform()
    screenshot = None
    
    if platform_type == 'windows':
        try:
            import win32gui
            import win32ui
            from ctypes import windll
            
            # Find the window using flexible matching
            hwnd = find_game_window()
            if not hwnd:
                print("Phoenix Wright window not found, falling back to pyautogui")
                screenshot = pyautogui.screenshot()
            else:
                print(f"Found game window: {win32gui.GetWindowText(hwnd)}")
                
                # Get window dimensions
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                width = right - left
                height = bottom - top
                
                # Create device context and bitmap
                hwndDC = win32gui.GetWindowDC(hwnd)
                mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                saveDC = mfcDC.CreateCompatibleDC()
                
                # Create bitmap and select it into DC
                saveBitMap = win32ui.CreateBitmap()
                saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
                saveDC.SelectObject(saveBitMap)
                
                # Copy window content
                result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
                
                # Convert to PIL Image
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                screenshot = Image.frombuffer(
                    'RGB',
                    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                    bmpstr, 'raw', 'BGRX', 0, 1)
                
                # Clean up
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwndDC)
            
        except ImportError:
            print("Windows dependencies not found, falling back to pyautogui")
            screenshot = pyautogui.screenshot()
    else:
        # For Mac and other platforms, use pyautogui
        screen_width, screen_height = pyautogui.size()
        region = (0, 0, screen_width, screen_height)
        screenshot = pyautogui.screenshot(region=region)
        
        os.makedirs(cache_dir, exist_ok=True)
        screenshot_path = os.path.join(cache_dir, "screenshot.png")
        screenshot.save(screenshot_path)

        annotate_image_path, grid_annotation_path, annotate_cropped_image_path= get_annotate_img(
            screenshot_path,
            crop_left=0,
            crop_right=710,
            crop_top=250,
            crop_bottom=300,
            grid_rows=1,
            grid_cols=1,
            cache_dir=cache_dir
        )
        screenshot = Image.open(annotate_cropped_image_path)
    
    if screenshot:
        # Get current dimensions
        current_width, current_height = screenshot.size
        
        # Calculate scale multiplier
        scale = calculate_scale_multiplier(current_width, current_height)
        
        # Calculate new dimensions
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        print(f"Scaling image from {current_width}x{current_height} to {new_width}x{new_height} (scale: {scale:.2f})")
        
        # Resize the image
        if scale > 1:  # Only resize if we're scaling up
            screenshot = screenshot.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save the screenshot in the specified cache directory
        os.makedirs(cache_dir, exist_ok=True)
        screenshot_path = os.path.join(cache_dir, image_name)
        screenshot.save(screenshot_path)
        return screenshot_path
    
    return None

def log_game_event(log_text, game_name="ace_attorney", cache_dir="cache"):
    """Logs game events with timestamp to a dedicated log file."""
    os.makedirs(os.path.join(cache_dir, game_name), exist_ok=True)
    with open(os.path.join(cache_dir, game_name, f"{game_name}.log"), "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {log_text}\n")


def log_request_cost(num_input, num_output, input_cost, output_cost, game_name, model_name,
                    input_image_tokens=0, cache_dir="cache"):
    """
    Logs API request costs including token counts and costs.
    
    Args:
        num_input (int): Total number of input tokens
        num_output (int): Number of output tokens
        input_cost (float): Total input cost
        output_cost (float): Total output cost
        game_name (str): Name of the game
        input_image_tokens (int, optional): Number of image tokens (default: 0)
        cache_dir (str): Base cache directory (default: "cache")
    """
    # Check and create base cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Handle model names with forward slashes and sanitize for file paths
    model_name = model_name.lower().split('/')[-1] if '/' in model_name else model_name.lower()

    # Create log file path
    log_file = os.path.join(cache_dir, f"{game_name}_api_costs.log")
    
    # Calculate text tokens
    input_text_tokens = num_input - input_image_tokens
    
    # Format log entry
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"[{timestamp}]\n"
        f"Game: {game_name}\n"
        f"Model: {model_name}\n"
        f"Total Input Tokens: {num_input}\n"
        f"Input Text Tokens: {input_text_tokens}\n"
        f"Input Image Tokens: {input_image_tokens}\n"
        f"Output Tokens: {num_output}\n"
        f"Total Input Cost: ${input_cost:.6f}\n"
        f"Total Output Cost: ${output_cost:.6f}\n"
        f"Total Cost: ${input_cost + output_cost:.6f}\n"
        f"{'-'*50}\n"
    )
    
    # Write to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)
    
    return log_file
