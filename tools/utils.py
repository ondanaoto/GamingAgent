import time
import os
import pyautogui
import base64
import anthropic
import numpy as np
import concurrent.futures
import re


def encode_image(image_path):
    """
    Read a file from disk and return its contents as a base64-encoded string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def log_output(thread_id, log_text, game):
    """
    Logs output to `cache/thread_{thread_id}/output.log`
    """
    thread_folder = f"cache/{game}/thread_{thread_id}"
    os.makedirs(thread_folder, exist_ok=True)
    
    log_path = os.path.join(thread_folder, "output.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
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