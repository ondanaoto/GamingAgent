import time
import os
import pyautogui
import numpy as np

from tools.utils import encode_image, log_output, get_annotate_img, capture_game_window
from tools.serving.api_providers import anthropic_completion, anthropic_text_completion, openai_completion, openai_text_reasoning_completion, gemini_completion, gemini_text_completion, deepseek_text_reasoning_completion
import re
import json

CACHE_DIR = "cache/ace_attorney"

def perform_move(move):
    key_map = {
        # Core actions
        "confirm": "enter",
        "present": "e",
        "court_record": "tab",
        "profiles": "r",
        "cancel": "backspace",
        "press": "q",
        "options": "esc",
        "return_title": "j",
        
        # Movement controls
        "up": "up",
        "down": "down",
        "right": "right",
        "left": "left",
        
        # Evidence controls
        "rotate_up": "h",
        "rotate_down": "n",
        "rotate_right": "m",
        "rotate_left": "b"
    }
    
    if move.lower() in key_map:
        pyautogui.press(key_map[move.lower()])
        print(f"Performed move: {move}")
    else:
        print(f"[WARNING] Invalid move: {move}")

def log_move_and_thought(move, thought, latency):
    """
    Logs the move and thought process into a log file inside the cache directory.
    """
    log_file_path = os.path.join(CACHE_DIR, "ace_attorney_moves.log")
    
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Move: {move}, Thought: {thought}, Latency: {latency:.2f} sec\n"
    
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write(log_entry)
    except Exception as e:
        print(f"[ERROR] Failed to write log entry: {e}")

def vision_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    ):
    """
    Captures and analyzes the current game screen.
    Returns scene analysis including game state, dialog text, and detailed scene description.
    """
    assert modality == "vision-text", "Vision worker requires vision-text modality"

    # Capture game window screenshot
    screenshot_path = capture_game_window(
        image_name="ace_attorney_screenshot.png",
        window_name="Phoenix Wright: Ace Attorney Trilogy",
        cache_dir=CACHE_DIR
    )
    if not screenshot_path:
        return {"error": "Failed to capture game window"}

    base64_image = encode_image(screenshot_path)
    
    prompt = (
        "You are now playing Ace Attorney. Analyze the current scene and provide the following information:\n\n"
        "1. Game State (for model's context):\n"
        "   - Cross-Examination mode is indicated by either:\n"
        "     * A blue bar in the upper right corner\n"
        "     * Dialog text appearing in green color\n"
        "   - If neither is present, it's Conversation mode\n\n"
        "2. Dialog Text (Focus on lower-left portion of screen):\n"
        "   - Look at the bottom-left area where dialog appears\n"
        "   - Extract the speaker's name and their dialog\n"
        "   - Format must be exactly: Dialog: NAME: dialog text\n"
        "   - Name should be exactly as shown on screen\n\n"
        "3. Scene Details:\n"
        "   - Describe any visible characters and their expressions/poses\n"
        "   - Is there an evidence presentation window? If yes, what evidence items are visible?\n"
        "   - Describe any other important visual elements or interactive UI components\n"
        "   - Note any visual cues that might be important for gameplay\n\n"
        "Format your response EXACTLY as:\n"
        "Game State: <'Cross-Examination' or 'Conversation'>\n"
        "Dialog: NAME: dialog text\n"
        "Scene: <detailed description of characters, evidence, UI elements, and important visual information>"
    )

    print(f"Calling {model_name} API for vision analysis...")
    
    if api_provider == "anthropic" and modality=="text-only":
        response = anthropic_text_completion(system_prompt, model_name, prompt, thinking)
    elif api_provider == "anthropic":
        response = anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking)
    elif api_provider == "openai" and "o3" in model_name and modality=="text-only":
        response = openai_text_reasoning_completion(system_prompt, model_name, prompt)
    elif api_provider == "openai":
        response = openai_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "gemini" and modality=="text-only":
        response = gemini_text_completion(system_prompt, model_name, prompt)
    elif api_provider == "gemini":
        response = gemini_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "deepseek":
        response = deepseek_text_reasoning_completion(system_prompt, model_name, prompt)
    else:
        raise NotImplementedError(f"API provider: {api_provider} is not supported.")

    return {
        "response": response,
        "screenshot_path": screenshot_path
    }

def long_term_memory_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    episode_name="The First Turnabout",
    dialog=None
    ):
    """
    Maintains a long-term memory of dialog history by appending new dialog text to a JSON file.
    Args:
        episode_name (str): Name of the current episode
        dialog (dict): Dialog information containing speaker name and text
    """
    if not dialog:
        return
        
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(CACHE_DIR, "dialog_history")
    os.makedirs(cache_dir, exist_ok=True)
    
    # JSON file path for the episode
    json_file = os.path.join(cache_dir, f"{episode_name.lower().replace(' ', '_')}.json")
    
    # Load existing dialog history or create new one
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            dialog_history = json.load(f)
    else:
        dialog_history = {
            episode_name: {
                "Case_Transcript": []
            }
        }
    
    # Format dialog as "NAME: text"
    dialog_text = f"{dialog['name']}: {dialog['text']}"
    
    # Add new dialog entry if it's not already the last entry
    current_transcript = dialog_history[episode_name]["Case_Transcript"]
    if not current_transcript or current_transcript[-1] != dialog_text:
        current_transcript.append(dialog_text)
        
        # Save updated dialog history
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(dialog_history, f, indent=4, ensure_ascii=False)
            
    return json_file

def short_term_memory_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    episode_name="The First Turnabout"
    ):
    """
    Maintains a short-term memory of previous responses by storing the last 7 responses in the JSON file.
    Args:
        episode_name (str): Name of the current episode
        prev_response (str): The new response to add to the queue
    """
    if not prev_response:
        return
        
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(CACHE_DIR, "dialog_history")
    os.makedirs(cache_dir, exist_ok=True)
    
    # JSON file path for the episode
    json_file = os.path.join(cache_dir, f"{episode_name.lower().replace(' ', '_')}.json")
    
    # Load existing dialog history or create new one
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            dialog_history = json.load(f)
    else:
        dialog_history = {
            episode_name: {
                "Case_Transcript": [],
                "prev_responses": []
            }
        }
    
    # Initialize prev_responses if it doesn't exist
    if "prev_responses" not in dialog_history[episode_name]:
        dialog_history[episode_name]["prev_responses"] = []
    
    # Add new response and maintain only last 7 responses
    prev_responses = dialog_history[episode_name]["prev_responses"]
    prev_responses.append(prev_response)
    if len(prev_responses) > 7:
        prev_responses.pop(0)  # Remove oldest response
        
    # Save updated dialog history
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(dialog_history, f, indent=4, ensure_ascii=False)
            
    return json_file

def memory_retrieval_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    episode_name="The First Turnabout"
    ):
    """
    Retrieves and composes memory context from long-term and short-term memory.
    Returns a formatted string containing background context, cross-examination context, and recent manipulations.
    """
    # Create cache directory path for dialog history
    cache_dir = os.path.join(CACHE_DIR, "dialog_history")
    json_file = os.path.join(cache_dir, f"{episode_name.lower().replace(' ', '_')}.json")
    
    # Initialize memory components
    background_context = ""
    cross_examination_context = ""
    previous_manipulations = ""
    
    # Load background conversation context from ace_attorney_1.json
    background_file = os.path.join("games/ace_attorney/ace_attorney_1.json")
    print(f"\n[DEBUG] Attempting to load background file from: {background_file}")
    
    if os.path.exists(background_file):
        print("[DEBUG] Background file found!")
        with open(background_file, 'r', encoding='utf-8') as f:
            background_data = json.load(f)
            print(f"[DEBUG] Loaded background data keys: {list(background_data.keys())}")
            
            if "The First Turnabout" in background_data and "Case_Transcript" in background_data["The First Turnabout"]:
                background_context = "\n".join(background_data["The First Turnabout"]["Case_Transcript"])
                print(f"[DEBUG] Successfully loaded {len(background_data['The First Turnabout']['Case_Transcript'])} lines of background context")
                print("[DEBUG] First few lines of background context:")
                print("\n".join(background_data["The First Turnabout"]["Case_Transcript"][:3]))
            else:
                print("[DEBUG] Could not find 'The First Turnabout' or 'Case_Transcript' in background data")
    else:
        print(f"[DEBUG] Background file not found at: {background_file}")
    
    # Load and process memory if file exists
    if os.path.exists(json_file):
        print(f"\n[DEBUG] Loading current episode memory from: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
            
            # Extract cross-examination context from current episode
            if episode_name in memory_data and "Case_Transcript" in memory_data[episode_name]:
                cross_examination_context = "\n".join(memory_data[episode_name]["Case_Transcript"])
                print(f"[DEBUG] Successfully loaded {len(memory_data[episode_name]['Case_Transcript'])} lines of cross-examination context")
            
            # Extract previous 7 manipulations
            if episode_name in memory_data and "prev_responses" in memory_data[episode_name]:
                previous_manipulations = "\n".join(memory_data[episode_name]["prev_responses"])
                print(f"[DEBUG] Successfully loaded {len(memory_data[episode_name]['prev_responses'])} previous manipulations")
    else:
        print(f"[DEBUG] Current episode memory file not found at: {json_file}")
    
    # Compose complete memory context
    complete_memory = (
        f"Background Conversation Context:\n{background_context}\n\n"
        f"Cross-Examination Conversation Context:\n{cross_examination_context}\n\n"
        f"Previous 7 manipulations:\n{previous_manipulations}"
    )
    
    return complete_memory

def ace_attorney_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    episode_name="The First Turnabout"
    ):
    """
    1) Captures a screenshot of the current game state.
    2) Analyzes the scene using vision worker.
    3) Makes decisions based on the scene analysis.
    4) Maintains dialog history for the current episode.
    5) Maintains short-term memory of previous responses.
    6) Retrieves and composes complete memory context.
    
    Args:
        episode_name (str): Name of the current episode (default: "The First Turnabout")
    """
    assert modality in ["text-only", "vision-text"], f"modality {modality} is not supported."

    # -------------------- Vision Processing -------------------- #
    # First, analyze the current game state using vision
    vision_result = vision_worker(
        system_prompt,
        api_provider,
        model_name,
        prev_response,
        thinking,
        modality
    )
    print("[Vision Analysis Result]")
    print(vision_result)

    if "error" in vision_result:
        return vision_result

    # Extract the formatted outputs using regex
    response_text = vision_result["response"]
    
    # Extract Game State
    game_state_match = re.search(r"Game State:\s*(Cross-Examination|Conversation)", response_text)
    game_state = game_state_match.group(1) if game_state_match else "Unknown"
    
    # Extract Dialog (with NAME: text format)
    dialog_match = re.search(r"Dialog:\s*([^:\n]+):\s*(.+?)(?=\n|$)", response_text)
    dialog = {
        "name": dialog_match.group(1) if dialog_match else "",
        "text": dialog_match.group(2).strip() if dialog_match else ""
    }
    
    # Extract Scene Description
    scene_match = re.search(r"Scene:\s*(.+?)(?=\n|$)", response_text, re.DOTALL)
    scene = scene_match.group(1).strip() if scene_match else ""

    # -------------------- Memory Processing -------------------- #
    # First, update long-term and short-term memory
    if dialog["name"] and dialog["text"]:
        dialog_file = long_term_memory_worker(
            system_prompt,
            api_provider,
            model_name,
            prev_response,
            thinking,
            modality,
            episode_name,
            dialog
        )
        print(f"[Dialog saved to {dialog_file}]")

    if prev_response:
        short_term_memory_worker(
            system_prompt,
            api_provider,
            model_name,
            prev_response,
            thinking,
            modality,
            episode_name
        )

    # Then, retrieve and compose complete memory context
    complete_memory = memory_retrieval_worker(
        system_prompt,
        api_provider,
        model_name,
        prev_response,
        thinking,
        modality,
        episode_name
    )
    print("[Memory Context]")
    print(complete_memory)

    parsed_result = {
        "game_state": game_state,
        "dialog": dialog,
        "scene": scene,
        "screenshot_path": vision_result["screenshot_path"],
        "memory_context": complete_memory
    }

    return parsed_result

# return move_thought_list