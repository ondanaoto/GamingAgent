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
        "1. Game State Detection Rules:\n"
        "   - Cross-Examination mode REQUIRES BOTH:\n"
        "     * A blue bar must be visible in the upper right corner\n"
        "     * The dialog text must be green in color\n"
        "   - If either the blue bar is missing OR the dialog text is not green, it is NOT Cross-Examination mode\n"
        "   - Evidence mode is indicated by:\n"
        "     * An evidence presentation window\n"
        "     * Evidence items being displayed\n"
        "   - If you don't see any Cross-Examination indicators (blue bar OR green text) and there's no evidence window, it's Conversation mode\n\n"
        "2. Dialog Text Analysis:\n"
        "   - Look at the bottom-left area where dialog appears\n"
        "   - Note the white color of the dialog text\n"
        "   - Extract the speaker's name and their dialog\n"
        "   - Format must be exactly: Dialog: NAME: dialog text\n"
        "   - If in Evidence mode, output: Dialog: None\n\n"
        "3. Evidence Analysis:\n"
        "   - Look at the evidence presentation window\n"
        "   - For each evidence item, provide its name and description\n"
        "   - Format must be exactly: Evidence: NAME: description\n"
        "   - If not in Evidence mode, output: Evidence: None\n\n"
        "4. Scene Analysis:\n"
        "   - Describe any visible characters and their expressions/poses\n"
        "   - Is there an evidence presentation window? How many evidence items are visible?\n"
        "   - If in Evidence mode, count and mention the total number of evidence items visible\n"
        "   - Describe any other important visual elements or interactive UI components\n"
        "   - You MUST explicitly mention:\n"
        "     * The color of the dialog text (green/white)\n"
        "     * Whether there is a blue bar in the upper right corner\n"
        "     * The presence/absence of evidence windows\n\n"
        "Format your response EXACTLY as:\n"
        "Game State: <'Cross-Examination' or 'Conversation' or 'Evidence'>\n"
        "Dialog: NAME: dialog text\n"
        "Evidence: NAME: description\n"
        "Scene: <detailed description including dialog color, blue bar presence, and other visual elements>"
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
    dialog=None,
    evidence=None
    ):
    """
    Maintains dialog history for the current episode.
    If evidence is provided, adds it to the evidences list.
    """
    # Create cache directory for dialog history if it doesn't exist
    cache_dir = os.path.join("cache", "ace_attorney", "dialog_history")
    os.makedirs(cache_dir, exist_ok=True)

    # Define the JSON file path based on the episode name
    json_file = os.path.join(cache_dir, f"{episode_name.lower().replace(' ', '_')}.json")

    # Load existing dialog history or initialize new structure
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            dialog_history = json.load(f)
    else:
        dialog_history = {
            episode_name: {
                "Case_Transcript": [],
                "evidences": []
            }
        }

    # Update dialog history if dialog is provided
    if dialog and dialog["name"] and dialog["text"]:
        dialog_entry = f"{dialog['name']}: {dialog['text']}"
        if dialog_entry not in dialog_history[episode_name]["Case_Transcript"]:
            dialog_history[episode_name]["Case_Transcript"].append(dialog_entry)

    # Update evidence if provided
    if evidence and evidence["name"] and evidence["description"]:
        evidence_entry = f"{evidence['name']}: {evidence['description']}"
        if evidence_entry not in dialog_history[episode_name]["evidences"]:
            dialog_history[episode_name]["evidences"].append(evidence_entry)

    # Save the updated dialog history
    with open(json_file, 'w') as f:
        json.dump(dialog_history, f, indent=2)

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
                "prev_responses": [],
                "evidences": []
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
    Retrieves and composes complete memory context from long-term and short-term memory.
    """
    # Load background conversation context
    background_file = "games/ace_attorney/ace_attorney_1.json"
    with open(background_file, 'r') as f:
        background_data = json.load(f)
    background_context = background_data[episode_name]["Case_Transcript"]

    # Load current episode memory
    memory_file = os.path.join("cache", "ace_attorney", "dialog_history", f"{episode_name.lower().replace(' ', '_')}.json")
    if os.path.exists(memory_file):
        with open(memory_file, 'r') as f:
            memory_data = json.load(f)
        current_episode = memory_data[episode_name]
        cross_examination_context = current_episode["Case_Transcript"]
        prev_responses = current_episode.get("prev_responses", [])
        collected_evidences = current_episode.get("evidences", [])
    else:
        cross_examination_context = []
        prev_responses = []
        collected_evidences = []

    # Compose complete memory context
    memory_context = f"""Background Conversation Context:
{chr(10).join(background_context)}

Cross-Examination Conversation Context:
{chr(10).join(cross_examination_context)}

Previous 7 manipulations:
{chr(10).join(prev_responses)}

Collected Evidences:
{chr(10).join(collected_evidences)}"""

    return memory_context

def reasoning_worker(system_prompt, api_provider, model_name, game_state, scene, memory_context, base64_image=None, modality="vision-text", thinking=True):
    """
    Makes decisions about game moves based on current game state, scene description, and memory context.
    Uses API to generate thoughtful decisions.
    
    Args:
        system_prompt (str): System prompt for the API
        api_provider (str): API provider to use
        model_name (str): Model name to use
        game_state (str): Current game state (Cross-Examination, Conversation, or Evidence)
        scene (str): Description of the current scene
        memory_context (str): Complete memory context including dialog history and evidences
        base64_image (str, optional): Base64 encoded screenshot of the current game state
        modality (str): Modality to use (vision-text or text-only)
        thinking (bool): Whether to use deep thinking
    
    Returns:
        dict: Contains move and thought
    """
    # Extract and format evidence information
    evidences_section = memory_context.split("Collected Evidences:")[1].strip()
    collected_evidences = [e for e in evidences_section.split("\n") if e.strip()]
    num_collected_evidences = len(collected_evidences)
    
    # Format evidence details for the prompt
    evidence_details = "\n".join([f"Evidence {i+1}: {e}" for i, e in enumerate(collected_evidences)])
    
    # Construct the prompt for the API
    prompt = f"""You are Phoenix Wright, a defense attorney in Ace Attorney. Your goal is to prove your client's innocence by finding contradictions in witness testimonies and presenting the right evidence at the right time.

Current Game State: {game_state}
Scene Description: {scene}

Evidence Status:
- Total Evidence Collected: {num_collected_evidences}
{evidence_details}

Memory Context:
{memory_context}

Based on this information, decide what move to make. Your response must be in the exact format:
move: <move>
thought: <explanation>

Game State Strategies:

1. Evidence Collection Priority:
   - If no evidence has been collected yet (Total Evidence Collected: 0):
     * In Conversation or Cross-Examination state: Use court_record to switch to Evidence state
     * In Evidence state: Use right to navigate through all evidence items
   - Always collect all available evidence before proceeding with other actions
   - Only exit Evidence state (using cancel) after collecting all evidence

2. Cross-Examination Mode:
   - Your primary goal is to find contradictions in the witness's testimony
   - You must choose between:
     * press: Ask for more details when you suspect there's more to the story
     * court_record: Open the court record to present evidence
     * present: Present selected evidence to point out contradictions
   - Strategy:
     * First, analyze the current testimony carefully
     * Look for inconsistencies with previous statements or evidence
     * If you need more information to find a contradiction, use press
     * When you've identified a contradiction, use court_record to present the evidence
     * Remember: You can only present evidence during cross-examination

3. Conversation Mode:
   - Your goal is to gather information and advance the story
   - ONLY use confirm to continue the conversation
   - DO NOT use press in Conversation mode
   - Pay attention to new information that might be useful later
   - Collect statements that might help in cross-examination

4. Evidence Mode:
   - Your goal is to examine and collect evidence
   - Use left/right to navigate through evidence
   - Be careful about your previous manipulations
   - Make sure your actions are coherent with the current investigation
   - When you've finished examining evidence, use cancel to exit evidence mode

Available moves:
- In Cross-Examination:
  * press: Press the witness for more information when you need clarification
  * court_record: Open the court record to present evidence
  * present: Present selected evidence to point out contradictions
- In Conversation:
  * confirm: Continue the conversation and collect statements
  * court_record: Open the court record to examine evidence
- In Evidence:
  * left/right: Navigate through evidence
  * cancel: Exit evidence mode when finished

Choose the most appropriate move and explain your reasoning. Focus on finding contradictions in cross-examination and presenting the right evidence at the right time."""

    # Call the API
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

    # Extract move and thought from response
    move_match = re.search(r"move:\s*(.+?)(?=\n|$)", response)
    thought_match = re.search(r"thought:\s*(.+?)(?=\n|$)", response)
    
    move = move_match.group(1).strip() if move_match else ""
    thought = thought_match.group(1).strip() if thought_match else ""

    return {
        "move": move,
        "thought": thought
    }

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
    7) Makes decisions about game moves.
    
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
    game_state_match = re.search(r"Game State:\s*(Cross-Examination|Conversation|Evidence)", response_text)
    game_state = game_state_match.group(1) if game_state_match else "Unknown"
    
    # Extract Dialog (with NAME: text format)
    dialog_match = re.search(r"Dialog:\s*([^:\n]+):\s*(.+?)(?=\n|$)", response_text)
    dialog = {
        "name": dialog_match.group(1) if dialog_match else "",
        "text": dialog_match.group(2).strip() if dialog_match else ""
    }
    
    # Extract Evidence
    evidence_match = re.search(r"Evidence:\s*([^:\n]+):\s*(.+?)(?=\n|$)", response_text)
    evidence = {
        "name": evidence_match.group(1) if evidence_match else "",
        "description": evidence_match.group(2).strip() if evidence_match else ""
    }
    
    # Extract Scene Description
    scene_match = re.search(r"Scene:\s*(.+?)(?=\n|$)", response_text, re.DOTALL)
    scene = scene_match.group(1).strip() if scene_match else ""

    # -------------------- Memory Processing -------------------- #
    # First, update long-term and short-term memory
    if game_state == "Evidence":
        # If in Evidence mode, update evidence instead of dialog
        if evidence["name"] and evidence["description"]:
            evidence_file = long_term_memory_worker(
                system_prompt,
                api_provider,
                model_name,
                prev_response,
                thinking,
                modality,
                episode_name,
                evidence=evidence
            )
    else:
        # If in Conversation or Cross-Examination mode, update dialog
        if dialog["name"] and dialog["text"]:
            dialog_file = long_term_memory_worker(
                system_prompt,
                api_provider,
                model_name,
                prev_response,
                thinking,
                modality,
                episode_name,
                dialog=dialog
            )

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
    # print("[Memory Context]")
    # print(complete_memory)

    # -------------------- Reasoning -------------------- #
    # Make decisions about game moves
    reasoning_result = reasoning_worker(
        system_prompt,
        api_provider,
        model_name,
        game_state,
        scene,
        complete_memory,
        base64_image=encode_image(vision_result["screenshot_path"]),
        modality=modality,
        thinking=thinking
    )
    print("[Reasoning Result]")
    print(reasoning_result)

    parsed_result = {
        "game_state": game_state,
        "dialog": dialog,
        "evidence": evidence,
        "scene": scene,
        "screenshot_path": vision_result["screenshot_path"],
        "memory_context": complete_memory,
        "move": reasoning_result["move"],
        "thought": reasoning_result["thought"]
    }

    return parsed_result

# return move_thought_list