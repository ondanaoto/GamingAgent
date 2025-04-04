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
    """
    Directly performs the move using keyboard input without key mapping.
    """
    if move.lower() in ["up", "down", "left", "right"]:
        # For arrow keys, use the direct key name
        pyautogui.keyDown(move.lower())
        time.sleep(0.1)
        pyautogui.keyUp(move.lower())
    else:
        # For other keys, use the direct key press
        pyautogui.press(move.lower())
        time.sleep(0.1)
    
    print(f"Performed move: {move}")

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

def vision_evidence_worker(system_prompt, api_provider, model_name, modality, thinking):
    # Capture game window screenshot
    screenshot_path = capture_game_window(
        image_name="ace_attorney_screenshot_evidence.png",
        window_name="Phoenix Wright: Ace Attorney Trilogy",
        cache_dir=CACHE_DIR
    )
    if not screenshot_path:
        return {"error": "Failed to capture game window"}

    base64_image = encode_image(screenshot_path)
    
    # Construct prompt for LLM
    prompt = (
        "You are now playing Ace Attorney. Analyze the current scene and provide the following information:\n\n"
        "Look at the evidence presentation window\n"
        "For the item, provide its name and description\n"

        "Format your response EXACTLY as:\n"
        "Game State: <'Evidence'>\n"
        "Dialog: None\n"
        "Evidence: NAME: description\n"
        "Scene: <detailed description includes other visual elements>"
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
        "   - Cross-Examination mode is indicated by ANY of these:\n"
        "     * A blue bar in the upper right corner\n"
        "     * Only Green dialog text\n"
        "     * EXACTLY three UI elements at the right down corner: Options, Press, Present\n"
        "     * An evidence window visible in the middle of the screen\n"
        "   - If you see an evidence window, it is ALWAYS Cross-Examination mode\n"
        "   - Conversation mode is indicated by:\n"
        "     * EXACTLY two UI elements at the right down corner: Options, Court Record\n"
        "     * Dialog text can be any color (most commonly white, but can also be blue, red, etc.)\n"
        "   - If you don't see any Cross-Examination indicators, it's Conversation mode\n\n"
        "2. Dialog Text Analysis:\n"
        "   - Look at the bottom-left area where dialog appears\n"
        "   - Note the color of the dialog text (green/white/blue/red)\n"
        "   - Extract the speaker's name and their dialog\n"
        "   - Format must be exactly: Dialog: NAME: dialog text\n\n"
        "3. Scene Analysis:\n"
        "   - Describe any visible characters and their expressions/poses\n"
        "   - Describe any other important visual elements or interactive UI components\n"
        "   - You MUST explicitly mention:\n"
        "     * The color of the dialog text (green/white/blue/red)\n"
        "     * Whether there is a blue bar in the upper right corner\n"
        "     * The exact UI elements present at the right down corner (Options, Press, Present for Cross-Examination or Options, Court Record for Conversation)\n"
        "     * Whether there is an evidence window visible\n"
        "     * If evidence window is visible:\n"
        "       - Name of the currently selected evidence\n"
        "       - Description of the evidence\n"
        "       - Position in the evidence list (if visible)\n"
        "       - Whether this is the evidence you intend to present\n\n"

        "Format your response EXACTLY as:\n"
        "Game State: <'Cross-Examination' or 'Conversation'>\n"
        "Dialog: NAME: dialog text\n"
        "Evidence: NAME: description\n"
        "Scene: <detailed description including dialog color, blue bar presence, UI elements(corresponding keys, like r Present/Court Record or x present), evidence window status and contents, and other visual elements>"
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
    if len(prev_responses) > 14:
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

CURRENT GAME STATE: {game_state}
This is your current state. All decisions must be based on this state only.

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

1. Conversation Mode (CURRENT STATE: {game_state}):
   - Use 'z' to continue the conversation
   - DO NOT use any other commands in Conversation mode
   - No press command in Conversation mode

2. Cross-Examination Mode (CURRENT STATE: {game_state}):
   - ALWAYS compare the witness's statement with the available evidence
   - For each statement, you have two options:
     * If you find a clear contradiction with evidence: (Three steps, you can only do one step at a time. Be coherent with previous response.)
       - First step: Use 'r' to open the evidence window
       - Second step: Navigate through evidence using 'right'/'left'
         * Look at each evidence carefully
         * Only stop when you find the evidence that directly contradicts the statement
         * If the current evidence doesn't match, keep navigating
         * Be absolutely sure the evidence contradicts the statement before presenting
       - Third step: Use 'x' to show the contradicting evidence
         * Only present when you're certain this is the right evidence
         * The evidence must directly contradict the witness's statement
       - No need to ask more questions if you're confident about the contradiction
     * If you don't find a contradiction or need more information:
       - Use 'l' to ask more details from the witness about their statement (it will swtich to conversation mode. You can use it to ask more details about his statement. After a few rounds of conversation, it will automatically switch back to cross-examination mode.)
       - Or use 'z' to move to their next statement if you don't need to ask more
   - The evidence window will automatically close after showing evidence
   - DO NOT use 'x'/'r' unless you have found a clear contradiction
   - DO NOT use 'x'/'r' if you don't see an evidence window

Available moves:
- In Conversation:
  * 'z': Continue the conversation
- In Cross-Examination:
  * 'l': Question the witness about their current statement
  * 'z': Move to the next statement if you don't need to ask more
  * 'r': Open the evidence window to show contradicting evidence
  * 'x': Show the selected evidence (only when you're certain it contradicts)
  * 'right'/'left': Navigate through evidence items to find the correct one

Before using 'x', always ask:
- Is the currently selected evidence the one I want to present?

If the currently selected evidence is NOT the one I intend to present:
- Use 'right' or 'left' to navigate until the correct one is selected
- Do NOT assume the correct evidence is selected
- NEVER use 'x' until the correct item is confirmed to be selected

Example 1 - Finding and Presenting Evidence:

Scene says: "The currently selected evidence is 'Attorney's Badge'."

But I want to present: "Cindy's Autopsy Report"

So I do:
Turn 1:
move: right
thought: The Autopsy Report is not currently selected. I'll navigate to it.

Turn 2:
move: x
thought: The Autopsy Report is now selected. I'll present it to contradict the witness.

Example 2 - Clear Contradiction with No Evidence Window:

Memory Context:
Witness: "I was at home at 8 PM last night."
Evidence: "Security Camera Footage: Shows the witness at the crime scene at 8 PM."

Scene: "Dialog text is green. There is a blue bar in the upper right corner. There are exactly three UI elements at the right down corner: Options, Press, Present. No evidence window is visible on screen. The witness is sweating and looking nervous."

Turn 1:
move: r
thought: I see a clear contradiction between the witness's statement and our security camera footage. The scene shows we're in cross-examination mode (green text, blue bar, three UI elements) but no evidence window is visible. I need to first open the evidence window by pressing 'r'.

Turn 2:
move: right
thought: I need to navigate to the security camera footage that proves the witness was at the crime scene at 8 PM.

Turn 3:
move: x
thought: I've found the security camera footage that proves the witness was at the crime scene at 8 PM. I'll present it to expose the contradiction.

Stuck Situation Handling:
- If you notice that you haven't made any progress in the last 7 responses (check prev_responses)
- If you're stuck in a loop or can't move forward
- Use 'b' to jump out of the stucking loop
- This is a general rule to prevent getting stuck in the game
"""

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

def ace_evidence_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    episode_name="The First Turnabout",
    ):
    """
    1) Loops through evidence items until all are seen or a duplicate appears.
    2) Updates memory with newly seen evidence.
    3) Returns after exiting Evidence view.
    """
    assert modality in ["text-only", "vision-text"], f"modality {modality} is not supported."

    seen_names = set()
    duplicate_found = False

    # Start in evidence mode
    perform_move("r")
    time.sleep(1)

    while not duplicate_found:
        # Analyze the current evidence screen
        vision_result = vision_evidence_worker(
            system_prompt,
            api_provider,
            model_name,
            modality,
            thinking
        )
        # print("[Vision Analysis Result]")
        # print(vision_result)

        if "error" in vision_result:
            return vision_result

        response_text = vision_result["response"]

        # Extract evidence info
        evidence_match = re.search(r"Evidence:\s*([^:\n]+):\s*(.+?)(?=\n|$)", response_text)
        evidence = {
            "name": evidence_match.group(1) if evidence_match else "",
            "description": evidence_match.group(2).strip() if evidence_match else ""
        }

        # Update memory if new
        if evidence["name"] and evidence["description"]:
            if evidence["name"] in seen_names:
                duplicate_found = True
                perform_move("cancel")
                time.sleep(1)
                print(f"[INFO] Duplicate evidence '{evidence['name']}' detected. Exiting evidence view.")
                break
            else:
                seen_names.add(evidence["name"])
                long_term_memory_worker(
                    system_prompt,
                    api_provider,
                    model_name,
                    prev_response,
                    thinking,
                    modality,
                    episode_name,
                    evidence=evidence
                )
                perform_move("right")
                print(f"[INFO] New evidence collected: {evidence['name']}")
                time.sleep(1)

    perform_move("r")
    # No need to run reasoning or return reasoning_result
    return {
        "game_state": "Evidence",
        "evidence": evidence,
        "seen": list(seen_names)
    }

def ace_attorney_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    episode_name="The First Turnabout",
    ):
    """
    1) Captures a screenshot of the current game state.
    2) Analyzes the scene using vision worker.
    3) Makes decisions based on the scene analysis.
    4) Maintains dialog history for the current episode.
    5) Makes decisions about game moves.
    
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
        modality="vision-text"
    )

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
    # Update long-term memory only
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
        modality='text-only',
        thinking=thinking
    )

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