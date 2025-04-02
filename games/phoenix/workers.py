import time
import os
import pyautogui
import numpy as np
import re
import json

from tools.utils import encode_image, log_output, extract_python_code, get_annotate_img
from tools.serving.api_providers import anthropic_completion, openai_completion, gemini_completion, anthropic_text_completion, gemini_text_completion, openai_text_reasoning_completion, deepseek_text_reasoning_completion

cache_dir = "cache/phoenix"
history_path = os.path.join(cache_dir, "phoenix_history.txt")

def update_history(entry):
    """
    Adds an entry to the Phoenix Wright history log.
    Avoids duplicates and only adds meaningful entries.
    """
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # Read existing history
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                existing = f.read()
        else:
            existing = ""

        if entry.strip() and entry.strip() not in existing:
            with open(history_path, "a") as f:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {entry.strip()}\n")
    except Exception as e:
        print(f"[ERROR] Failed to update history: {e}")



def log_move_and_thought(move, thought, latency):
    """
    Logs the move and thought process into a log file inside the cache directory.
    """
    log_file_path = os.path.join(cache_dir, "phoenix_moves.log")
    
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Move: {move}, Thought: {thought}, Latency: {latency:.2f} sec\n"
    
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write(log_entry)
    except Exception as e:
        print(f"[ERROR] Failed to write log entry: {e}")

intro = """
August 3, 9:47 AM
District Court
Defendant Lobby No. 2

Phoenix:
(Boy am I nervous!)

Mia:
Wright!

Phoenix:
Oh, h-hiya, Chief.

Mia:
Whew, I'm glad I made it on time. Well, I have to say Phoenix, I'm impressed! Not everyone takes on a murder trial right off the bat like this. It says a lot about you... and your client as well.

Phoenix:
Um... thanks. Actually, it's because I owe him a favor.

Mia:
A favor? You mean, you knew the defendant before this case?

Phoenix:
Yes. Actually, I kind of owe my current job to him. He's one of the reasons I became an attorney.

Mia:
Well, that's news to me!

Phoenix:
I want to help him out any way I can! I just... really want to help him. I owe him that much.

???:
(It's over! My life, everything, it's all over!)

Mia:
... Is that your client screaming over there?

Phoenix:
Yeah... it's him.

???:
(Death! Despair! Ohhhh! I'm gonna do it, I'm gonna die!!!)

Mia:
It sounds like he wants to die...

Phoenix:
Um, yeah. *sigh*

Butz:
Nick!!!

Phoenix:
Hey. Hey there, Larry.

Butz:
Dude, I'm so guilty!! Tell them I'm guilty!!! Gimme the death sentence! I ain't afraid to die!

Phoenix:
What!? What's wrong, Larry?

Butz:
Oh, it's all over... I... I'm finished. Finished! I can't live in a world without her! I can't! Who... who took her away from me, Nick? Who did this!? Aww, Nick, ya gotta tell me! Who took my baby away!?

Phoenix:
(Hmm... The person responsible for your girlfriend's death? The newspapers say it was you...)

Phoenix:
My name is Phoenix Wright. Here's the story: My first case is a fairly simple one. A young woman was killed in her apartment. The guy they arrested was the unlucky sap dating her: Larry Butz... my best friend since grade school. Our school had a saying: "When something smells, it's usually the Butz." In the 23 years I've known him, it's usually been true. He has a knack for getting himself in trouble. One thing I can say though: it's usually not his fault. He just has terrible luck. But I know better than anyone, that he's a good guy at heart. That and I owe him one. Which is why I took the case... to clear his name. And that's just what I'm going to do!

August 3, 10:00 AM
District Court
Courtroom No. 2

Judge:
Court is now in session for the trial of Mr. Larry Butz.

Payne:
The prosecution is ready, Your Honor.

Phoenix:
The, um, defense is ready, Your Honor.

Judge:
Ahem. Mr. Wright? This is your first trial, is it not?

Phoenix:
Y-Yes, Your Honor. I'm, um, a little nervous.

Judge:
Your conduct during this trial will decide the fate of your client. Murder is a serious charge. For your client's sake, I hope you can control your nerves.

Phoenix:
Thank... thank you, Your Honor.

Judge:
... Mr. Wright, given the circumstances... I think we should have a test to ascertain your readiness.

Phoenix:
Yes, Your Honor. (Gulp... Hands shaking... Eyesight... fading...)

Judge:
The test will consist of a few simple questions. Answer them clearly and concisely. Please state the name of the defendant in this case.

Phoenix Wright


Larry Butz


Mia Fey


Phoenix:
The defendant? Well, that's Larry Butz, Your Honor.

Judge:
Correct. Just keep your wits about you and you'll do fine. Next question: This is a murder trial. Tell me, what's the victim's name?

Phoenix:
(Whew, I know this one! Glad I read the case report cover to cover so many times. It's... wait... Uh-oh! No... no way! I forgot! I'm drawing a total blank here!)

Mia:
Phoenix! Are you absolutely SURE you're up to this? You don't even know the victim's name!?

Phoenix:
Oh, the victim! O-Of course I know the victim's name! I, um, just forgot. ...Temporarily.

Mia:
I think I feel a migraine coming on. Look, the victim's name is listed in the Court Record. Just press [the R Button / Tab] to check it at any time, okay? Remember to check it often. Do it for me, please. I'm begging you.

Judge:
Mr. Wright. Who is the victim in this case?

Mia Fey


Cinder Block


Cindy Stone


Phoenix:
Um... the victim's name is Cindy Stone.

Judge:
Correct. Now, tell me, what was the cause of death? She died because she was...?

Poisoned


Hit with a blunt object


Strangled


Phoenix:
She was struck once, by a blunt object.

Judge:
Correct. You've answered all my questions. I see no reason why we shouldn't proceed. You seem much more relaxed, Mr. Wright. Good for you.

Phoenix:
Thank you, Your Honor. (Because I don't FEEL relaxed, that's for sure.)

Judge:
Well, then... First, a question for the prosecution. Mr. Payne?

Payne:
Yes, Your Honor?

Judge:
As Mr. Wright just told us, the victim was struck with a blunt object. Would you explain to the court just what that "object" was?

Payne:
The murder weapon was this statue of "The Thinker." It was found lying on the floor, next to the victim.

Judge:
I see... The court accepts it into evidence.

Statue added to the Court Record.

Mia:
Wright... Be sure to pay attention to any evidence added during the trial. That evidence is the only ammunition you have in court. Use [the R Button / Tab] to check the Court Record frequently.

Judge:
Mr. Payne, the prosecution may call its first witness.

Payne:
The prosecution calls the defendant, Mr. Butz, to the stand.

Phoenix:
Um, Chief, what do I do now?

Mia:
Pay attention. You don't want to miss any information that might help your client's case. You'll get your chance to respond to the prosecution later, so be ready! Let's just hope he doesn't say anything... unfortunate.

Phoenix:
(Uh oh, Larry gets excited easily... This could be bad.)

Payne:
Ahem. Mr. Butz. Is it not true that the victim had recently dumped you?

Butz:
Hey, watch it buddy! We were great together! We were Romeo and Juliet, Cleopatra and Mark Anthony!

Phoenix:
(Um... didn't they all die?)

Butz:
I wasn't dumped! She just wasn't taking my phone calls. Or seeing me... Ever. WHAT'S IT TO YOU, ANYWAY!?

Payne:
Mr. Butz, what you describe is generally what we mean by "dumped." In fact, she had completely abandoned you... and was seeing other men! She had just returned from overseas with one of them the day before the murder!

Butz:
Whaddya mean, "one of them"!? Lies! All of it, lies! I don't believe a word of it!

Payne:
Your Honor, the victim's passport. According to this, she was in Paris until the day before she died.

Passport added to the Court Record.

Judge:
Hmm... Indeed, she appears to have returned the day before the murder.

Butz:
Dude... no way...

Payne:
The victim was a model, but did not have a large income. It appears that she had several "Sugar Daddies."

Butz:
Daddies? Sugar?

Payne:
Yes. Older men, who gave her money and gifts. She took their money and used it to support her lifestyle.

Butz:
Duuude!

Payne:
We can clearly see what kind of woman this Ms. Stone was. Tell me, Mr. Butz, what do you think of her now?

Mia:
Wright... I don't think you want him to answer that question.

Phoenix:
(Yeah... Larry has a way of running his mouth in all the wrong directions. Should I...?)

Wait and see what happens


Stop him from answering


Butz:
I'm gonna die. I'm just gonna drop dead! Yeah, and when I meet her in the afterlife... I'm going to get to the bottom of this!

Judge:
Let's continue with the trial, shall we?

Payne:
I believe the accused's motive is clear to everyone.

Judge:
Yes, quite.

Phoenix:
(Oh boy. This is so not looking good.)

Payne:
Next question! You went to the victim's apartment on the day of the murder, did you not?

Butz:
Gulp!

Payne:
Well, did you, or did you not?

Butz:
Heh? Heh heh. Well, maybe I did, and maybe I didn't!

Phoenix:
(Uh oh. He went. What do I do?)

Have him answer honestly


Stop him from answering


Judge:
Well, that simplifies matters. Who is your witness?

Payne:
The man who found the victim's body. Just before making the gruesome discovery... He saw the defendant fleeing the scene of the crime!

Judge:
Order! Order in the court! Mr. Payne, the prosecution may call its witness.

Payne:
Yes, Your Honor.

Phoenix:
(This is bad...)

Payne:
On the day of the murder, my witness was selling newspapers at the victim's building. Please bring Mr. Frank Sahwit to the stand!

Payne:
Mr. Sahwit, you sell newspaper subscriptions, is this correct?

Sahwit:
Oh, oh yes! Newspapers, yes!

Judge:
Mr. Sahwit, you may proceed with your testimony. Please tell the court what you saw on the day of the murder.

Witness Testimony
-- Witness's Account --
Sahwit:
I was going door-to-door, selling subscriptions when I saw a man fleeing an apartment.
I thought he must be in a hurry because he left the door half-open behind him.
Thinking it strange, I looked inside the apartment.
Then I saw her lying there... A woman... not moving... dead!
I quailed in fright and found myself unable to go inside.
I thought to call the police immediately!
However, the phone in her apartment wasn't working.
I went to a nearby park and found a public phone.
I remember the time exactly: It was 1:00 PM.
The man who ran was, without a doubt, the defendant sitting right over there.

Judge:
Hmm...

Phoenix:
(Larry! Why didn't you tell the truth? I can't defend you against a testimony like that!)

Judge:
Incidentally, why wasn't the phone in the victim's apartment working?

Payne:
Your Honor, at the time of the murder, there was a blackout in the building.

Judge:
Aren't phones supposed to work during a blackout?

Payne:
Yes, Your Honor... However, some cordless phones do not function normally. The phone that Mr. Sahwit used was one of those. Your Honor... I have a record of the blackout, for your perusal.

Blackout Record added to the Court Record.

Judge:
Now, Mr. Wright...

Phoenix:
Yes! Er... yes, Your Honor?

Judge:
You may begin your cross-examination.

Phoenix:
C-Cross-examination, Your Honor?

Mia:
Alright, Wright, this is it. The real deal!

Phoenix:
Uh... what exactly am I supposed to do?

Mia:
Why, you expose the lies in the testimony the witness just gave!

Phoenix:
Lies! What?! He was lying!?

Mia:
Your client is innocent, right? Then that witness must have lied in his testimony! Or is your client really... guilty?

Phoenix:
!!! How do I prove he's not?

Mia:
You hold the key! It's in the evidence! Compare the witness's testimony to the evidence at hand. There's bound to be a contradiction in there! First, find contradictions between the Court Record and the witness's testimony. Then, once you've found the contradicting evidence... present it and rub it in the witness's face!

Phoenix:
Um... okay.

Mia:
Open the Court Record with [the R Button / Tab], then point out contradictions in the testimony!

Cross Examination
-- Witness's Account --
"""
update_history(intro.strip())
 
def phoenix_read_worker(system_prompt, api_provider, model_name, image_path, modality, thinking):
    base64_image = encode_image(image_path)
    
    # Construct prompt for LLM
    prompt = (
        "You are analyzing a screenshot from the game 'Phoenix Wright: Ace Attorney'. "
        "Your task is to extract key scene information for decision-making:\n"
        "- Who is speaking (name or 'Unknown')?\n"
        "- What is the dialogue text?\n"
        "- What color is used for the dialogue text, if any? (e.g., white, blue, green)\n"
        "- Are there menu choices, evidence shown, or objection moments?\n"
        "- Any visual cues or context that hint at the intensity or purpose of this moment?\n\n"

        "Respond in this strict format:\n"
        "- Speaker: [Name or Unknown]\n"
        "- Color: [Text color or 'Unknown']\n"
        "- Dialogue: \"...\"\n"
        "- UI Elements: [List of any visible buttons, evidence, or menus]\n"
        "- Context: [Brief scene summary]"
    )


    
    # Call LLM API based on provider
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
    
    # Process response and format as structured board output
    structured_board = response.strip()
    
    # Generate final text output
    final_output = "\nScene analysis:\n" + structured_board
    update_history(final_output)

    return final_output

def phoenix_read_evidence_worker(system_prompt, api_provider, model_name, image_path, modality, thinking):
    base64_image = encode_image(image_path)
    
    # Construct prompt for LLM
    prompt = (
        "You are analyzing a screenshot from the game 'Phoenix Wright: Ace Attorney'. "
        "Your task is to extract key scene information for decision-making:\n"
        "- What is the current evidence name?\n"
        "- Is there any text or description in the evidence? \n"
        "- Any visual cues or context that hint at the intensity or purpose of this moment?\n\n"

        "Respond in this strict format:\n"
        "- Evidence: [Item Name or Unknown]\n"
        "- Text: [any text in the evidence]\n"
        "- Context: [Brief scene summary]"
    )


    
    # Call LLM API based on provider
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
    
    # Process response and format as structured board output
    structured_board = response.strip()
    
    # Generate final text output
    final_output = "\nScene analysis:\n" + structured_board
    update_history(final_output)

    return final_output


def phoenix_worker(system_prompt, api_provider, model_name, modality, thinking,
                   crop_left=0, crop_right=0, crop_top=0, crop_bottom=0,
                   grid_rows=7, grid_cols=7, prev_response="", evidence=False):
    """
    AI agent worker for Phoenix Wright: Ace Attorney.
    1) Waits, captures a screenshot of the current game state.
    2) Extracts scene information via an LLM.
    3) Decides the best single key press for next move.
    4) Logs key press and reasoning.
    5) Presses the key using PyAutoGUI.
    """
    # Wait before capturing the screen
    print("[INFO] Waiting 3 seconds before screenshot...")
    time.sleep(3)

    # Capture a screenshot of the current game state
    screen_width, screen_height = pyautogui.size()
    region = (0, 0, screen_width, screen_height)
    screenshot = pyautogui.screenshot(region=region)

    # Save screenshot
    os.makedirs(cache_dir, exist_ok=True)
    screenshot_path = os.path.join(cache_dir, "screenshot.png")
    screenshot.save(screenshot_path)
    if evidence == False:
        # Annotate and crop image
        annotate_image_path, grid_annotation_path, annotate_cropped_image_path= get_annotate_img(
            screenshot_path,
            crop_left=crop_left,
            crop_right=crop_right,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            grid_rows=1,
            grid_cols=1,
            cache_dir=cache_dir
        )
        phoenix_text_table = phoenix_read_worker(system_prompt, api_provider, model_name,
                                             annotate_cropped_image_path, modality, thinking)
    else:
        # Annotate and crop image
        annotate_image_path, grid_annotation_path, annotate_cropped_image_path= get_annotate_img(
            screenshot_path,
            crop_left=crop_left+50,
            crop_right=crop_right+50,
            crop_top=crop_top+100,
            crop_bottom=crop_bottom+200,
            grid_rows=1,
            grid_cols=1,
            cache_dir=cache_dir
        )
        phoenix_text_table = phoenix_read_evidence_worker(system_prompt, api_provider, model_name,
                                             annotate_cropped_image_path, modality, thinking)

    

    # Load recent history
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            recent_history = "".join(f.readlines()[-50:])
    else:
        recent_history = "No history yet."

    # Adjust prompt based on chapter
    allowed_keys = (
            "- 'Z': Confirm/Next text\n"
            "- 'B': Cancel/go back to dialogue\n"
            "- Arrow keys (up down left right): Navigate choices or evidence\n"
            "- 'R': Open Court Record, you may need to press left or right to view different evidence pages,\n"
            "- 'X': Present selected evidence to the court\n"
            "- 'L': Press the witness during cross-examination (ask for more detail). After pressing, you must press 'Z' to proceed.\n"
        )
    prompt = (
        "As an AI assistant playing Phoenix Wright, your goal is to decide the best next **single key press** "
        "based on the current game context.\n\n"
        f"Scene:\n{phoenix_text_table}\n\n"
        f"Available keys:\n{allowed_keys}\n\n"

        f"Recent History:\n{recent_history}\n\n"
        f"Previous response: {prev_response}\n\n"

        "**Rules:**\n"
        "- You may press 'R' to open the Court Record at any time to review evidence.\n"
        "- After pressing 'R', you can only use left or right to browse evidence,'B' to back to dialogue or press 'X' to present evidence **if the previous dialogue text color is green**.\n"
        "- After pressing 'L' to press a witness, press 'Z' to continue dialogue.\n"

        "Only choose one key. Respond in this format:\n"
        '**move: "KEY", thought: "Reasoning or justification"**\n\n'
        "Example:\n"
        '**move: "Z", thought: "Dialogue is ongoing; we should proceed to the next line."**'
        '**move: "R", thought: "I want to review current evidence and present the correct evidence to the court in next move"**'
    )


    start_time = time.time()

    print(f"[INFO] Calling {model_name} API...")

    # Call LLM
    if api_provider == "anthropic" and modality == "text-only":
        response = anthropic_text_completion(system_prompt, model_name, prompt, thinking)
    elif api_provider == "anthropic":
        response = anthropic_completion(system_prompt, model_name, None, prompt, thinking)
    elif api_provider == "openai" and "o3" in model_name and modality == "text-only":
        response = openai_text_reasoning_completion(system_prompt, model_name, prompt)
    elif api_provider == "openai":
        response = openai_completion(system_prompt, model_name, None, prompt)
    elif api_provider == "gemini" and modality == "text-only":
        response = gemini_text_completion(system_prompt, model_name, prompt)
    elif api_provider == "gemini":
        response = gemini_completion(system_prompt, model_name, None, prompt)
    elif api_provider == "deepseek":
        response = deepseek_text_reasoning_completion(system_prompt, model_name, prompt)
    else:
        raise NotImplementedError(f"API provider: {api_provider} is not supported.")

    latency = time.time() - start_time

    # Extract key and thought
    move_match = re.search(r'move:\s*"([A-Z])"', response)
    thought_match = re.search(r'thought:\s*"(.*?)"', response)

    if not move_match or not thought_match:
        log_output("phoenix_worker", f"[ERROR] Invalid LLM response: {response}", "phoenix")
        return

    key = move_match.group(1).lower()
    thought_text = thought_match.group(1)

    print(f"[INFO] Extracted key press: {key.upper()}")
    print(f"[INFO] LLM Thought: {thought_text}")

    valid_keys = {"z", "b", "c", "l", "r", "up", "down", "left", "right"}
    if key not in valid_keys:
        log_output("phoenix_worker", f"[ERROR] Invalid key received: {key}", "phoenix")
        return

    # Press the key
    print(f"[ACTION] Sending key press: {key}")

    pyautogui.keyDown(key)
    time.sleep(0.1)
    pyautogui.keyUp(key)

    # Log
    log_move_and_thought(key.upper(), thought_text, latency)
    log_output("phoenix_worker", f"[INFO] Key pressed: {key.upper()} | Thought: {thought_text} | Latency: {latency:.2f} sec", "phoenix")

    return response, key
