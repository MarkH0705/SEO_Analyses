import json
import os
from chatbot import Chatbot

def load_prompts(root, file_name):
    file_path = os.path.join(root, file_name)

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            prompts = json.load(file)
        return prompts
    except FileNotFoundError:
        print(f"❌ Fehler: Die Datei {file_name} wurde nicht gefunden.")
        return {}
    except json.JSONDecodeError:
        print(f"❌ Fehler: Die Datei {file_name} enthält ungültiges JSON.")
        return {}

def generate_llm_text(prompts, agents, **kwargs):
    system_prompt = prompts[agents]["system_prompt"]
    user_prompt = prompts[agents]["user_prompt_template"].format(**kwargs)

    cb = Chatbot(system_prompt, user_prompt)
    response = cb.chat()
    return response