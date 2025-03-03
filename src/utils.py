import json
import os
from chatbot import Chatbot

def load_prompts(file_path):
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

def generate_llm_text_streaming(prompts, agents, **kwargs):
    system_prompt = prompts[agents]["system_prompt"]
    user_prompt = prompts[agents]["user_prompt"].format(**kwargs)

    cb = Chatbot(system_prompt, user_prompt)
    response = cb.chat_with_streaming()
    return response

def generate_llm_text(prompts, agents, **kwargs):
    system_prompt = prompts[agents]["system_prompt"]
    user_prompt = prompts[agents]["user_prompt"].format(**kwargs)

    cb = Chatbot(system_prompt, user_prompt)
    response = cb.chat()
    return response