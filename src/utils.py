import json
import os
from chatbot import Chatbot
import re

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

def extract_sections_to_json(keys, texts):
    """
    Extrahiert Abschnitte aus mehreren Texten und konvertiert sie in JSON.
    Gesucht werden die Überschriften 'Analyse', 'SEO', 'Erklärung' und der jeweils
    folgende Inhalt bis zur nächsten Überschrift oder zum Ende.
    """

    all_sections = []  # Liste für alle Abschnitte

    # Neues, robusteres Pattern:
    pattern = re.compile(
        r"(?m)^\s*(Analyse|SEO|Erklärung)\s*(?:\r?\n\s*)+"
        r"(.*?)(?=^\s*(?:Analyse|SEO|Erklärung)|\Z)",
        flags=re.DOTALL
    )

    for text in texts:
        sections_dict = {}
        matches = pattern.findall(text)
        for match in re.finditer(pattern, text):
            heading = match.group(1)
            content = match.group(2).strip()
            sections_dict[heading] = content

        all_sections.append(sections_dict)

    # Kombinieren der Abschnitte mit Keys
    final_json_data = {}
    for i, sections_dict in enumerate(all_sections):
        key = keys[i]  # Key aus der Liste holen
        final_json_data[key] = sections_dict  # Abschnitte zum Dictionary hinzufügen

    json_data = json.dumps(final_json_data, indent=4, ensure_ascii=False)
    return json_data