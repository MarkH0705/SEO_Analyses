import json
import os
import re
from chatbot import Chatbot

class LLMProcessor:
    """
    Diese Klasse verwaltet alle Anfragen an das LLM und führt die Analyse & Optimierung durch.
    """

    def __init__(self, project_root, filtered_texts):
        """
        :param project_root: Der Hauptpfad des Projekts
        :param filtered_texts: Das Dictionary {URL: Text}, das optimiert werden soll
        """
        self.project_root = project_root
        self.filtered_texts = filtered_texts

        # Prompts laden
        self.analysis_prompts = self.load_prompts(os.path.join(project_root, "data/analysis_prompts.json"))
        self.seo_prompts = self.load_prompts(os.path.join(project_root, "data/seo_prompts.json"))

        # Ergebnisse speichern
        self.keywords_raw = []
        self.keywords_final = None
        self.keyword_city = None
        self.combined_analysis_dict = {}

    @staticmethod
    def load_prompts(file_path):
        """Lädt eine JSON-Datei mit Prompts."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                prompts = json.load(file)
            return prompts
        except FileNotFoundError:
            print(f"❌ Fehler: Die Datei {file_path} wurde nicht gefunden.")
            return {}
        except json.JSONDecodeError:
            print(f"❌ Fehler: Die Datei {file_path} enthält ungültiges JSON.")
            return {}

    def generate_llm_text(self, prompts, agents, **kwargs):
        """Generiert eine Standardantwort vom LLM mit den gegebenen Prompts."""
        system_prompt = prompts[agents]["system_prompt"]
        user_prompt = prompts[agents]["user_prompt"].format(**kwargs)

        cb = Chatbot(system_prompt, user_prompt)
        response = cb.chat()
        return response

    def generate_llm_text_streaming(self, prompts, agents, **kwargs):
        """Generiert eine Streaming-Antwort vom LLM mit den gegebenen Prompts."""
        system_prompt = prompts[agents]["system_prompt"]
        user_prompt = prompts[agents]["user_prompt"].format(**kwargs)

        cb = Chatbot(system_prompt, user_prompt)
        response = cb.chat_with_streaming()
        return response

    def extract_keywords(self):
        """Extrahiert die Keywords aus den gefilterten Texten."""
        self.keywords_raw = [
            self.generate_llm_text(self.analysis_prompts, 'keyword_extraction', input_text_1=text)
            for text in self.filtered_texts.values()
        ]

    def optimize_keywords(self):
        """Optimiert die extrahierten Keywords."""
        self.keywords_final = self.generate_llm_text(
            self.analysis_prompts, 'keyword_optimization', input_text_1=self.keywords_raw
        )

    def get_keyword_city(self):
        """Ermittelt die für SEO relevante Stadt."""
        self.keyword_city = self.generate_llm_text(
            self.analysis_prompts, 'keyword_city', input_text_1=list(self.filtered_texts.values())
        )

    def perform_seo_analysis(self):
        """Führt die SEO-Optimierung für alle URLs durch und speichert sie in `combined_analysis_dict`."""
        for url, text in self.filtered_texts.items():
            print(f"\n=== Analyzing {url} ===")

            self.combined_analysis_dict[url] = self.generate_llm_text_streaming(
                self.seo_prompts,
                'seo_optimization',
                input_text_1=text,
                stadt=self.keyword_city,
                keywords_final=self.keywords_final
            )

    def get_keywords(self):
        """
        Gibt die exakten Keywords zurück, die für die SEO-Optimierung verwendet wurden.
        """
        return {
            "keywords_raw": self.keywords_raw,
            "keywords_final": self.keywords_final,
            "keyword_city": self.keyword_city
        }

    def run_all(self):
        """Führt alle Verarbeitungsschritte nacheinander aus."""
        self.extract_keywords()
        self.optimize_keywords()
        self.get_keyword_city()
        self.perform_seo_analysis()

        return self.combined_analysis_dict  # Gibt die optimierten Texte als Dictionary zurück
