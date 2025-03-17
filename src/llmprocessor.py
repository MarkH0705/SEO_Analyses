import json
import os
from chatbot import Chatbot


class LLMProcessor:
    """
    Diese Klasse verwaltet alle Anfragen an das LLM und führt die Analyse & Optimierung durch.
    """

    def __init__(self, prompt_path, filtered_texts, google_ads_keywords=None):
        """
        :param project_root: Der Hauptpfad des Projekts.
        :param filtered_texts: Das Dictionary {URL: Text}, das optimiert werden soll.
        :param google_ads_keywords: (Optional) Liste von Google Ads Keywords für die SEO-Optimierung.
        """
        self.prompt_path = prompt_path
        self.filtered_texts = filtered_texts
        self.google_ads_keywords = google_ads_keywords if google_ads_keywords else []

        # Prompts laden
        self.analysis_prompts = self.load_prompts(os.path.join(self.prompt_path, "analysis_prompts.json"))
        self.seo_prompts = self.load_prompts(os.path.join(self.prompt_path, "seo_prompts.json"))

        # Ergebnisse
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
        return cb.chat()

    def generate_llm_text_streaming(self, prompts, agents, **kwargs):
        """Generiert eine Streaming-Antwort vom LLM mit den gegebenen Prompts."""
        system_prompt = prompts[agents]["system_prompt"]
        user_prompt = prompts[agents]["user_prompt"].format(**kwargs)

        cb = Chatbot(system_prompt, user_prompt)
        return cb.chat_with_streaming()

    def extract_or_use_google_keywords(self):
        """Extrahiert die Keywords oder nutzt die bereitgestellten Google Ads Keywords."""
        if self.google_ads_keywords:
            print("✅ Google Ads Keywords werden verwendet. LLM Keyword Extraktion wird übersprungen.")
            self.keywords_final = self.google_ads_keywords
        else:
            print("🔍 Starte Keyword Extraktion via LLM...")
            self.keywords_raw = [
                self.generate_llm_text(self.analysis_prompts, 'keyword_extraction', input_text_1=text)
                for text in self.filtered_texts.values()
            ]
            print("✨ Starte Keyword Optimierung via LLM...")
            self.keywords_final = self.generate_llm_text(
                self.analysis_prompts, 'keyword_optimization', input_text_1=self.keywords_raw
            )

    def get_keyword_city(self):
        """Ermittelt die für SEO relevante Stadt."""
        print("📍 Stadt für SEO-Kontext wird via LLM ermittelt...")
        self.keyword_city = self.generate_llm_text(
            self.analysis_prompts, 'keyword_city', input_text_1=list(self.filtered_texts.values())
        )

    def perform_seo_analysis(self):
        """Führt die SEO-Optimierung für alle URLs durch und speichert die Ergebnisse."""
        print("🚀 Starte SEO-Optimierung mit LLM...")
        for url, text in self.filtered_texts.items():
            print(f"\n=== SEO-Optimierung für {url} ===")
            optimized_text = self.generate_llm_text_streaming(
                self.seo_prompts,
                'seo_optimization',
                input_text_1=text,
                stadt=self.keyword_city,
                keywords_final=self.keywords_final
            )
            self.combined_analysis_dict[url] = optimized_text

    def get_keywords(self):
        """Gibt die verwendeten Keywords zurück (inkl. Quelle)."""
        return {
            "keywords_source": "Google Ads" if self.google_ads_keywords else "LLM Generated",
            "keywords_raw": self.keywords_raw,
            "keywords_final": self.keywords_final,
            "keyword_city": self.keyword_city
        }

    def run_all(self):
        """
        Führt alle Verarbeitungen aus:
        - Keyword-Management (Google Ads oder LLM)
        - Stadterkennung
        - SEO-Optimierung
        """
        self.extract_or_use_google_keywords()
        self.get_keyword_city()
        self.perform_seo_analysis()

        return self.combined_analysis_dict  # Gibt die finalen SEO-optimierten Texte zurück
