import os
import json
from google.colab import userdata
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

class LangchainSEOPipeline:
    """
    Eine Klasse, die deinen LangChain-Workflow für SEO-Optimierung in sich vereint:
     1) Extrahiert Keywords (nur falls keine Google-Ads-Keywords vorhanden)
     2) Optimiert den Text für SEO
     3) Grammatik-Check
     4) Supervisor-Abschluss
     
    Alle Prompts werden aus JSON-Dateien im Ordner `prompts_folder` geladen.
    """

    def __init__(self, prompts_folder="data/prompts", google_ads_keywords=None):
        """
        :param prompts_folder: Ordner, in dem .json-Dateien liegen (z.B. extract_keywords.json, ...)
        :param google_ads_keywords: (Optional) Liste vorhandener Keywords (z.B. von Google Ads)
        """
        self.prompts_folder = prompts_folder
        self.google_ads_keywords = google_ads_keywords if google_ads_keywords else []

        # OpenAI-API-Key aus Colab userdata (oder eigener Mechanismus)
        os.environ['OPENAI_API_KEY'] = userdata.get('open_ai_api_key')
        
        # LangChain Chat-Modell
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini-2024-07-18"
            openai_api_key=os.environ['OPENAI_API_KEY'])

    def load_prompt(self, file_name: str) -> ChatPromptTemplate:
        """
        Lädt eine JSON-Datei (z.B. extract_keywords.json) aus dem prompts_folder
        und erstellt daraus ein ChatPromptTemplate.
        
        JSON-Struktur:
        {
          "system_prompt": "...",
          "user_prompt": "..."
        }
        """
        path = os.path.join(self.prompts_folder, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt-Datei '{path}' nicht gefunden!")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Erzeuge SystemMessage + HumanMessage
        system_content = data.get("system_prompt", "")
        user_content   = data.get("user_prompt", "")
        
        # Erzeuge ChatPromptTemplate
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_content),
            HumanMessage(content=user_content)
        ])
        return prompt_template

    def extract_keywords(self, original_text: str) -> str:
        """
        Lädt extract_keywords.json, formt die Messages und ruft das LLM auf.
        Gibt die gefundenen Keywords als String zurück.
        """
        prompt_template = self.load_prompt("extract_keywords.json")
        # Mit placeholders: {original_text}
        messages = prompt_template.format_messages(original_text=original_text)
        response = self.llm(messages)
        return response.content.strip()

    def optimize_text_for_seo(self, original_text: str, keywords: str) -> str:
        """
        Lädt optimize_seo.json, formt die Messages und ruft das LLM auf.
        Gibt den SEO-optimierten Text zurück.
        """
        prompt_template = self.load_prompt("optimize_seo.json")
        # placeholders: {original_text}, {keywords}
        messages = prompt_template.format_messages(
            original_text=original_text,
            keywords=keywords
        )
        response = self.llm(messages)
        return response.content.strip()

    def grammar_and_style_check(self, optimized_text: str) -> str:
        """
        Lädt grammar_check.json, ruft das LLM auf, gibt final bereinigten Text zurück.
        """
        prompt_template = self.load_prompt("grammar_check.json")
        # placeholder: {optimized_text}
        messages = prompt_template.format_messages(optimized_text=optimized_text)
        response = self.llm(messages)
        return response.content.strip()

    def supervisor_check(self, original_text: str, keywords: str, optimized_text: str, final_text: str) -> str:
        """
        Lädt supervisor.json, führt finalen QA-Check durch.
        """
        prompt_template = self.load_prompt("supervisor.json")
        # placeholders: {original_text}, {keywords}, {optimized_text}, {final_text}
        messages = prompt_template.format_messages(
            original_text=original_text,
            keywords=keywords,
            optimized_text=optimized_text,
            final_text=final_text
        )
        response = self.llm(messages)
        return response.content.strip()

    def run_pipeline(self, original_text: str) -> str:
        """
        Gesamtworkflow:
          1) Keywords extrahieren, wenn google_ads_keywords leer
          2) SEO-Optimierung
          3) Grammatik-/Stil-Check
          4) Supervisor-Freigabe
        """
        print("=== Langchain SEO Pipeline ===")

        # Schritt 1: Keywords check
        if not self.google_ads_keywords:
            print("-> Extrahiere Keywords via LLM (keine Google Ads Keywords vorhanden).")
            keywords = self.extract_keywords(original_text)
        else:
            print("-> Nutze vorhandene Google Ads Keywords, überspringe LLM-Keyword-Extraktion.")
            keywords = ", ".join(self.google_ads_keywords)

        print("Gefundene/Genutzte Keywords:", keywords)

        # Schritt 2: SEO-Optimierung
        optimized_text = self.optimize_text_for_seo(original_text, keywords)
        print("\nSEO-Optimierter Text:\n", optimized_text)

        # Schritt 3: Grammatik & Stil
        final_text = self.grammar_and_style_check(optimized_text)
        print("\nFinaler Text nach Lektorat:\n", final_text)

        # Schritt 4: Supervisor
        supervisor_feedback = self.supervisor_check(
            original_text=original_text,
            keywords=keywords,
            optimized_text=optimized_text,
            final_text=final_text
        )
        print("\nSupervisor Feedback:\n", supervisor_feedback)

        return final_text
