import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from chatbot import Chatbot

class RAG:
    """
    Diese Klasse kombiniert:
    - FAISS-basierte Vektorsuche für fehlerhafte Formulierungen
    - Retrieval-Augmented Generation (RAG) zur Textoptimierung
    """

    def __init__(
        self,
        db_folder,
        chatbot_system_prompt="Du bist ein hochqualifizierter Korrektor.",
        chatbot_user_prompt="Bitte verbessere den folgenden Text:"
    ):
        """
        :param db_folder: Ordner, in dem die FAISS-Index- und JSON-Dateien gespeichert werden.
        :param chatbot_system_prompt: System-Prompt für die Chatbot-Klasse.
        :param chatbot_user_prompt: Standard-User-Prompt für die Chatbot-Klasse.
        """
        self.db_folder = db_folder
        self.index = None
        self.error_dict = {}

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.chatbot_system_prompt = chatbot_system_prompt
        self.chatbot_user_prompt = chatbot_user_prompt

        # Dateipfade
        self.index_file_path = os.path.join(self.db_folder, "faiss_index.bin")
        self.json_file_path  = os.path.join(self.db_folder, "faiss_index.json")


    # ======================= DB INITIALISIERUNG =======================

    def initialize_db(self, error_corrections=None):
        """
        Prüft, ob bereits ein FAISS-Index existiert. Falls nicht:
        - build_index() mit den übergebenen error_corrections
        Falls ja:
        - load_index()

        :param error_corrections: dict - Start-Fehler-Korrektur-Paare
        """
        if os.path.exists(self.index_file_path) and os.path.exists(self.json_file_path):
            # Index existiert schon
            print("✅ Ein vorhandener FAISS-Index wurde gefunden. Lade Index...")
            self.load_index()
        else:
            # Index muss neu aufgebaut werden
            if error_corrections is None or not error_corrections:
                raise ValueError("❌ Keine 'error_corrections' übergeben, um eine neue DB zu bauen!")
            self.build_index(error_corrections)


    def build_index(self, error_corrections):
        """
        Legt den FAISS-Index komplett neu an und speichert ihn.

        :param error_corrections: dict - {Fehlerhafte Formulierung: Verbesserte Version}
        """
        print("🔨 Baue neuen FAISS-Index...")
        self.error_dict = error_corrections

        errors = list(self.error_dict.keys())
        embeddings = np.array([self.model.encode(e) for e in errors], dtype="float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        # Speichern
        faiss.write_index(self.index, self.index_file_path)
        with open(self.json_file_path, "w", encoding="utf-8") as f:
            json.dump(self.error_dict, f, ensure_ascii=False)

        print(f"✅ Neuer Index + JSON in '{self.db_folder}' erstellt.")


    def load_index(self):
        """
        Lädt die vorhandene FAISS-Datenbank und Fehler-Korrektur-Paare.
        """
        print("🔎 Lade vorhandenen FAISS-Index...")
        self.index = faiss.read_index(self.index_file_path)

        with open(self.json_file_path, "r", encoding="utf-8") as f:
            self.error_dict = json.load(f)

        print(f"✅ Index und Fehler-Korrekturen aus '{self.db_folder}' geladen.")


    # ======================= DB ERWEITERUNG =======================

    def add_entries(self, new_error_corrections):
        """
        Fügt weitere Fehler-Korrektur-Paare hinzu, ohne komplett neu zu starten.

        :param new_error_corrections: dict - z.B. {"Fehler...":"Korrektur..."}
        """
        # Stelle sicher, dass Index + error_dict geladen sind
        if self.index is None:
            # Prüfe, ob Files existieren, sonst kann man nicht "hinzufügen"
            if not os.path.exists(self.index_file_path) or not os.path.exists(self.json_file_path):
                raise FileNotFoundError("⚠️ Kein vorhandener Index! Bitte 'initialize_db()' oder 'build_index()' nutzen.")
            # Falls Index nicht im RAM ist, lade ihn
            self.load_index()

        # Merger: error_dict + new_error_corrections
        for f, c in new_error_corrections.items():
            if f not in self.error_dict:
                self.error_dict[f] = c

        # Berechne Embeddings nur für neue Einträge
        existing_errors = list(self.index_range())  # Verknüpft Index-Positionen und Keys
        existing_count = len(existing_errors)

        new_items = []
        for e in new_error_corrections.keys():
            # Nur hinzufügen, falls e nicht schon drin ist
            if e not in [k for k in self.error_dict if k != e]:
                new_items.append(e)

        if not new_items:
            print("ℹ️ Keine neuen Einträge zum Hinzufügen.")
            return

        # Embeddings für die neuen Items
        new_embeds = np.array([self.model.encode(e) for e in new_items], dtype="float32")

        # Index erweitern
        self.index.add(new_embeds)

        # Index & JSON überschreiben
        faiss.write_index(self.index, self.index_file_path)
        with open(self.json_file_path, "w", encoding="utf-8") as f:
            json.dump(self.error_dict, f, ensure_ascii=False)

        print(f"✅ {len(new_items)} neue Korrektur-Einträge hinzugefügt und Index aktualisiert.")


    def index_range(self):
        """
        Gibt die Liste (Position -> Key) zurück,
        um Kollisionen oder Überprüfungen machen zu können.
        """
        # Da wir rein "flat" und rein "append" arbeiten, index i -> key i
        # in sorted Reihenfolge? Um Kollisionen zu vermeiden, könnten wir
        # überlegen, wie man das sauber abbildet. Hier eine einfache Annahme:
        return enumerate(self.error_dict.keys())


    # ======================= RAG-ABLÄUFE =======================

    def query_index(self, text, top_k=3):
        """
        Sucht nach ähnlichen fehlerhaften Formulierungen in der Datenbank.
        """
        if self.index is None:
            print("⚠️ FAISS-Index nicht im Arbeitsspeicher. Lade existierende DB...")
            self.load_index()

        query_embedding = np.array([self.model.encode(text)], dtype="float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        all_errors = list(self.error_dict.keys())
        for i in range(top_k):
            if distances[0][i] < 0.6:
                err = all_errors[indices[0][i]]
                corr = self.error_dict[err]
                results.append((err, corr))

        return results

    def retrieve_context(self, seo_text):
        """
        Baut einen Kontext-String auf, basierend auf fehlerhaften Formulierungen aus seo_text.
        """
        context_lines = []
        sentences = seo_text.split(". ")
        for s in sentences:
            suggestions = self.query_index(s)
            for old, new in suggestions:
                context_lines.append(f"- Fehler: {old} ➝ Verbesserung: {new}")

        if context_lines:
            return "Hier sind bekannte Fehler und deren Korrekturen:\n" + "\n".join(context_lines)
        else:
            return "Keine bekannten Fehler gefunden."

    def check_text(self, seo_text):
        """
        Führt das RAG-Verfahren aus: Sucht Korrektur-Vorschläge und verbessert den Text.
        """
        retrieval_context = self.retrieve_context(seo_text)

        system_prompt_enriched = (
            f"{self.chatbot_system_prompt}\n"
            f"{retrieval_context}"
        )

        user_prompt = (
            f"{self.chatbot_user_prompt}\n"
            f"{seo_text}"
        )

        cb = Chatbot(systemprompt=system_prompt_enriched, userprompt=user_prompt)
        final_text = cb.chat_with_streaming()
        return final_text
