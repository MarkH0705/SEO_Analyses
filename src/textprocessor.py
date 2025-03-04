
import json
import re

class TextProcessor:
    """
    Diese Klasse verarbeitet Texte: Sie bereinigt Texte und extrahiert strukturierte Abschnitte.
    """

    def __init__(self):
        pass  # Kein spezieller Initialisierungsprozess nötig

    @staticmethod
    def clean_text(text):
        """
        Bereinigt den Text, indem unnötige Zeichen entfernt und Formatierungen standardisiert werden.
        """
        text = text.replace('\n', ' ')  # Zeilenumbrüche durch Leerzeichen ersetzen
        text = re.sub(r'[^a-zA-Z0-9äöüÄÖÜß.,!?;:\-\s]', '', text)  # Unerwünschte Zeichen entfernen
        text = re.sub(r'\s+', ' ', text)  # Mehrere Leerzeichen zu einem zusammenführen
        return text.strip()  # Führende und nachfolgende Leerzeichen entfernen

    @staticmethod
    def extract_sections_to_json(keys, texts):
        """
        Extrahiert Abschnitte ('Analyse', 'SEO', 'Erklärung') aus den Texten und speichert sie als JSON.
        """
        all_sections = []
        pattern = re.compile(
            r"(?m)^\s*(Analyse|SEO|Erklärung)\s*(?:\r?\n\s*)+"
            r"(.*?)(?=^\s*(?:Analyse|SEO|Erklärung)|\Z)",
            flags=re.DOTALL
        )

        for text in texts:
            sections_dict = {}
            matches = pattern.findall(text)
            for match in matches:
                heading = match[0]
                content = match[1].strip()
                sections_dict[heading] = content

            all_sections.append(sections_dict)

        # Kombinieren der Abschnitte mit Keys
        final_json_data = {keys[i]: sections for i, sections in enumerate(all_sections)}

        return json.dumps(final_json_data, indent=4, ensure_ascii=False)

    @staticmethod
    def add_cleaned_text(seo_json, original_texts):
        """
        Fügt die bereinigten Originaltexte als 'alt' zum JSON hinzu und reinigt auch die anderen Abschnitte.
        """
        for i, (url, url_content) in enumerate(seo_json.items()):
            url_content["alt"] = TextProcessor.clean_text(list(original_texts.values())[i])
            url_content["SEO"] = TextProcessor.clean_text(url_content.get("SEO", ""))
            url_content["Erklärung"] = TextProcessor.clean_text(url_content.get("Erklärung", ""))
            url_content["Analyse"] = TextProcessor.clean_text(url_content.get("Analyse", ""))
        return seo_json
