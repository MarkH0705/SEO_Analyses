# 🚀 SEO Automation Pipeline mit OpenAI & Retrieval (RAG)

Dieses Projekt bietet eine **komplette End-to-End-Pipeline für die SEO-Optimierung von Websites**, inklusive **Web-Scraping, SEO-Analyse, KI-gestützter Text-Optimierung und Qualitätskontrolle**.

Kern des Projekts sind **automatisierte Abläufe**, die von der **Datengewinnung bis zur SEO-optimierten Textgenerierung** reichen.
Mithilfe von **OpenAI (ChatGPT)** und einer **Retrieval Augmented Generation (RAG)-Architektur** wird sichergestellt, dass die finalen Texte nicht nur **SEO-freundlich**, sondern auch **grammatikalisch korrekt und hochwertig** sind.

## 📚 Inhaltsverzeichnis

- [Features](#features)
- [Projektstruktur](#projektstruktur)
- [Ablauf & Module](#ablauf--module)
- [Technologien](#technologien)
- [Installation](#installation)
- [Nutzung](#nutzung)
- [Ziele](#ziele)
- [Roadmap](#roadmap)

## ✅ Features

- 🌐 **Automatisiertes Web Scraping** (inkl. Filter für relevante Inhalte)
- ✍️ **Generierung von SEO-optimierten Texten** mithilfe der OpenAI API
- 🧠 **RAG-gestützte Fehlererkennung & Textkorrektur** mit Vektordatenbank (FAISS)
- 📊 **Analyse der Optimierungsergebnisse** (Statistiken, Ähnlichkeiten, Visualisierungen)
- 📈 **Keyword-Analyse und Keyword-Optimierung**
- 📦 Ausgabe in **HTML und PDF** für Kunden
- 📊 Umfangreiche **Datenvisualisierungen** (Wordclouds, Cosine Similarity, Keyword-Verteilung)

## 🗂️ Projektstruktur

```
SEO-Project/
├── data/                # Prompts, Fehler-Korrektur-Daten, weitere JSON Dateien
├── notebooks/           # Colab/Notebooks zum Starten und Entwickeln
├── src/                # Source Code (Python-Klassen und Module)
│   ├── webscraper.py    # Webscraping und Text-Extraktion
│   ├── llm_processor.py # Anbindung an OpenAI API, Keyword Extraktion
│   ├── chatbot.py       # Zentrale Chatbot-Klasse zur Kommunikation mit GPT
│   ├── seoanalyzer.py   # Analyse und Auswertung der Texte
│   ├── github.py        # Automatischer Upload ins GitHub Repo
│   ├── rag_checker.py   # RAG-Modul für Fehlerkorrektur via FAISS
│   ├── utils.py         # Hilfsmodule (z.B. für Prompt-Management)
│   └── embedding_demo.py# 3D Embedding- und Cosine Similarity Visualisierungen
└── requirements.txt    # Python-Abhängigkeiten
```

## ⚙️ Ablauf & Module

### 1. **Web Scraping**
- **src/webscraper.py**: Holt Inhalte von Webseiten, filtert irrelevante Seiten (z.B. Impressum, AGB).

### 2. **SEO-Optimierung mit OpenAI**
- **src/llm_processor.py**:
  - Extrahiert Keywords aus den Inhalten.
  - Optimiert die Texte für SEO mit gezielten Prompts.

### 3. **Fehlerkontrolle mit RAG**
- **src/rag_checker.py**: Erstellt eine Vektordatenbank mit bekannten Fehlern und Korrekturen. Erkennt fehlerhafte Formulierungen via Cosine Similarity und optimiert mit ChatGPT.

### 4. **Analyse & Visualisierung**
- **src/seoanalyzer.py**: Verarbeitet und analysiert die Original- und optimierten Texte.

### 5. **GitHub Automation**
- **src/github.py**: Lädt finale Ergebnisse in ein GitHub-Repo hoch.

## 🧰 Technologien

| Technologie                  | Beschreibung                                       |
|-----------------------------|---------------------------------------------------|
| Python                      | Hauptsprache                                       |
| OpenAI API (ChatGPT, GPT-4)  | Generative KI für SEO-Texte                       |
| FAISS                      | Vektorsuche für RAG und Text-Fehler                |
| Pandas, NumPy               | Datenanalyse und Verarbeitung                      |
| Matplotlib, Seaborn         | Visualisierungen                                   |
| Sentence Transformers       | Embedding-Erstellung für Vektordatenbank          |
| BeautifulSoup, Requests     | Webscraping                                        |
| Google Colab                | Entwicklung und Ausführung                        |

## 🚀 Installation

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
pip install faiss-cpu sentence-transformers openai wordcloud matplotlib seaborn
```

## 💻 Nutzung

```python
scraper = WebsiteScraper(start_url="https://www.example.com")
scraper.scrape_website()

llm_processor = LLMProcessor(PROJECT_ROOT, scraper.get_scraped_data())
llm_processor.run_all()

seo_checker = SEORAGChecker(quality_checker, chatbot_system_prompt)
final_text = seo_checker.check_text(optimized_text)

seo_analyzer = SEOAnalyzer(seo_json, original_texts, keywords_final)
seo_analyzer.run_analysis()
```

## 🎯 Ziele

- ✅ Vollständige Automatisierung der SEO-Optimierung
- ✅ RAG für sprachliche Qualitätskontrolle
- ✅ Kundenfertige PDF/HTML-Reports

## 🚧 Roadmap

- [ ] Automatische SEO Scores (z.B. Google Ads API)
- [ ] Automatische Keyword-Erweiterung
- [ ] Mehrsprachigkeit (aktuell Deutsch)
- [ ] WordPress-Integration

## 🤝 Zusammenarbeit

Contributions und Ideen willkommen!
👉 Pull-Request oder Issue eröffnen.
