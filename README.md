# 🚀 SEO Automation Pipeline mit OpenAI & Retrieval (RAG)

Dieses Projekt bietet eine **komplette End-to-End-Pipelines für die SEO-Optimierung von Websites**, inklusive **Web-Crawling, SEO-Analyse, KI-gestützter Text-Optimierung und Qualitätskontrolle**.

Kern des Projekts sind **automatisierte Abläufe**, die von der **Datengewinnung bis zur SEO-optimierten Textgenerierung** reichen.
Es wird eine pipeline zur Text-Optimierung Mithilfe von **OpenAI (ChatGPT)** hergestellt. Die Textelemente eines HTML Dokumentes werden mit ChatGPT SEO optimiert und danach wieder in den code eingebaut.
Der Kunde erhält eine Datei als Vorschau auf seine SEO optimierte website.

## 📚 Inhaltsverzeichnis

- Features
- Projektstruktur
- Ablauf & Module
- Technologien
- Installation
- Nutzung
- Ziele
- Roadmap

## ✅ Features

- 🌐 **Automatisiertes Web Crawling** (inkl. Filter für relevante Inhalte)
- ✍️ **Generierung von SEO-optimierten Texten** mithilfe der OpenAI API
- 🧠 **RAG-gestützte Fehlererkennung & Textkorrektur** mit Vektordatenbank (FAISS)
- 📊 **Analyse der Optimierungsergebnisse** (Statistiken, Ähnlichkeiten, Visualisierungen)
- 📈 **Keyword-Analyse und Keyword-Optimierung**
- 📦 Ausgabe in **HTML und PDF** für Kunden
- 📊 Umfangreiche **Datenvisualisierungen** (Wordclouds, Cosine Similarity, Keyword-Verteilung)




<img src="https://drive.google.com/uc?id=10oR2bcugvN2MClp14ia7gnzMGX5b896t" alt="SEO Heatmap" width="600">




## 🗂️ Projektstruktur

```
SEO-Project/
├── data/                # Prompts, Fehler-Korrektur-Daten, weitere JSON Dateien
├── notebooks/           # Colab/Notebooks zum Starten und Entwickeln
├── output/              # Erzeugte Dateien (HTML, PDF, Bilder)
│   ├── final           # Dokumente für Kunden (HTML, PDF)
│   └── images          # Visualisierungen
├── src/                # Source Code (Python-Klassen und Module)
│   ├── webscraper.py    # Webscrawling und Text-Extraktion
│   ├── llmprocessor.py # Anbindung an OpenAI API, Keyword Extraktion
│   ├── chatbot.py       # Zentrale Chatbot-Klasse zur Kommunikation mit GPT
│   ├── seoanalyzer.py   # Analyse und Auswertung der Texte
│   ├── github.py        # Automatischer Upload ins GitHub Repo
│   ├── utils.py         # Hilfsmodule (z.B. für Prompt-Management)
│   └── embeddingdemo.py# 3D Embedding- und Cosine Similarity Visualisierungen
├── tests/              # pytest der Hauptfunktionalitäten
└── requirements.txt    # Python-Abhängigkeiten
```

## ⚙️ Ablauf & Module

### 1. **Web Crawling**
- **src/webscraper.py**: Holt Inhalte von Webseiten, filtert irrelevante Seiten (z.B. Impressum, AGB).

### 2. **SEO-Optimierung mit OpenAI**
- **src/llmprocessor.py**:
  - Extrahiert Keywords aus den Inhalten.
  - Optimiert die Texte für SEO mit gezielten Prompts.

### 3. **Analyse & Visualisierung**
- **src/seoanalyzer.py**: Verarbeitet und analysiert die Original- und optimierten Texte.

### 4. **GitHub Automation**
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
| BeautifulSoup, Requests     | Webcrawling                                        |
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

llm_processor = LLMProcessor(prompts_folder, get_filtered_texts, google_ads_keywords)
llm_processor.run_all()

seo_analyzer = SEOAnalyzer(seo_json, original_texts, keywords_final)
seo_analyzer.run_analysis()
```

## 🎯 Ziele

- ✅ Vollständige Automatisierung der SEO-Optimierung
- ✅ RAG für sprachliche Qualitätskontrolle
- ✅ Kundenfertige PDF/HTML-Reports

## 🚧 Roadmap

- [ ] **Produkt für Kunden finalisieren:** all-in-one solution für webcrawl + SEO + optimierten html code
- [ ] Automatische SEO Scores (z.B. Google Ads API)
- [ ] Automatische Keyword-Erweiterung
- [ ] Mehrsprachigkeit
- [ ] WordPress-Integration
