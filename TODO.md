# To-Do Liste: SEO Automation & KI-Projekt

Diese Liste fasst alle anstehenden Aufgaben im Projekt zusammen

---

## 0. **Aktuelles und dringendes**
- [ ] 18.3.2025 **Version missmatch**: numpy 2.2.3 und spacy 3.5 **side effects on**: dependencies.py, excelimporter.py, Installation.ipynb **ursachen**: spacy version 3.5 verlangt numpy 1.26.4 -> version missmatch
      -> 23.5. class SEOAnlyzer refaktoriert und spacy ersetzt
- [ ] 24.3.25 class **webscraper** meta titel?
- [ ] 28.3.25 lokale SEO keywords liefern manchmal falsche Stadt

---

## 1. **Allgemeine Projektorganisation**
- [ ] **Projektstruktur verbessern**: Ordner übersichtlich gestalten (z.B. `src/`, `data/`, `tests/`, `notebooks/`, dependencies.py).
- [ ] **Dokumentation erweitern**: READ_ME und Wiki (bzw. GitHub Pages) zu jedem Modul anlegen.
- [ ] **Automatisierte Tests** Pytest für Kernfunktionen ausbauen.
- [ ] **Produkt für Kunden finalisieren**
- [ ] **FAISS DB**: automatisierte Erweiterung bei neu gefundenen Fehlern
- [ ] **Template GitHub**: issues
- [ ] Funktionalitäten aus **utils.py** überdenken
- [ ] langfristig Umstieg auf **langchain**
- [ ] textprocessor durch openai **function calling** ersetzen
- [ ] **dependencies** und versionen robuster machen
- [ ] **bug reporting system** einrichten

---

## 2. **Vector-Datenbank (FAISS) & Retrieval**
- [ ] **VectorDB-Klasse finalisieren**:
  - [ ] Kleinere Bugs beheben
  - [ ] Userfreundliche Methoden für neue Einträge
- [ ] **Einrichtung der DB** bei Projektstart (Neubau vs. Laden) vereinheitlichen
- [ ] **Konfigurierbare Ähnlichkeits-Schwelle** (z.B. `threshold=0.6`) besser dokumentieren
- [ ] **Dynamische Filter** für bestimmte Fehlerkategorien (z.B. Stil vs. Grammatik) überlegen
- [ ] **hybrides system mit knowledge tree und RAG** etablieren

---

## 3. **SEO-Optimierungs-Pipeline (LangChain)**
- [ ] **Prompts in JSON-Dateien** verlagern (z.B. `/data/prompts/`) und sauber verlinken
- [ ] **Supervisor-Feedback** integrieren & QA-Schritte definieren

---

## 4. **SEOGrammarChecker & PromptManager**
- [ ] Klassenrefactoring:
  - [ ] **`VectorDB`** vs. **`PromptManager`** vs. **`SEOGrammarChecker`** sauber trennen
  - [ ] Möglichst wenig Code-Duplikate, mehr modulare Testbarkeit
- [ ] **Konfigurationsdatei** (z.B. YAML) für Pfade, wie `FAISS_PATH` & Promptordner
- [ ] **Erweiterbare Prompt-Templates**:
  - [ ] Z.B. `seo_optimization.json`, `grammar_check.json`, `supervisor.json`, etc.

---

## 5. **Abschluss & Integration**
- [ ] **Dokumentation** aller Pipelines & Klassen in der README (oder in separater Doku)
- [ ] **Optionale WordPress-Integration** in der Zukunft (Ideenspeicher)
  - [ ] Upload via REST API
  - [ ] Metadaten (Title, Slug, Tags etc.)
