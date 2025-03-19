# To-Do Liste: SEO Automation & KI-Projekt

Diese Liste fasst alle anstehenden Aufgaben im Projekt zusammen

---

## 0. **Aktuelles und dringendes**
- [ ] 18.3.2025 **Version missmatch**: numpy 2.2.3 und pandas 2.2.4 **side effects on**: dependencies.py, excelimporter.py, Installation.ipynb

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

---

## 2. **Vector-Datenbank (FAISS) & Retrieval**
- [ ] **VectorDB-Klasse finalisieren**:
  - [ ] Kleinere Bugs beheben
  - [ ] Userfreundliche Methoden für neue Einträge
- [ ] **Einrichtung der DB** bei Projektstart (Neubau vs. Laden) vereinheitlichen
- [ ] **Konfigurierbare Ähnlichkeits-Schwelle** (z.B. `threshold=0.6`) besser dokumentieren
- [ ] **Dynamische Filter** für bestimmte Fehlerkategorien (z.B. Stil vs. Grammatik) überlegen

---

## 3. **SEO-Optimierungs-Pipeline (LangChain)**
- [ ] **LangChain-Workflow** debuggen und lauffähig machen
  - [ ] Placeholder & Prompt-Mapping für `format_messages(...)` beheben
  - [ ] `.role` vs. `_role`-Konflikt lösen (Debug-Statements anpassen)
- [ ] **Prompts in JSON-Dateien** verlagern (z.B. `/data/prompts/`) und sauber verlinken
- [ ] **Google Ads Keywords** Integration:
  - [ ] Nur LLM-Keyword-Extraktion aufrufen, wenn keine Google-Keywords vorliegen
- [ ] **Supervisor-Feedback** integrieren (optional) & QA-Schritte definieren

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
