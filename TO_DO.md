# To-Do Liste: SEO Automation & KI-Projekt

Diese Liste fasst alle anstehenden Aufgaben im Projekt zusammen

---

## 1. **Allgemeine Projektorganisation**
- [ ] **Projektstruktur verbessern**: Ordner übersichtlich gestalten (z.B. `src/`, `data/`, `tests/`, `notebooks/`).
- [ ] **Dokumentation erweitern**: README und Wiki (bzw. GitHub Pages) zu jedem Modul anlegen.
- [ ] **Automatisierte Tests** (z.B. Pytest) für Kernfunktionen einrichten.

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
- [ ] **GitHub-Repo** publizieren (Entweder privat oder als Public Demo)
- [ ] **Optionale WordPress-Integration** in der Zukunft (Ideenspeicher)
  - [ ] Upload via REST API
  - [ ] Metadaten (Title, Slug, Tags etc.)

---

## 6. **Nächste Schritte**
- [ ] **Liste priorisieren**: Welche Punkte sind dringlich? Was kann warten?
- [ ] **Zeitplan erstellen**: Welche Teilaufgaben, bis wann?
- [ ] **Issues anlegen**: Jedes To-Do hier als eigenes Issue im GitHub-Repository, um Fortschritt zu tracken.
