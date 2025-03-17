# ✅ Pytest Template: Chatbot Klasse

## 1. Klassen & Methoden die getestet werden sollen

- **Chatbot**
  - `chat()`
  - `chat_with_streaming()`

---

## 2. Beispielhafte Inputs + erwartete Outputs pro Methode

| Methode                      | Beispiel Input                                                           | Erwartete Ausgabe    |
|-----------------------------|-------------------------------------------------------------------------|----------------------|
| `Chatbot.chat()`             | 'Das ist ein Test. Schreibe als Antwort "Test erfolgreich".'           | "Test erfolgreich"   |
| `Chatbot.chat_with_streaming()` | 'Das ist ein Test. Schreibe als Antwort "Test erfolgreich".'         | "Test erfolgreich"   |

---

## 3. Return-Typen der Methoden

| Methode                      | Rückgabe-Typ |
|-----------------------------|--------------|
| `Chatbot.chat()`             | `str`        |
| `Chatbot.chat_with_streaming()` | `str`     |

---

## 4. Externe Services mocken?

| Service         |  Mocken?                           |
|-----------------|-------------------------------------|
| OpenAI API      |  Nein                                |
| FAISS Index     |  Ja (kleine Test-Datenbank für FAISS) |

---

## 5. Ordnerstruktur für Tests

```bash
/project-root/
    /src/
        chatbot.py
    /tests/
        test_chatbot.py
    /logs/
        test_report.log
    ...
```

---
