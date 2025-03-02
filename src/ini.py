import os
import sys
import subprocess
from google.colab import drive, userdata

# Google Drive mounten
drive.mount('/content/drive', force_remount=True)

# Definiere den richtigen Pfad zum Projekt-Root
PROJECT_ROOT = userdata.get("gdrive_seo_root")
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

# Stelle sicher, dass src/ als Modul erkannt wird
os.makedirs(SRC_PATH, exist_ok=True)
open(f"{SRC_PATH}/__init__.py", 'a').close()  # Erstellt/leert __init__.py

# sys.path aktualisieren
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Installiere Abhängigkeiten
requirements_file = f"{PROJECT_ROOT}/requirements.txt"
subprocess.run(["pip", "install", "-r", requirements_file], check=True)
subprocess.run(["python", "-m", "spacy", "download", "de_core_news_sm"], check=True)
subprocess.run(["playwright", "install"], check=True)
subprocess.run(["apt-get", "update"], check=True)
subprocess.run(["apt-get", "install", "-y", "pandoc"], check=True)

# Importiere dependencies
import dependencies
