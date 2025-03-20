import os
import pandas as pd
import numpy as np
from google.colab import drive

class ExcelImporter:
    """
    Lädt alle .xls/.xlsx-Dateien aus dem angegebenen Projektordner und kombiniert sie in einem DataFrame.
    """

    def __init__(self, project_folder, header=0):
        """
        :param project_folder: Pfad zum Ordner in Google Drive
        :param header: Zeilennummer, ab der Spaltennamen gelesen werden (Default = 0)
        """
        self.project_folder = project_folder
        self.header = header
        self.dataframes = []

    def find_excel_files(self):
        excel_files = []
        for root, _, files in os.walk(self.project_folder):
            for file in files:
                if file.endswith('.xls') or file.endswith('.xlsx'):
                    excel_files.append(os.path.join(root, file))
        return excel_files

    def load_excel_files(self):
        excel_files = self.find_excel_files()
        if not excel_files:
            print("⚠️ Keine Excel-Dateien gefunden!")
            return None

        for file in excel_files:
            try:
                # Hier setzen wir das 'header=' entsprechend deiner Vorgabe:
                df = pd.read_excel(
                    file, 
                    engine="openpyxl" if file.endswith(".xlsx") else None,
                    header=self.header
                )
                df["Source_File"] = os.path.basename(file)  # optional
                self.dataframes.append(df)
                print(f"✅ Geladen: {file}")
            except Exception as e:
                print(f"❌ Fehler beim Laden von {file}: {e}")

    def merge_dataframes(self):
        if not self.dataframes:
            print("⚠️ Keine Daten zum Zusammenführen!")
            return None
        return pd.concat(self.dataframes, ignore_index=True)


    def import_all(self):
        self.load_excel_files()
        combined_df = self.merge_dataframes()
        return combined_df
