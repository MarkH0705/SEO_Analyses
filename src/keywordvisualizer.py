import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class KeywordVisualizer:
    def __init__(self, df, output_dir="output", image_paths=None):
        """
        :param df: DataFrame mit Keywords als Zeilen und Monaten als Spalten
        :param output_dir: Ausgabeordner für gespeicherte Plots
        :param image_paths: Dictionary für alle generierten Bildpfade (kann von außen als shared dict übergeben werden)
        """
        self.searches_df = df
        self.output_dir = output_dir  
        os.makedirs(self.output_dir, exist_ok=True)

        # Falls der Nutzer bereits ein gemeinsames Dict führt, nutzen wir das. Sonst eigenes.
        self.image_paths = image_paths if image_paths is not None else {}

    def should_use_log_scale(self, threshold=50):
        """
        Prüft, ob wir eine log-Skalierung anwenden (großer Wertebereich),
        z.B. wenn max_val / min_val > threshold.
        """
        min_val = self.searches_df.replace(0, np.nan).min().min()
        max_val = self.searches_df.max().max()
        return max_val / max(1, min_val) > threshold  

    def heatmap(self):
        """
        Erstellt eine Heatmap für das Keyword-Suchvolumen.
        Gibt den Pfad zum gespeicherten Bild zurück.
        """
        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, ax = plt.subplots(figsize=(12, 6))  # Explizite Figure & Axes

        use_log = self.should_use_log_scale()

        if use_log:
            heatmap_data = np.log10(self.searches_df + 1)  
            cmap_type = "RdYlGn"
            print("ℹ️ Logarithmische Skalierung aktiviert.")
        else:
            heatmap_data = self.searches_df  
            cmap_type = "RdYlGn"
            print("ℹ️ Normale Farbgebung aktiviert.")

        # Echten Wertebereich für die log-Skala bestimmen
        real_min = max(self.searches_df.replace(0, np.nan).min().min(), 10)
        real_max = self.searches_df.max().max()

        # Heatmap plotten
        sns.heatmap(
            heatmap_data, 
            cmap=cmap_type, 
            annot=self.searches_df,  
            fmt=".0f",  
            linewidths=0.5,
            vmin=np.log10(real_min) if use_log else None,  
            vmax=np.log10(real_max) if use_log else None,
            ax=ax
        )

        plt.title("Google Ads Keyword-Suchvolumen (Heatmap)", fontsize=14, fontweight="bold")
        plt.xlabel("")
        plt.ylabel("")

        # X-Achse beschriften, gedreht
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        # Farbskala anpassen, falls log
        if use_log:
            colorbar = ax.collections[0].colorbar
            log_min = np.log10(real_min)  
            log_max = np.log10(real_max)  
            num_ticks = 3  
            log_ticks = np.linspace(log_min, log_max, num_ticks)  
            real_values = [round(10**tick) for tick in log_ticks]  
            colorbar.set_ticks(log_ticks)
            colorbar.set_ticklabels(real_values)

        # Plot speichern
        heatmap_path = self._save_heatmap(fig)
        plt.show()

        # Pfad in unser Dictionary speichern
        self.image_paths["keyword_heatmap"] = heatmap_path

        return heatmap_path

    def _save_heatmap(self, fig):
        """
        Speichert die Heatmap im Output-Ordner mit Timestamp.
        Gibt den Dateipfad zurück.
        """
        timestamp = datetime.now().strftime("%Y")
        filename = f"heatmap_{timestamp}.png"
        save_path = os.path.join(self.output_dir, filename)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Heatmap gespeichert als: {save_path}")
        return save_path

    def get_image_paths(self):
        """
        Falls du direkt aus diesem Visualizer alle Pfade abrufen willst.
        """
        return self.image_paths
