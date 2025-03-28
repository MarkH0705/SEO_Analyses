import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

class KeywordVisualizer:

    def __init__(self, df, output_dir="output"):
        self.searches_df = df
        self.output_dir = output_dir  # Speichert Bilder in diesem Ordner

    def should_use_log_scale(self, threshold=50):
        min_val = self.searches_df.replace(0, np.nan).min().min()
        max_val = self.searches_df.max().max()
        return max_val / max(1, min_val) > threshold  

    def heatmap(self):
        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, ax = plt.subplots(figsize=(12, 6))  # 🎯 Explizite Figure & Axes

        use_log = self.should_use_log_scale()

        if use_log:
            heatmap_data = np.log10(self.searches_df + 1)  
            cmap_type = "RdYlGn"
            print("ℹ️ Logarithmische Skalierung aktiviert.")
        else:
            heatmap_data = self.searches_df  
            cmap_type = "RdYlGn"
            print("ℹ️ Normale Farbgebung aktiviert.")

        real_min = max(self.searches_df.replace(0, np.nan).min().min(), 10)
        real_max = self.searches_df.max().max()

        ax = sns.heatmap(
            heatmap_data, 
            cmap=cmap_type, 
            annot=self.searches_df,  
            fmt=".0f",  
            linewidths=0.5,
            vmin=np.log10(real_min),  
            vmax=np.log10(real_max)   
        )

        plt.title("Google Ads Keyword-Suchvolumen (Heatmap)", fontsize=14, fontweight="bold")
        plt.xlabel("", fontsize=12)
        plt.ylabel("", fontsize=12)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        if use_log:
            colorbar = ax.collections[0].colorbar
            log_min = np.log10(real_min)  
            log_max = np.log10(real_max)  
            num_ticks = 3  
            log_ticks = np.linspace(log_min, log_max, num_ticks)  
            real_values = [round(10**tick) for tick in log_ticks]  
            colorbar.set_ticks(log_ticks)
            colorbar.set_ticklabels(real_values)

        # **🔥 Speichern der Heatmap als Bild**
        save_path = self.save_heatmap(fig)  # ⬅️ `fig` übergeben

        # **🔥 Jetzt die Heatmap auch anzeigen**
        plt.show()  # **Hier wird sie in Colab angezeigt!**
        
        return save_path  # Optional: Speicherpfad zurückgeben

    def save_heatmap(self, fig):
        """Speichert die Heatmap im Output-Ordner mit Timestamp."""
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m") # Hier ändern für timestamp
        filename = f"heatmap_{timestamp}.png"
        save_path = os.path.join(self.output_dir, filename)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")  # **Speichern**
        print(f"✅ Heatmap gespeichert als: {save_path}")

        return save_path  # Speicherpfad zurückgeben
