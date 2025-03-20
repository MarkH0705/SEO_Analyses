import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class KeywordVisualizer:

    def __init__(self, df):
        self.searches_df = df

    def should_use_log_scale(self, threshold=50):
        min_val = self.searches_df.replace(0, np.nan).min().min()
        max_val = self.searches_df.max().max()
        return max_val / max(1, min_val) > threshold  

    def heatmap(self):
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.figure(figsize=(12, 6))

        use_log = self.should_use_log_scale()

        if use_log:
            heatmap_data = np.log10(self.searches_df + 1)  
            cmap_type = "RdYlGn"
            print("ℹ️ Logarithmische Skalierung aktiviert.")
        else:
            heatmap_data = self.searches_df  
            cmap_type = "RdYlGn"
            print("ℹ️ Normale Farbgebung aktiviert.")

        # **Setze die Farbrange exakt auf min/max der echten Daten**
        real_min = max(self.searches_df.replace(0, np.nan).min().min(), 10)
        real_max = self.searches_df.max().max()

        # 🎨 **Fix: Seaborn zwingen, die volle Farbskala zu nutzen!**
        ax = sns.heatmap(
            heatmap_data, 
            cmap=cmap_type, 
            annot=self.searches_df,  
            fmt=".0f",  
            linewidths=0.5,
            vmin=np.log10(real_min),  # **Manuelle Untergrenze**
            vmax=np.log10(real_max)   # **Manuelle Obergrenze**
        )

        plt.title("Google Ads Keyword-Suchvolumen (Heatmap)", fontsize=14, fontweight="bold")
        plt.xlabel("", fontsize=12)
        plt.ylabel("", fontsize=12)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        # 🎨 **Colorbar perfekt anpassen**
        if use_log:
            colorbar = ax.collections[0].colorbar

            # **1️⃣ Berechne log-Werte für Min/Max**
            log_min = np.log10(real_min)  
            log_max = np.log10(real_max)  

            # **2️⃣ Setze exakte Ticks für log-Skala**
            num_ticks = 3  
            log_ticks = np.linspace(log_min, log_max, num_ticks)  
            real_values = [round(10**tick) for tick in log_ticks]  

            # **3️⃣ Wende die korrekten Ticks an**
            colorbar.set_ticks(log_ticks)
            colorbar.set_ticklabels(real_values)

        plt.show()
