import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer
import nltk

nltk.download('stopwords')

class SEOAnalyzer:
    def __init__(
        self,
        seo_json,
        keywords_final,
        output_dir="",
        historical_data=None,
        wordcloud_exclude=None,
        shared_image_dict={}
    ):
        """
        :param seo_json: JSON mit den optimierten Texten + alten Texten
        :param keywords_final: Liste der finalen Keywords
        :param output_dir: Ausgabeordner für gespeicherte Plots (PNG)
        :param historical_data: Optionale historische Daten
        :param wordcloud_exclude: Zusätzliche Wörter, die aus Wordclouds gefiltert werden
        """
        self.seo_json = seo_json
        self.keywords_final = keywords_final
        self.original_texts_list_clean = [seo_json[key]['original_text'] for key in seo_json]
        self.optimized_texts_list_clean = [seo_json[key]['optimized_text'] for key in seo_json]

        # Falls None, nimm leere Liste
        self.wordcloud_exclude = wordcloud_exclude if wordcloud_exclude else []

        self.stop_words = set(stopwords.words('german'))
        self.stemmer = GermanStemmer()

        self.df_metrics = self.load_historical_data(historical_data)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Neu: Alle Bild-Pfade werden hier gesammelt
        self.image_paths = shared_image_dict

    @staticmethod
    def load_historical_data(historical_data):
        if historical_data is None:
            return None
        df = pd.DataFrame(historical_data)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def save_plot(self, fig, filename):
        """
        Speichert einen Matplotlib-Plot als PNG im Output-Ordner.
        Gibt den Dateipfad zurück.
        """
        timestamp = datetime.now().strftime("%Y")
        filepath = os.path.join(self.output_dir, f"{filename}_{timestamp}.png")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Wichtig: Figure schließen, um Speicher zu sparen
        print(f"✅ Plot gespeichert: {filepath}")
        return filepath

    def preprocess_text(self, text):
        """Bereitet Texte für NLP-Analysen vor (Regex, Stopword-Entfernung, Stemming)."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Satzzeichen entfernen
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words and t.isalpha()]
        stemmed_tokens = [self.stemmer.stem(t) for t in tokens]
        return ' '.join(stemmed_tokens)

    def compute_similarity_scores(self):
        """
        Berechnet Cosine-Similarity zwischen Keywords und Texten.
        Gibt zwei Arrays zurück: avg_original_sim, avg_optimized_sim.
        """
        self.preprocessed_original = [self.preprocess_text(t) for t in self.original_texts_list_clean]
        self.preprocessed_optimized = [self.preprocess_text(t) for t in self.optimized_texts_list_clean]
        self.preprocessed_keywords = [self.preprocess_text(k) for k in self.keywords_final]

        all_texts = self.preprocessed_original + self.preprocessed_optimized + self.preprocessed_keywords

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        original_indices = range(len(self.preprocessed_original))
        optimized_indices = range(len(self.preprocessed_original), len(self.preprocessed_original) + len(self.preprocessed_optimized))
        keyword_indices = range(len(self.preprocessed_original) + len(self.preprocessed_optimized), len(all_texts))

        similarities_original = []
        similarities_optimized = []

        for i in original_indices:
            row_sims = []
            for j in keyword_indices:
                sim_val = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
                row_sims.append(sim_val)
            similarities_original.append(row_sims)

        for i in optimized_indices:
            row_sims = []
            for j in keyword_indices:
                sim_val = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
                row_sims.append(sim_val)
            similarities_optimized.append(row_sims)

        avg_original_sim = np.mean(similarities_original, axis=0)
        avg_optimized_sim = np.mean(similarities_optimized, axis=0)

        print("📊 Similarities (Original -> Keywords):", avg_original_sim)
        print("📊 Similarities (Optimiert -> Keywords):", avg_optimized_sim)
        return avg_original_sim, avg_optimized_sim

    def visualize_similarity_scores(self, keywords_final, avg_original_sim, avg_optimized_sim):
        """Erzeugt & speichert Balkendiagramm mit Cosine Similarities."""
        df_sim = pd.DataFrame({
            'Keywords': keywords_final,
            'Original': avg_original_sim,
            'Optimiert': avg_optimized_sim
        })
        df_melted = df_sim.melt(id_vars='Keywords', var_name='Textart', value_name='Cosine Similarity')

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Keywords', y='Cosine Similarity', hue='Textart', data=df_melted, ax=ax)
        plt.title('Durchschnittliche Cosine Similarity zu Keywords')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        path = self.save_plot(fig, "similarity_scores")
        plt.show()

        # Bildpfad im Dictionary speichern
        self.image_paths["similarity_scores"] = path
        return path


    def generate_wordclouds(self):
        """
        Erstellt zwei Varianten von Wordclouds:
        1. Gefiltert (Stemming + Stopwords)
        2. Bereinigt (nur Stopwords + Blacklist)
        Gibt Dictionary mit Pfaden zurück: {"filtered":..., "raw":...}
        """
        def safe_make_wc(text):
            if not text.strip():
                # Falls String leer => Placeholder
                print("⚠️ Wordcloud: Keine Wörter enthalten. Verwende Platzhaltertext.")
                text = "leer"
            return WordCloud(width=600, height=400, background_color='white').generate(text)

        # (A) Gefilterte Variante
        original_clean = ' '.join([self.preprocess_text(t) for t in self.original_texts_list_clean])
        optimized_clean = ' '.join([self.preprocess_text(t) for t in self.optimized_texts_list_clean])

        fig1, ax1 = plt.subplots(1, 2, figsize=(20, 10))
        wc_orig_clean = safe_make_wc(original_clean)
        wc_opt_clean = safe_make_wc(optimized_clean)

        ax1[0].imshow(wc_orig_clean, interpolation='bilinear')
        ax1[0].set_title('Wordcloud Original (gefiltert)')
        ax1[0].axis('off')

        ax1[1].imshow(wc_opt_clean, interpolation='bilinear')
        ax1[1].set_title('Wordcloud SEO (gefiltert)')
        ax1[1].axis('off')

        path_filtered = self.save_plot(fig1, "wordclouds_filtered")
        plt.show()

        # (B) Raw-Variante
        exclude_set = set(self.stop_words).union(self.wordcloud_exclude)
        
        def filter_words(text):
            words = re.findall(r'\b\w+\b', text.lower())
            return ' '.join([w for w in words if w not in exclude_set and len(w) > 2])

        original_raw = filter_words(' '.join(self.original_texts_list_clean))
        optimized_raw = filter_words(' '.join(self.optimized_texts_list_clean))

        fig2, ax2 = plt.subplots(1, 2, figsize=(20, 10))
        wc_orig_raw = safe_make_wc(original_raw)
        wc_opt_raw = safe_make_wc(optimized_raw)

        ax2[0].imshow(wc_orig_raw, interpolation='bilinear')
        ax2[0].set_title('Wordcloud Original (bereinigt)')
        ax2[0].axis('off')

        ax2[1].imshow(wc_opt_raw, interpolation='bilinear')
        ax2[1].set_title('Wordcloud SEO (bereinigt)')
        ax2[1].axis('off')

        path_raw = self.save_plot(fig2, "wordclouds_raw")
        plt.show()

        self.image_paths["wordcloud_filtered"] = path_filtered
        self.image_paths["wordcloud_raw"] = path_raw

        return {"filtered": path_filtered, "raw": path_raw}


    def plot_seo_trends(self):
        if self.df_metrics is None:
            print("⚠️ Keine historischen SEO-Daten vorhanden.")
            return None

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(self.df_metrics['Date'], self.df_metrics['Organic_Sessions'], color='blue', marker='o')
        ax1.set_xlabel('Datum')
        ax1.set_ylabel('Organische Sitzungen', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(self.df_metrics['Date'], self.df_metrics['Conversion_Rate'], color='red', marker='s')
        ax2.set_ylabel('Conversion Rate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title('Sitzungen & Conversion Rate')
        path = self.save_plot(fig, "seo_trends")
        plt.tight_layout()
        plt.show()

        self.image_paths["seo_trends"] = path
        return path

    def predict_future_sessions(self, months=6):
        if self.df_metrics is None:
            print("⚠️ Keine historischen SEO-Daten vorhanden.")
            return None

        X = np.arange(len(self.df_metrics)).reshape(-1, 1)
        y = np.array(self.df_metrics["Organic_Sessions"])
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, y)

        future_X = np.arange(len(self.df_metrics), len(self.df_metrics) + months).reshape(-1, 1)
        future_preds = model.predict(future_X)
        future_dates = pd.date_range(start=self.df_metrics["Date"].max(), periods=months, freq='ME')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.df_metrics["Date"], self.df_metrics["Organic_Sessions"], marker='o', label="Historisch")
        ax.plot(future_dates, future_preds, marker='x', linestyle='--', label="Prognose")
        plt.title("Prognose: Organische Sitzungen")
        plt.xlabel("Datum")
        plt.ylabel("Sitzungen")
        plt.legend()
        path = self.save_plot(fig, "session_forecast")
        plt.show()

        self.image_paths["session_forecast"] = path
        return path

    def predict_future_conversion_rate(self, months=6):
        if self.df_metrics is None:
            print("⚠️ Keine historischen SEO-Daten vorhanden.")
            return None

        last_val = self.df_metrics["Conversion_Rate"].iloc[-1]
        growth = (last_val / self.df_metrics["Conversion_Rate"].iloc[0]) ** (1 / len(self.df_metrics)) - 1
        future_vals = [last_val * ((1 + growth) ** i) for i in range(1, months + 1)]
        future_dates = pd.date_range(start=self.df_metrics["Date"].max(), periods=months, freq='ME')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.df_metrics["Date"], self.df_metrics["Conversion_Rate"], marker='o', label="Historisch")
        ax.plot(future_dates, future_vals, marker='x', linestyle='--', label="Prognose")
        plt.title("Prognose: Conversion Rate")
        plt.xlabel("Datum")
        plt.ylabel("Conversion Rate")
        plt.legend()
        path = self.save_plot(fig, "conversion_forecast")
        plt.show()

        self.image_paths["conversion_forecast"] = path
        return path

    def run_analysis(self):
        """
        Führt die Analyse durch:
         1) Similarities
         2) Similarity Scores Diagramm
         3) Wordclouds
        """
        print("🔍 Ähnlichkeitsanalyse gestartet...")
        avg_orig, avg_opt = self.compute_similarity_scores()

        print("📊 Visualisierung der Similarities:")
        similarity_chart = self.visualize_similarity_scores(self.keywords_final, avg_orig, avg_opt)

        print("🌍 Wordcloud-Visualisierung:")
        wc_paths = self.generate_wordclouds()

        # Komplette Pfade sammeln
        result_paths = {
            "similarities": similarity_chart,
            "wordcloud_filtered": wc_paths["filtered"],
            "wordcloud_raw": wc_paths["raw"]
        }

        return result_paths

    def run_models(self):
        """
        Führt alle Modellanalysen durch:
         1) SEO Trends
         2) Prognose Sitzungen
         3) Prognose Conversion Rate
        """
        print("📊 Starte Modellanalysen...")

        seo_trends_path = self.plot_seo_trends()
        session_forecast_path = self.predict_future_sessions()
        conversion_forecast_path = self.predict_future_conversion_rate()

        result_paths = {
            "seo_trends": seo_trends_path,
            "session_forecast": session_forecast_path,
            "conversion_forecast": conversion_forecast_path
        }
        return result_paths

    def get_all_image_paths(self):
        """
        Gibt das interne Dictionary self.image_paths zurück,
        falls du ALLE Pfade in einer Methode abrufen willst.
        """
        return self.image_paths
