import os
import json
import re
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from datetime import datetime

class SEOAnalyzer:
    """
    Diese Klasse analysiert den SEO-Optimierungseffekt, berechnet Statistiken & visualisiert die Ergebnisse.
    """

    def __init__(self, seo_json, keywords_final, output_dir="output", historical_data=None):
        self.seo_json = seo_json
        self.keywords_final = keywords_final
        self.original_texts_list_clean = [seo_json[key]['alt'] for key in seo_json]
        self.optimized_texts_list_clean = [seo_json[key]['SEO'] for key in seo_json]
        self.nlp = spacy.load('de_core_news_sm')
        self.stop_words = set(stopwords.words('german'))
        self.df_metrics = self.load_historical_data(historical_data)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def load_historical_data(historical_data):
        """Lädt historische SEO-Daten in einen Pandas DataFrame."""
        if historical_data is None:
            return None
        
        df = pd.DataFrame(historical_data)
        df["Date"] = pd.to_datetime(df["Date"])  # Konvertiere das Datum in datetime-Format
        return df


    def save_plot(self, fig, filename):
        """Speichert einen Matplotlib-Plot als PNG im Output-Ordner."""
        timestamp = datetime.now().strftime("%Y%m%d")
        filepath = os.path.join(self.output_dir, f"{filename}_{timestamp}.png")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"✅ Plot gespeichert: {filepath}")


    def preprocess_text(self, text):
        """Bereitet Texte für NLP-Analysen vor (Tokenisierung, Stopwort-Entfernung)."""
        text = text.lower()
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in self.stop_words and token.is_alpha]
        return ' '.join(tokens)


    def compute_similarity_scores(self):
        """Berechnet Cosine-Similarity zwischen Keywords und Texten."""
        self.preprocessed_original = [self.preprocess_text(t) for t in self.original_texts_list_clean]
        self.preprocessed_optimized = [self.preprocess_text(t) for t in self.optimized_texts_list_clean]
        self.preprocessed_keywords = self.keywords_final

        all_texts = self.preprocessed_original + self.preprocessed_optimized + self.preprocessed_keywords
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        original_indices = range(len(self.preprocessed_original))
        optimized_indices = range(len(self.preprocessed_original), len(self.preprocessed_original) + len(self.preprocessed_optimized))
        keyword_indices = range(len(self.preprocessed_original) + len(self.preprocessed_optimized), len(all_texts))

        similarities_original = []
        similarities_optimized = []

        for i in original_indices:
            similarities_original.append([cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0] for j in keyword_indices])
        
        for i in optimized_indices:
            similarities_optimized.append([cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0] for j in keyword_indices])

        avg_original_sim = np.mean(similarities_original, axis=0)
        avg_optimized_sim = np.mean(similarities_optimized, axis=0)

        print("📊 Durchschnittliche Similarities (Original -> Keywords):", avg_original_sim)
        print("📊 Durchschnittliche Similarities (Optimiert -> Keywords):", avg_optimized_sim)

        return avg_original_sim, avg_optimized_sim


    def visualize_similarity_scores(self, keywords_final, avg_original_sim, avg_optimized_sim):
        """Erstellt und speichert eine Balkendiagramm-Visualisierung der Similarities."""
        df_sim = pd.DataFrame({'Keywords': keywords_final, 'Original': avg_original_sim, 'Optimiert': avg_optimized_sim})
        df_melted = df_sim.melt(id_vars='Keywords', var_name='Textart', value_name='Cosine Similarity')

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Keywords', y='Cosine Similarity', hue='Textart', data=df_melted, ax=ax)
        plt.title('Durchschnittliche Cosine Similarity zu Keywords')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        self.save_plot(fig, "similarity_scores")
        plt.show()


    def generate_wordclouds(self):
        """Erstellt und speichert Wordclouds für Original- und SEO-optimierte Texte."""
        all_original_text = ' '.join(self.original_texts_list_clean)
        all_optimized_text = ' '.join(self.optimized_texts_list_clean)

        wc_original = WordCloud(width=600, height=400, background_color='white').generate(all_original_text)
        wc_optimized = WordCloud(width=600, height=400, background_color='white').generate(all_optimized_text)

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(wc_original, interpolation='bilinear')
        ax[0].set_title('Wordcloud Original Texte')
        ax[0].axis('off')

        ax[1].imshow(wc_optimized, interpolation='bilinear')
        ax[1].set_title('Wordcloud SEO-Optimierte Texte')
        ax[1].axis('off')

        self.save_plot(fig, "wordclouds")
        plt.show()


    def plot_seo_trends(self):
        """Visualisiert historische SEO-Daten und speichert die Plots."""
        if self.df_metrics is None:
            print("⚠️ Keine historischen SEO-Daten vorhanden.")
            return

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(self.df_metrics['Date'], self.df_metrics['Organic_Sessions'], color='blue', marker='o', label='Organic Sessions')
        ax1.set_xlabel('Datum')
        ax1.set_ylabel('Anzahl organischer Sitzungen', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(self.df_metrics['Date'], self.df_metrics['Conversion_Rate'], color='red', marker='s', label='Conversion Rate')
        ax2.set_ylabel('Conversion Rate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        fig.tight_layout()
        plt.title('Entwicklung der Sitzungen und Conversion Rate')

        self.save_plot(fig, "seo_trends")
        plt.show()


    def predict_future_sessions(self, months=6):
        """Einfache lineare Prognose der organischen Sitzungen und speichert den Plot."""
        if self.df_metrics is None:
            print("⚠️ Keine historischen SEO-Daten vorhanden.")
            return
        
        X = np.arange(len(self.df_metrics)).reshape(-1, 1)
        y = np.array(self.df_metrics["Organic_Sessions"])

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(len(self.df_metrics), len(self.df_metrics) + months).reshape(-1, 1)
        future_predictions = model.predict(future_X)
        future_dates = pd.date_range(start=self.df_metrics["Date"].max(), periods=months, freq='ME')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.df_metrics["Date"], self.df_metrics["Organic_Sessions"], marker='o', label="Vergangene Sessions")
        ax.plot(future_dates, future_predictions, marker='x', linestyle='dashed', label="Prognose Sessions")
        plt.xlabel("Datum")
        plt.ylabel("Organische Sitzungen")
        plt.legend()
        plt.title("Prognose der organischen Sitzungen")

        self.save_plot(fig, "session_forecast")
        plt.show()


    def predict_future_conversion_rate(self, months=6):
        """Einfache Prognose der Conversion-Rate und speichert den Plot."""
        if self.df_metrics is None:
            print("⚠️ Keine historischen SEO-Daten vorhanden.")
            return
        
        last_value = self.df_metrics["Conversion_Rate"].iloc[-1]
        growth_rate = (self.df_metrics["Conversion_Rate"].iloc[-1] / self.df_metrics["Conversion_Rate"].iloc[0]) ** (1 / len(self.df_metrics)) - 1
        future_values = [last_value * ((1 + growth_rate) ** i) for i in range(1, months + 1)]
        future_dates = pd.date_range(start=self.df_metrics["Date"].max(), periods=months, freq='ME')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.df_metrics["Date"], self.df_metrics["Conversion_Rate"], marker='o', label="Vergangene Conversion Rate")
        ax.plot(future_dates, future_values, marker='x', linestyle='dashed', label="Prognose Conversion Rate")
        plt.xlabel("Datum")
        plt.ylabel("Conversion Rate")
        plt.legend()
        plt.title("Prognose der Conversion-Rate")

        self.save_plot(fig, "conversion_forecast")
        plt.show()


    def run_analysis(self):
        """Führt die gesamte Analyse und Visualisierung aus."""

        print("\n🔍 Berechnung der Similarities:")
        avg_original_sim, avg_optimized_sim = self.compute_similarity_scores()

        print("\n📊 Visualisierung der Similarities:")
        self.visualize_similarity_scores(self.keywords_final, avg_original_sim, avg_optimized_sim)

        print("\n📊 Wordcloud-Visualisierung:")
        self.generate_wordclouds()


    def run_models(self):
        """Führt alle SEO- und Modellanalysen durch."""
        print("\n📊 Historische SEO-Trends")
        self.plot_seo_trends()

        print("\n📈 Prognose der organischen Sitzungen")
        self.predict_future_sessions()

        print("\n📈 Prognose der Conversion-Rate")
        self.predict_future_conversion_rate()