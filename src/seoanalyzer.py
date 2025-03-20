import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from datetime import datetime
import utils


class SEOAnalyzer:
    """
    Diese Klasse analysiert den SEO-Optimierungseffekt, berechnet Statistiken & visualisiert die Ergebnisse.
    """

    def __init__(self, seo_json, original_texts, keywords_final, historical_data=None):
        """
        :param seo_json: JSON mit den optimierten Texten und alten Website-Texten
        :param original_texts: Dictionary mit den originalen Website-Texten
        """
        self.seo_json = seo_json
        self.original_texts = original_texts

        self.original_texts_list_clean = [seo_json[key]['alt'] for key in seo_json]
        self.optimized_texts_list_clean = [seo_json[key]['SEO'] for key in seo_json]

        self.keywords_final = keywords_final
        self.preprocessed_original = []
        self.preprocessed_optimized = []
        self.preprocessed_keywords = []
        self.nlp = spacy.load('de_core_news_sm')
        self.stop_words = set(stopwords.words('german'))
        self.df_metrics = self.load_historical_data(historical_data)

    def text_stats(self, text):
        """Berechnet grundlegende Textstatistiken."""
        words = text.split()
        return {
            "Zeichenanzahl": len(text),
            "Wortanzahl": len(words),
            "Satzanzahl": text.count('.') + text.count('!') + text.count('?')
        }

    def compare_word_frequencies(self):
        """Vergleicht die Wortfrequenzen zwischen Original- & SEO-optimierten Texten."""
        for idx, _ in enumerate(self.original_texts_list_clean):
            original_freq = Counter(self.original_texts_list_clean[idx].lower().split())
            optimized_freq = Counter(self.optimized_texts_list_clean[idx].lower().split())

            diff = {word: optimized_freq[word] - original_freq[word] for word in set(original_freq) | set(optimized_freq)}
            sorted_diff = sorted(diff.items(), key=lambda x: x[1], reverse=True)

            print(f"🔹 Wortfrequenz-Differenzen für Text {idx + 1}:")
            for word, change in sorted_diff[:10]:  # Zeige Top 10 Veränderungen
                print(f"{word}: {change}")

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
        """Erstellt eine Balkendiagramm-Visualisierung der Similarities."""
        df_sim = pd.DataFrame({'Keywords': keywords_final, 'Original': avg_original_sim, 'Optimiert': avg_optimized_sim})
        df_melted = df_sim.melt(id_vars='Keywords', var_name='Textart', value_name='Cosine Similarity')

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Keywords', y='Cosine Similarity', hue='Textart', data=df_melted)
        plt.title('Durchschnittliche Cosine Similarity zu Keywords')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def visualize_keyword_frequencies(self, preprocessed_keywords):
        """Erstellt eine Visualisierung der Keyword-Häufigkeiten."""
        original_counts = [sum(t.count(kw) for t in self.preprocessed_original) for kw in preprocessed_keywords]
        optimized_counts = [sum(t.count(kw) for t in self.preprocessed_optimized) for kw in preprocessed_keywords]

        df_counts = pd.DataFrame({'Keyword': preprocessed_keywords, 'Original Count': original_counts, 'Optimiert Count': optimized_counts})
        df_counts_melt = df_counts.melt(id_vars='Keyword', var_name='Textart', value_name='Count')

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Keyword', y='Count', hue='Textart', data=df_counts_melt)
        plt.title('Keyword Häufigkeit (exakte Matches)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def generate_wordclouds(self):
        """Erstellt Wordclouds für Original- und SEO-optimierte Texte."""
        all_original_text = ' '.join(self.preprocessed_original)
        all_optimized_text = ' '.join(self.preprocessed_optimized)

        wc_original = WordCloud(width=600, height=400, background_color='white').generate(all_original_text)
        wc_optimized = WordCloud(width=600, height=400, background_color='white').generate(all_optimized_text)

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(wc_original, interpolation='bilinear')
        ax[0].set_title('Wordcloud Original Texte')
        ax[0].axis('off')

        ax[1].imshow(wc_optimized, interpolation='bilinear')
        ax[1].set_title('Wordcloud SEO-Optimierte Texte')
        ax[1].axis('off')

        plt.show()

    def run_analysis(self):
        """Führt die gesamte Analyse und Visualisierung aus."""

        print("\n🔍 Wortfrequenz-Vergleich:")
        self.compare_word_frequencies()

        print("\n🔍 Berechnung der Similarities:")
        avg_original_sim, avg_optimized_sim = self.compute_similarity_scores()

        print("\n📊 Visualisierung der Similarities:")
        self.visualize_similarity_scores(self.keywords_final, avg_original_sim, avg_optimized_sim)

        print("\n📊 Visualisierung der Keyword-Häufigkeiten:")
        self.visualize_keyword_frequencies(self.keywords_final)

        print("\n☁️ Wordcloud-Visualisierung:")
        self.generate_wordclouds()

    @staticmethod
    def load_historical_data(historical_data):
        """Lädt historische SEO-Daten in einen Pandas DataFrame."""
        if historical_data is None:
            return None
        
        df = pd.DataFrame(historical_data)
        df["Date"] = pd.to_datetime(df["Date"])  # Konvertiere das Datum in datetime-Format
        return df

    def plot_seo_trends(self):
        """Visualisiert historische SEO-Daten mit Sitzungen, Conversion-Rate & Verweildauer."""
        if self.df_metrics is None:
            print("⚠️ Keine historischen SEO-Daten vorhanden.")
            return

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Primäre y-Achse: Organische Sitzungen
        ax1.plot(self.df_metrics['Date'], self.df_metrics['Organic_Sessions'], color='blue', marker='o', label='Organic Sessions')
        ax1.set_xlabel('Datum')
        ax1.set_ylabel('Anzahl organischer Sitzungen', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Sekundäre y-Achse: Conversion Rate
        ax2 = ax1.twinx()
        ax2.plot(self.df_metrics['Date'], self.df_metrics['Conversion_Rate'], color='red', marker='s', label='Conversion Rate')
        ax2.set_ylabel('Conversion Rate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        fig.tight_layout()
        plt.title('Entwicklung der Sitzungen und Conversion Rate')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.bar(self.df_metrics['Date'].dt.strftime('%b %Y'), self.df_metrics['Average_Time_on_Page'], color='green')
        plt.title('Durchschnittliche Verweildauer pro Monat')
        plt.xlabel('Monat')
        plt.ylabel('Verweildauer in Sekunden')
        plt.show()

    def predict_future_sessions(self, months=6):
        """Einfache lineare Prognose der organischen Sitzungen basierend auf historischen Daten."""
        if self.df_metrics is None:
            print("⚠️ Keine historischen SEO-Daten vorhanden.")
            return
        
        X = np.arange(len(self.df_metrics)).reshape(-1, 1)  # Zeit als numerischen Wert nehmen
        y = np.array(self.df_metrics["Organic_Sessions"])  # Zielvariable

        # Einfache lineare Regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(len(self.df_metrics), len(self.df_metrics) + months).reshape(-1, 1)
        future_predictions = model.predict(future_X)

        future_dates = pd.date_range(start=self.df_metrics["Date"].max(), periods=months, freq='ME')

        # Visualisierung
        plt.figure(figsize=(10, 5))
        plt.plot(self.df_metrics["Date"], self.df_metrics["Organic_Sessions"], marker='o', label="Vergangene Sessions")
        plt.plot(future_dates, future_predictions, marker='x', linestyle='dashed', label="Prognose Sessions")
        plt.xlabel("Datum")
        plt.ylabel("Organische Sitzungen")
        plt.legend()
        plt.title("Prognose der organischen Sitzungen für die nächsten Monate")
        plt.show()

    def predict_future_conversion_rate(self, months=6):
        """Einfache Prognose der Conversion-Rate auf Basis der letzten Werte."""
        if self.df_metrics is None:
            print("⚠️ Keine historischen SEO-Daten vorhanden.")
            return
        
        last_value = self.df_metrics["Conversion_Rate"].iloc[-1]
        growth_rate = (self.df_metrics["Conversion_Rate"].iloc[-1] / self.df_metrics["Conversion_Rate"].iloc[0]) ** (1 / len(self.df_metrics)) - 1
        future_values = [last_value * ((1 + growth_rate) ** i) for i in range(1, months + 1)]

        future_dates = pd.date_range(start=self.df_metrics["Date"].max(), periods=months, freq='ME')

        # Visualisierung
        plt.figure(figsize=(10, 5))
        plt.plot(self.df_metrics["Date"], self.df_metrics["Conversion_Rate"], marker='o', label="Vergangene Conversion Rate")
        plt.plot(future_dates, future_values, marker='x', linestyle='dashed', label="Prognose Conversion Rate")
        plt.xlabel("Datum")
        plt.ylabel("Conversion Rate")
        plt.legend()
        plt.title("Prognose der Conversion-Rate für die nächsten Monate")
        plt.show()

    def run_models(self):
        """Führt alle SEO- und Modellanalysen durch."""
        print("\n📊 Historische SEO-Trends")
        self.plot_seo_trends()

        print("\n📈 Prognose der organischen Sitzungen")
        self.predict_future_sessions()

        print("\n📈 Prognose der Conversion-Rate")
        self.predict_future_conversion_rate()
