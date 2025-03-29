import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import plotly.graph_objects as go
from datetime import datetime

class EmbeddingDemo:
    def __init__(self, output_dir="output", final_images=None):
        """
        :param output_dir: Ordner, in dem die Diagramme gespeichert werden
        :param final_images: Gemeinsames Dictionary zum Sammeln aller Bildpfade
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Falls man ein gemeinsames Dict für mehrere Klassen nutzt, kann man es hier übergeben
        # Ansonsten legen wir hier ein eigenes an.
        self.FINAL_IMAGES = final_images if final_images is not None else {}

    def _save_matplotlib(self, fig, name, transparent=False):
        """
        Speichert ein Matplotlib-Plot als PNG.
        Speichert den Pfad in self.FINAL_IMAGES und gibt ihn zurück.
        """
        timestamp = datetime.now().strftime("%Y")
        file_path = os.path.join(self.output_dir, f"{name}_{timestamp}.png")
        fig.savefig(file_path, dpi=300, bbox_inches='tight', transparent=transparent)
        plt.close(fig)
        print(f"✅ Plot gespeichert: {file_path}")

        # Bildpfad ins Dictionary eintragen
        self.FINAL_IMAGES[name] = file_path
        return file_path

    def _save_plotly(self, fig, name):
        """
        Speichert ein Plotly-Plot als PNG und SVG.
        Speichert beide Pfade in self.FINAL_IMAGES, als Liste oder dict.
        """
        timestamp = datetime.now().strftime("%Y")
        path_png = os.path.join(self.output_dir, f"{name}_{timestamp}.png")
        path_svg = os.path.join(self.output_dir, f"{name}_{timestamp}.svg")

        fig.write_image(path_png)
        fig.write_image(path_svg)
        print(f"✅ Plot gespeichert: {path_png} / {path_svg}")

        # Speichern im Dictionary, z. B. als Liste
        self.FINAL_IMAGES[name] = {"png": path_png, "svg": path_svg}

    def get_bert_embedding(self, word):
        inputs = self.tokenizer(word, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

    def plot_3d_keyword_similarity(self, title="Keyword Cloud Similarity Visualization"):
        """
        Erstellt ein 3D-Scatter-Diagramm (matplotlib).
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        np.random.seed(42)
        keywords = np.random.rand(50, 3) - 0.5
        old_text = keywords + np.array([0.4, 0.4, 0.4])
        optimized_text = keywords + np.array([-0.4, -0.3, -0.2])

        ax.scatter(keywords[:, 0], keywords[:, 1], keywords[:, 2],
                   color='g', label='Keywords', alpha=0.6, s=40)
        ax.scatter(old_text[:, 0], old_text[:, 1], old_text[:, 2],
                   color='r', label='Old Text', alpha=0.7, s=50)
        ax.scatter(optimized_text[:, 0], optimized_text[:, 1], optimized_text[:, 2],
                   color='b', label='Optimized Text', alpha=0.7, s=50)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title(title)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.legend()

        self._save_matplotlib(fig, "embedding_keyword_3d", transparent=False)

    def plot_cosine_similarity_steps(self):
        """
        Matplotlib-Plot in 3 Schritten:
        1) Vektor-Darstellung
        2) Dot-Product
        3) Cosine Similarity
        """
        fig = plt.figure(figsize=(18, 6))

        ax1 = fig.add_subplot(131, projection='3d')
        vec1 = np.array([0.8, 0.6, 0.3])
        vec2 = np.array([0.9, 0.4, 0.5])
        ax1.quiver(0, 0, 0, vec1[0], vec1[1], vec1[2], color='r',
                   label='Old Text', arrow_length_ratio=0.1)
        ax1.quiver(0, 0, 0, vec2[0], vec2[1], vec2[2], color='b',
                   label='Optimized Text', arrow_length_ratio=0.1)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_zlim([0, 1])
        ax1.set_title('Step 1: Embeddings')
        ax1.legend()

        ax2 = fig.add_subplot(132)
        dot_product = np.dot(vec1, vec2)
        ax2.bar(['Dot Product'], [dot_product], color='orange')
        ax2.set_title('Step 2: Dot Product')

        ax3 = fig.add_subplot(133)
        cos_sim = cosine_similarity([vec1], [vec2])[0][0]
        ax3.bar(['Cosine Similarity'], [cos_sim], color='green')
        ax3.set_title('Step 3: Cosine Similarity')

        self._save_matplotlib(fig, "cosine_similarity_steps", transparent=True)

    def plot_3d_cosine_comparison_with_labels(self):
        """
        Matplotlib 3D-Vergleich:
        SEO Keywords, Old Text, Optimized Text
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        vec_keywords = np.array([0.8, 0.7, 0.6])
        vec_old_text = np.array([0.4, 0.3, 0.2])
        vec_optimized_text = np.array([0.6, 0.5, 0.4])

        ax.quiver(0, 0, 0, *vec_keywords, color='#4CAF50',
                  label='SEO Keywords', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, *vec_old_text, color='#FF5733',
                  label='Old Text', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, *vec_optimized_text, color='#1E90FF',
                  label='Optimized Text', arrow_length_ratio=0.1)

        ax.text(*vec_keywords, 'SEO Keywords', color='#4CAF50', fontsize=12)
        ax.text(*vec_old_text, 'Old Text', color='#FF5733', fontsize=12)
        ax.text(*vec_optimized_text, 'Optimized Text', color='#1E90FF', fontsize=12)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.set_title("3D Cosine Similarity: Keywords vs Texte")
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.legend()

        self._save_matplotlib(fig, "cosine_comparison_3d", transparent=True)

    def plot_bert_embeddings_3d(self, words, relationships=None):
        """
        Erzeugt interaktive 3D-Visualisierung (Plotly) von BERT Embeddings.
        Speichert PNG + SVG, Pfade in self.FINAL_IMAGES
        """
        bert_embeddings = np.array([self.get_bert_embedding(word) for word in words])
        pca = PCA(n_components=3)
        bert_embeddings_3d = pca.fit_transform(bert_embeddings)

        fig = go.Figure()

        for i, word in enumerate(words):
            fig.add_trace(go.Scatter3d(
                x=[bert_embeddings_3d[i, 0]],
                y=[bert_embeddings_3d[i, 1]],
                z=[bert_embeddings_3d[i, 2]],
                mode='markers+text',
                marker=dict(size=10, color='skyblue', opacity=0.8),
                text=word,
                textposition='top center',
                name=word
            ))

        if relationships:
            for word1, word2, color in relationships:
                if word1 in words and word2 in words:
                    idx1, idx2 = words.index(word1), words.index(word2)
                    fig.add_trace(go.Scatter3d(
                        x=[bert_embeddings_3d[idx1, 0], bert_embeddings_3d[idx2, 0]],
                        y=[bert_embeddings_3d[idx1, 1], bert_embeddings_3d[idx2, 1]],
                        z=[bert_embeddings_3d[idx1, 2], bert_embeddings_3d[idx2, 2]],
                        mode='lines',
                        line=dict(color=color, width=4),
                        showlegend=False
                    ))

        fig.update_layout(
            title="Interactive 3D Visualization of BERT Word Embeddings",
            scene=dict(
                xaxis=dict(title="PCA Component 1"),
                yaxis=dict(title="PCA Component 2"),
                zaxis=dict(title="PCA Component 3"),
                aspectmode='cube'
            ),
            showlegend=False
        )

        # Speichere Plotly-Plot
        timestamp = datetime.now().strftime("%Y")
        name = "bert_embeddings_3d"
        path_png = os.path.join(self.output_dir, f"{name}_{timestamp}.png")
        path_svg = os.path.join(self.output_dir, f"{name}_{timestamp}.svg")

        fig.write_image(path_png)
        fig.write_image(path_svg)

        print(f"✅ Plot gespeichert: {path_png} / {path_svg}")

        # Ins Dictionary eintragen
        self.FINAL_IMAGES[name] = {"png": path_png, "svg": path_svg}

    def run_all_visualizations(self):
        """
        Führt alle Visualisierungen aus:
        1) 3D Keyword Similarity (matplotlib)
        2) Cosine Similarity Steps (matplotlib)
        3) 3D Cosine Comparison (matplotlib)
        4) BERT Embeddings 3D (plotly)
        """
        print("\n📊 3D Keyword Similarity")
        self.plot_3d_keyword_similarity()

        print("\n🔍 Cosine Similarity Steps")
        self.plot_cosine_similarity_steps()

        print("\n📍 Vergleich als Vektoren (3D)")
        self.plot_3d_cosine_comparison_with_labels()

        print("\n🧠 BERT Embeddings Visualisierung (plotly)")
        words = ["king", "queen", "man", "woman", "red", "yellow",
                 "banana", "apple", "day", "sun", "night", "moon"]
        relationships = [
            ("king", "man", "crimson"),
            ("queen", "woman", "crimson"),
            ("apple", "red", "olivedrab"),
            ("banana", "yellow", "gold"),
            ("day", "sun", "orange"),
            ("night", "moon", "purple")
        ]
        self.plot_bert_embeddings_3d(words, relationships)

    def get_image_paths(self):
        """
        Gibt das Dictionary mit allen bisher gespeicherten Pfaden zurück.
        """
        return self.FINAL_IMAGES
