import os
from datetime import datetime
import kaleido
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import plotly.graph_objects as go

class EmbeddingDemo:
    def __init__(self, output_dir="output"):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_matplotlib(self, fig, name, transparent=False):
        timestamp = datetime.now().strftime("%Y")
        file_path = os.path.join(self.output_dir, f"{name}_{timestamp}.png")
        fig.savefig(file_path, dpi=300, bbox_inches='tight', transparent=transparent)
        print(f"✅ Plot gespeichert: {file_path}")
        plt.close(fig)

    def _save_plotly(self, fig, name):
        timestamp = datetime.now().strftime("%Y")
        path_png = os.path.join(self.output_dir, f"{name}_{timestamp}.png")
        path_svg = os.path.join(self.output_dir, f"{name}_{timestamp}.svg")
        fig.write_image(path_png)
        fig.write_image(path_svg)
        print(f"✅ Plot gespeichert: {path_png} / {path_svg}")

    def plot_3d_keyword_similarity(self, title="Keyword Cloud Similarity Visualization"):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        np.random.seed(42)
        keywords = np.random.rand(50, 3) - 0.5
        old_text = keywords + np.array([0.4, 0.4, 0.4])
        optimized_text = keywords + np.array([-0.4, -0.3, -0.2])

        ax.scatter(keywords[:, 0], keywords[:, 1], keywords[:, 2], color='g', label='Keywords', alpha=0.6, s=40)
        ax.scatter(old_text[:, 0], old_text[:, 1], old_text[:, 2], color='r', label='Old Text', alpha=0.7, s=50)
        ax.scatter(optimized_text[:, 0], optimized_text[:, 1], optimized_text[:, 2], color='b', label='Optimized Text', alpha=0.7, s=50)

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
        fig = plt.figure(figsize=(18, 6))

        ax1 = fig.add_subplot(131, projection='3d')
        vec1 = np.array([0.8, 0.6, 0.3])
        vec2 = np.array([0.9, 0.4, 0.5])
        ax1.quiver(0, 0, 0, vec1[0], vec1[1], vec1[2], color='r', label='Old Text', arrow_length_ratio=0.1)
        ax1.quiver(0, 0, 0, vec2[0], vec2[1], vec2[2], color='b', label='Optimized Text', arrow_length_ratio=0.1)
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
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        vec_keywords = np.array([0.8, 0.7, 0.6])
        vec_old_text = np.array([0.4, 0.3, 0.2])
        vec_optimized_text = np.array([0.6, 0.5, 0.4])

        ax.quiver(0, 0, 0, *vec_keywords, color='#4CAF50', label='SEO Keywords', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, *vec_old_text, color='#FF5733', label='Old Text', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, *vec_optimized_text, color='#1E90FF', label='Optimized Text', arrow_length_ratio=0.1)

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
        bert_embeddings = np.array([self.get_bert_embedding(word) for word in words])
        pca = PCA(n_components=3)
        bert_embeddings_3d = pca.fit_transform(bert_embeddings)

        fig = go.Figure()

        for i, word in enumerate(words):
            fig.add_trace(go.Scatter3d(
                x=[bert_embeddings_3d[i, 0]], y=[bert_embeddings_3d[i, 1]], z=[bert_embeddings_3d[i, 2]],
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

        self._save_plotly(fig, "bert_embeddings_3d")

    def get_bert_embedding(self, word):
        inputs = self.tokenizer(word, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

    def calculate_pairwise_distances(self, word_pairs, words):
        bert_embeddings = np.array([self.get_bert_embedding(word) for word in words])
        distance_results = []

        for (word1, word2, word3, word4) in word_pairs:
            idx1, idx2, idx3, idx4 = words.index(word1), words.index(word2), words.index(word3), words.index(word4)
            distance1 = np.linalg.norm(bert_embeddings[idx1] - bert_embeddings[idx2])
            distance2 = np.linalg.norm(bert_embeddings[idx3] - bert_embeddings[idx4])
            distance_results.append({
                "Word Pair 1": f"{word1} - {word2}",
                "Distance 1": round(distance1, 2),
                "Word Pair 2": f"{word3} - {word4}",
                "Distance 2": round(distance2, 2)
            })

        return pd.DataFrame(distance_results)

    def run_all_visualizations(self):
        print("\n📊 3D Keyword Similarity")
        self.plot_3d_keyword_similarity()

        print("\n🔍 Cosine Similarity Steps")
        self.plot_cosine_similarity_steps()

        print("\n📍 Vergleich als Vektoren")
        self.plot_3d_cosine_comparison_with_labels()

        print("\n🧠 BERT Embeddings Visualisierung")
        words = ["king", "queen", "man", "woman", "red", "yellow", "banana", "apple", "day", "sun", "night", "moon"]
        relationships = [("king", "man", "crimson"), ("queen", "woman", "crimson"),
                         ("apple", "red", "olivedrab"), ("banana", "yellow", "gold"),
                         ("day", "sun", "orange"), ("night", "moon", "purple")]
        self.plot_bert_embeddings_3d(words, relationships)
