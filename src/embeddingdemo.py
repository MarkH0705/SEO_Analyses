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
    """
    Diese Klasse visualisiert und analysiert semantische Ähnlichkeiten mithilfe von Embeddings.
    """

    def __init__(self):
        """Lädt das BERT-Modell für Embeddings."""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def plot_3d_keyword_similarity(self, title="Keyword Cloud Similarity Visualization"):
        """Visualisiert Keywords, Original- und optimierte Texte in einem 3D-Plot."""
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

        plt.show()

    def plot_cosine_similarity_steps(self):
        """Visualisiert Cosine Similarity in drei Schritten."""
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

        plt.show()

    def get_bert_embedding(self, word):
        """Berechnet das BERT-Embedding für ein Wort."""
        inputs = self.tokenizer(word, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

    def calculate_pairwise_distances(self, word_pairs, words):
        """Berechnet paarweise Distanzen für Wortpaare mit BERT-Embeddings."""
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

    def plot_bert_embeddings_3d(self, words, relationships=None):
        """Erstellt eine interaktive 3D-Visualisierung der BERT-Wort-Embeddings mit Pfeilen für Beziehungen."""
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

        fig.show()


    # 3D Visualization with vector names instead of cosine similarity values
    def plot_3d_cosine_comparison_with_labels(self):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Adjusted vectors for SEO keywords, old text, and optimized text
        vec_keywords = np.array([0.8, 0.7, 0.6])
        vec_old_text = np.array([0.4, 0.3, 0.2])
        vec_optimized_text = np.array([0.6, 0.5, 0.4])

        # Plot vectors with improved color scheme
        ax.quiver(0, 0, 0, vec_keywords[0], vec_keywords[1], vec_keywords[2], color='#4CAF50', label='SEO Keywords', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, vec_old_text[0], vec_old_text[1], vec_old_text[2], color='#FF5733', label='Old Text', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, vec_optimized_text[0], vec_optimized_text[1], vec_optimized_text[2], color='#1E90FF', label='Optimized Text', arrow_length_ratio=0.1)

        # Add labels to the vector tips
        ax.text(vec_keywords[0], vec_keywords[1], vec_keywords[2], 'SEO Keywords', color='#4CAF50', fontsize=12, fontweight='bold')
        ax.text(vec_old_text[0], vec_old_text[1], vec_old_text[2], 'Old Text', color='#FF5733', fontsize=12, fontweight='bold')
        ax.text(vec_optimized_text[0], vec_optimized_text[1], vec_optimized_text[2], 'Optimized Text', color='#1E90FF', fontsize=12, fontweight='bold')

        # Plot settings
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.set_title("3D Cosine Similarity: SEO Keywords vs Old Text vs Optimized Text\nWith Labels")
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.legend()

        plt.show()




    def run_all_visualizations(self):
        """Führt alle Visualisierungen und Analysen aus."""
        print("\n📊 3D-Keyword Similarity")
        self.plot_3d_keyword_similarity()

        print("\n🔍 Cosine Similarity Steps")
        self.plot_cosine_similarity_steps()

        print("\n🔍 Cosine Comparision")
        self.plot_3d_cosine_comparison_with_labels()

        print("\n📌 3D BERT Embedding Visualization")
        words = ["king", "queen", "man", "woman", "red", "yellow", "banana", "apple", "day", "sun", "night", "moon"]
        relationships = [("king", "man", "crimson"), ("queen", "woman", "crimson"), ("apple", "red", "olivedrab"),
                         ("banana", "yellow", "olivedrab"), ("day", "sun", "orange"), ("night", "moon", "orange")]
        self.plot_bert_embeddings_3d(words, relationships)
