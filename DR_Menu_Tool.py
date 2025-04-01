# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

# Dimensionality Reduction Methods
def apply_pca(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_scaled)
    return reduced


def apply_lda(data, target):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    lda = LDA(n_components=2)
    reduced = lda.fit_transform(data_scaled, target)
    return reduced


def apply_tsne(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(data_scaled)
    return reduced


def apply_svd(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    svd = TruncatedSVD(n_components=2)
    reduced = svd.fit_transform(data_scaled)
    return reduced


def apply_mds(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    mds = MDS(n_components=2, random_state=42)
    reduced = mds.fit_transform(data_scaled)
    return reduced


# Visualization
def visualize_reduction(reduced_data, title):
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6)
    plt.title(title)
    plt.show()


# Load and reduce data
def Load_and_reduce():
    try:
        # User input for file path and method
        filename = input("Enter the file path: ")
        method = input("Choose method (pca/lda/tsne/svd/mds): ")

        # Load data
        data = pd.read_csv(filename)
        print("Available features in the dataset:", list(data.columns))
        target_col = input("Enter the target column name (if applicable, else leave blank): ")
        has_target = bool(target_col)
        
        num_samples = min(5000, len(data))
        data = data.sample(n=num_samples, random_state=42)
        print(f"Loaded {filename} with {len(data)} randomly selected entries.")
        print("Available features:", list(data.columns))

        # Separate features and target
        if has_target:
            features = data.drop(columns=[target_col])
            target = data[target_col]
        else:
            features = data
            target = None

        # Apply chosen method
        if method == 'pca':
            reduced = apply_pca(features)
            visualize_reduction(reduced, 'PCA')
        elif method == 'lda' and has_target:
            reduced = apply_lda(features, target)
            visualize_reduction(reduced, 'LDA')
        elif method == 'tsne':
            reduced = apply_tsne(features)
            visualize_reduction(reduced, 't-SNE')
        elif method == 'svd':
            reduced = apply_svd(features)
            visualize_reduction(reduced, 'SVD')
        elif method == 'mds':
            reduced = apply_mds(features)
            visualize_reduction(reduced, 'MDS')
        else:
            print("Invalid method or missing target.")
    except Exception as e:
        print(f"Error: {e}")

# Run the tool
Load_and_reduce()
