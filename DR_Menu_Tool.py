# Enhanced Dimensionality Reduction Script

# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Set Seaborn style for better visualization
sns.set(style='whitegrid')

# Dataset Description
print("Dimensionality reduction helps visualize high-dimensional data and can also enhance model performance by reducing noise.")
print("This script applies various dimensionality reduction techniques (PCA, LDA, t-SNE, SVD, MDS) on user-provided datasets.")

# Data Sampling Function
def sample_data(data, max_samples=10000):
    """
    Randomly samples data to reduce size for computational efficiency.
    """
    if len(data) > max_samples:
        print(f"Dataset is large ({len(data)} entries). Sampling {max_samples} entries for efficiency.")
        return data.sample(n=max_samples, random_state=42)
    print("Dataset size is manageable. Using full data.")
    return data

# Data Pre-processing
def preprocess_data(data):
    """
    Preprocess the data by handling missing values and encoding categorical features.
    """
    data = data.dropna()  # Drop rows with any missing data
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()  # Select numeric columns
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()  # Select categorical columns
    
    # One-hot encode categorical columns
    if categorical_cols:
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
        data = pd.concat([data[numeric_cols], encoded], axis=1)
    return data

# Exploratory Data Analysis (EDA)
def eda(data):
    """
    Perform basic exploratory data analysis (EDA) to understand the dataset.
    """
    print("Basic Statistics:\n", data.describe())
    print("Missing Values:\n", data.isnull().sum())
    sns.pairplot(data.sample(min(100, len(data))), diag_kind='kde')
    plt.show()

# Dimensionality Reduction Methods
def apply_pca(data):
    """
    Apply PCA to reduce dimensionality to 2 components and return the transformed data.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_scaled)
    print("Explained Variance Ratio (PCA):", pca.explained_variance_ratio_)
    return reduced

def apply_lda(data, target):
    """
    Apply Linear Discriminant Analysis (LDA) to reduce dimensionality to 2 components and return the transformed data.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    lda = LDA(n_components=2)
    reduced = lda.fit_transform(data_scaled, target)
    return reduced

def apply_tsne(data):
    """
    Apply t-SNE to reduce dimensionality to 2 components and return the transformed data.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(data_scaled)
    return reduced

def apply_svd(data):
    """
    Apply SVD to reduce dimensionality to 2 components and return the transformed data.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    svd = TruncatedSVD(n_components=2)
    reduced = svd.fit_transform(data_scaled)

    # Reconstruct the data using the SVD components
    reconstructed_data = svd.inverse_transform(reduced)
    
    # Calculate reconstruction error
    reconstruction_error = np.mean((data_scaled - reconstructed_data) ** 2)
    print(f"Reconstruction Error (SVD): {reconstruction_error}")
    
    return reduced

def apply_mds(data):
    """
    Apply MDS (Multidimensional Scaling) to reduce dimensionality to 2 components and return the transformed data.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    mds = MDS(n_components=2, random_state=42)
    reduced = mds.fit_transform(data_scaled)
    return reduced

# Performance Evaluation
def evaluate_performance(method, reduced_data, original_data, target=None):
    """
    Evaluate the performance of the dimensionality reduction method.
    """
    if method == 'pca':
        # Print explained variance for PCA
        pca = PCA(n_components=2)
        pca.fit(original_data)
        print("Explained Variance Ratio (PCA):", pca.explained_variance_ratio_)
    elif method == 'svd':
        # Compute reconstruction error for SVD
        svd = TruncatedSVD(n_components=2)
        svd.fit(original_data)
        reconstructed_data = svd.inverse_transform(reduced_data)
        reconstruction_error = np.mean((original_data - reconstructed_data) ** 2)
        print(f"Reconstruction Error (SVD): {reconstruction_error}")
    elif method == 'lda' and target is not None:
        # Check classification accuracy for LDA
        clf = LogisticRegression()
        clf.fit(reduced_data, target)
        accuracy = accuracy_score(target, clf.predict(reduced_data))
        print(f"Classification Accuracy (LDA): {accuracy}")
    elif method == 'tsne':
        # Print perplexity for t-SNE
        print(f"Perplexity used in t-SNE: {tsne.perplexity_}")

# Visualization
def visualize_reduction(reduced_data, title):
    """
    Visualize the reduced data using a scatter plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6)
    plt.title(title)
    plt.show()

# Load and reduce data
def Load_and_reduce():
    try:
        filename = input("Enter the file path: ")
        method = input("Choose method (pca/lda/tsne/svd/mds): ")
        data = pd.read_csv(filename)
        eda(data)
        data = sample_data(data)
        print("Available features in the dataset:", list(data.columns))
        target_col = input("Enter the target column name (if applicable, else leave blank): ")
        has_target = bool(target_col)
        data = preprocess_data(data)
        features = data.drop(columns=[target_col]) if has_target else data
        target = data[target_col] if has_target else None

        if method == 'pca':
            reduced = apply_pca(features)
            visualize_reduction(reduced, 'PCA')
            evaluate_performance('pca', reduced, features, target)
        elif method == 'lda' and has_target:
            reduced = apply_lda(features, target)
            visualize_reduction(reduced, 'LDA')
            evaluate_performance('lda', reduced, features, target)
        elif method == 'tsne':
            reduced = apply_tsne(features)
            visualize_reduction(reduced, 't-SNE')
            evaluate_performance('tsne', reduced, features)
        elif method == 'svd':
            reduced = apply_svd(features)
            visualize_reduction(reduced, 'SVD')
            evaluate_performance('svd', reduced, features)
        elif method == 'mds':
            reduced = apply_mds(features)
            visualize_reduction(reduced, 'MDS')
            evaluate_performance('mds', reduced, features)
        else:
            print("Invalid method or missing target.")
    except Exception as e:
        print(f"Error: {e}")

# Run the tool
Load_and_reduce()
