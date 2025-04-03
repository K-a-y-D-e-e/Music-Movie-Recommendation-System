# Importing the Required Libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import KBinsDiscretizer

# Set Seaborn style for better visualization
sns.set(style='whitegrid')

# Dataset Description
def dataset_description(data):
    print('Dataset Shape:', data.shape)
    print('Column Information:')
    print(data.info())
    print('Summary Statistics:')
    print(data.describe())
    print('Missing Values:')
    print(data.isnull().sum())
    print('Data Types:')
    print(data.dtypes)
    print('Head of the Dataset:')
    print(data.head())

# Data Pre-processing
def preprocess_data(data):
    # Drop rows with missing values before encoding
    data = data.dropna()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    # One-hot encode categorical columns, if any
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
        data = pd.concat([data[numeric_cols], encoded], axis=1)

    # Final check for missing values after encoding
    data.dropna(inplace=True)
    return data


# Apply PCA with 2 components
def apply_pca(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_scaled)
    print("Explained Variance Ratio (PCA):", pca.explained_variance_ratio_)
    return reduced

# Apply LDA
def apply_lda(data, target_variable):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.drop(columns=[target_variable]))
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    data['rating_category'] = discretizer.fit_transform(data[[target_variable]])
    lda = LDA(n_components=2)
    reduced = lda.fit_transform(data_scaled, data['rating_category'])
    print("Explained Variance Ratio (LDA):", lda.explained_variance_ratio_)
    return reduced

# Apply SVD with 2 components
def apply_svd(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    svd = TruncatedSVD(n_components=2)
    reduced = svd.fit_transform(data_scaled)
    print("Explained Variance Ratio (SVD):", svd.explained_variance_ratio_)
    return reduced

# Apply t-SNE
def apply_tsne(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    reduced = tsne.fit_transform(data_scaled)
    print("t-SNE reduction completed.")
    return reduced

# Apply MDS
def apply_mds(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    mds = MDS(n_components=2)
    reduced = mds.fit_transform(data_scaled)
    print("MDS reduction completed.")
    return reduced

# Visualization function
def visualize_reduction(reduced, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
    plt.title(title)
    plt.show()
    

# Set Seaborn style for better visualization
sns.set(style='whitegrid')

# Dataset Description
def dataset_description(data):
    print('Dataset Shape:', data.shape)
    print('Column Information:')
    print(data.info())
    print('Summary Statistics:')
    print(data.describe())
    print('Missing Values:')
    print(data.isnull().sum())
    print('Data Types:')
    print(data.dtypes)
    print('Head of the Dataset:')
    print(data.head())

'''
# EDA
def exploratory_data_analysis(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
    sns.pairplot(data, diag_kind='kde')
    plt.show()
'''

# Data Pre-processing
def preprocess_data(data):
    # Drop rows with missing values before encoding
    data = data.dropna()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    # One-hot encode categorical columns, if any
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
        data = pd.concat([data[numeric_cols], encoded], axis=1)

    # Final check for missing values after encoding
    data.dropna(inplace=True)
    return data

# Apply PCA with 2 components
def apply_pca(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_scaled)
    print("Explained Variance Ratio (PCA):", pca.explained_variance_ratio_)
    return reduced

# Apply LDA
def apply_lda(data, target_variable):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.drop(columns=[target_variable]))
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    data['rating_category'] = discretizer.fit_transform(data[[target_variable]])
    lda = LDA(n_components=2)
    reduced = lda.fit_transform(data_scaled, data['rating_category'])
    print("Explained Variance Ratio (LDA):", lda.explained_variance_ratio_)
    return reduced

# Apply SVD with 2 components
def apply_svd(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    svd = TruncatedSVD(n_components=2)
    reduced = svd.fit_transform(data_scaled)
    print("Explained Variance Ratio (SVD):", svd.explained_variance_ratio_)
    return reduced

# Apply t-SNE
def apply_tsne(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    reduced = tsne.fit_transform(data_scaled)
    print("t-SNE reduction completed.")
    return reduced

# Apply MDS
def apply_mds(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    mds = MDS(n_components=2)
    reduced = mds.fit_transform(data_scaled)
    print("MDS reduction completed.")
    return reduced

# Visualization function
def visualize_reduction(reduced, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
    plt.title(title)
    plt.show()

# Main execution function
def main():
    try:
        filename = "/content/drive/MyDrive/final_dataset.csv"
        data = pd.read_csv(filename)
        print("Dataset loaded.")
        features = data[['year', 'duration', 'rating', 'votes', 'meta_score', 'budget', 'opening_weekend_gross', 'gross_worldwide', 'gross_us_canada']]
        data = preprocess_data(features)
        dataset_description(data)

        # PCA Visualization
        reduced_pca = apply_pca(data)
        visualize_reduction(reduced_pca, "PCA")

        # LDA Visualization
        reduced_lda = apply_lda(data, 'rating')
        visualize_reduction(reduced_lda, "LDA")

        # SVD Visualization
        reduced_svd = apply_svd(data)
        visualize_reduction(reduced_svd, "SVD")

        # t-SNE Visualization
        reduced_tsne = apply_tsne(data)
        visualize_reduction(reduced_tsne, "t-SNE")

        # MDS Visualization
        reduced_mds = apply_mds(data)
        visualize_reduction(reduced_mds, "MDS")

    except Exception as e:
        print(f"Error: {e}")

# Run the script
if __name__ == "__main__":
    main()
