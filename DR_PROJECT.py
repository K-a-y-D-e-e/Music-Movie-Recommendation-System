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

filename = "C:/Academics/Project/USL/final_dataset.csv"
data = pd.read_csv(filename)

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

# 🔹 **Updated EDA Function (Optimized for Memory)**
def eda(data):
    print("\n=== Basic Dataset Information ===")
    print("Shape:", data.shape)
    print("Column Information:")
    print(data.info())
    print("\nSummary Statistics:")
    print(data.describe())
    print("\nMissing Values Count:\n", data.isnull().sum())

    # 🔹 **Visualizing Missing Data**
    plt.figure(figsize=(10, 5))
    sns.heatmap(data.isnull(), cmap='viridis', cbar=False)
    plt.title("Missing Values Heatmap")
    plt.show()

    # 🔹 **Handling Missing Data**
    data = data.dropna()  # Drop rows with missing values

    # 🔹 **Feature Distribution (For Numeric Columns)**
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # 🔹 **Convert float64 to float32 (Memory Optimization)**
    data[numeric_cols] = data[numeric_cols].astype(np.float32)

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()

    # 🔹 **Handling Categorical Features (Frequency Encoding Instead of One-Hot)**
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if categorical_cols:
        for col in categorical_cols:
            freq_map = data[col].value_counts(normalize=True).to_dict()
            data[col] = data[col].map(freq_map)  # Convert categorical values to frequency

    # 🔹 **Scaling Numeric Features**
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Convert back to DataFrame
    processed_data = pd.DataFrame(data_scaled, columns=data.columns)

    print("\n✅ Data Preprocessing Complete. Cleaned Data Stored in `processed_data`.")

    return processed_data


# Apply PCA with 2 components
def apply_pca(processed_data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(processed_data)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_scaled)
    print("Explained Variance Ratio (PCA):", pca.explained_variance_ratio_)
    return reduced

# Apply LDA
def apply_lda(processed_data, target_variable):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(processed_data.drop(columns=[target_variable]))
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    processed_data['rating_category'] = discretizer.fit_transform(processed_data[[target_variable]])
    lda = LDA(n_components=2)
    reduced = lda.fit_transform(data_scaled, processed_data['rating_category'])
    print("Explained Variance Ratio (LDA):", lda.explained_variance_ratio_)
    return reduced

# Apply SVD with 2 components
def apply_svd(processed_data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(processed_data)
    svd = TruncatedSVD(n_components=2)
    reduced = svd.fit_transform(data_scaled)
    print("Explained Variance Ratio (SVD):", svd.explained_variance_ratio_)
    return reduced

# Apply t-SNE
def apply_tsne(processed_data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(processed_data)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
    reduced = tsne.fit_transform(data_scaled)
    print("t-SNE reduction completed.")
    return reduced

# Apply MDS
def apply_mds(processed_data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(processed_data)
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
        print("Dataset loaded.")
        
        # Features to be used for processing
        features = data[['year', 'duration', 'rating', 'votes', 'meta_score', 'budget', 'opening_weekend_gross', 'gross_worldwide', 'gross_us_canada']]
        
        # Apply updated EDA
        processed_data = eda(features)
        dataset_description(processed_data)

        # PCA Visualization
        reduced_pca = apply_pca(processed_data)
        visualize_reduction(reduced_pca, "PCA")

        # LDA Visualization
        reduced_lda = apply_lda(processed_data, 'rating')
        visualize_reduction(reduced_lda, "LDA")

        # SVD Visualization
        reduced_svd = apply_svd(processed_data)
        visualize_reduction(reduced_svd, "SVD")

        # t-SNE Visualization
        reduced_tsne = apply_tsne(processed_data)
        visualize_reduction(reduced_tsne, "t-SNE")

        # MDS Visualization
        reduced_mds = apply_mds(processed_data)
        visualize_reduction(reduced_mds, "MDS")

    except Exception as e:
        print(f"Error: {e}")

# Run the script
if __name__ == "__main__":
    main()
