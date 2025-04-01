# Dimensionality Reduction Techniques in Python

## Overview

This project presents an implementation of several popular dimensionality reduction techniques for high-dimensional datasets, including **Principal Component Analysis (PCA)**, **Linear Discriminant Analysis (LDA)**, **t-Distributed Stochastic Neighbor Embedding (t-SNE)**, **Truncated Singular Value Decomposition (SVD)**, and **Multi-Dimensional Scaling (MDS)**. The goal of this implementation is to provide a clear and effective way of reducing the dimensionality of a dataset while preserving the essential structure of the data. This is crucial in machine learning and data analysis, as working with high-dimensional data can be computationally expensive and complex.

### Techniques Implemented:
1. **PCA (Principal Component Analysis)**
2. **LDA (Linear Discriminant Analysis)**
3. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
4. **SVD (Singular Value Decomposition)**
5. **MDS (Multi-Dimensional Scaling)**

### Key Features:
- Each technique is implemented as a separate function that scales the data before applying the dimensionality reduction method.
- A user-friendly interface allows for selecting the method to be applied, providing flexibility for different scenarios.
- The resulting reduced data is visualized in a 2D scatter plot, allowing for intuitive understanding of the dimensionality reduction output.
- The ability to work with datasets that may or may not contain target columns, offering support for both supervised and unsupervised learning approaches.

---

## Purpose

Dimensionality reduction is an essential tool for data analysis and machine learning. It helps with:
- **Improving Computational Efficiency**: Reducing the number of features can improve model training and performance, especially with large datasets.
- **Visualizing Data**: By reducing high-dimensional data to 2D or 3D, we can better understand the structure and relationships between data points.
- **Removing Redundancy**: Reducing dimensionality helps to remove noisy and redundant features, which may improve model performance.

The project aims to implement several dimensionality reduction techniques to explore their individual characteristics and effects on the dataset. These techniques are widely used in the machine learning and data science fields for tasks such as feature extraction, data preprocessing, and visualization.

---

## Techniques and Implementation

### 1. **Principal Component Analysis (PCA)**

PCA is a widely-used method for reducing the dimensionality of data by projecting it onto new orthogonal axes, called principal components, which capture the largest variance in the data. This technique is unsupervised and purely based on the data structure.

**Key Steps**:
- Standardize the dataset.
- Apply PCA and extract the top principal components (in this case, two components).
- Visualize the reduced data.

```python
def apply_pca(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_scaled)
    return reduced
```

### 2. ** Linear Discriminant Analysis (LDA)**
LDA is a supervised technique that reduces dimensionality while maximizing the separability between known categories (classes). LDA uses class labels to find the optimal projection that maximizes the distance between the means of the classes while minimizing the spread within each class.

**Key Steps**:
- Standardize the dataset.
- Apply LDA, using the target class labels to guide the projection.
- Visualize the resulting 2D data.

```python
def apply_lda(data, target):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    lda = LDA(n_components=2)
    reduced = lda.fit_transform(data_scaled, target)
    return reduced
```

### 3. t-Distributed Stochastic Neighbor Embedding (t-SNE))**
t-SNE is a non-linear technique primarily used for the visualization of high-dimensional data. It preserves local similarities by converting pairwise distances into probabilities and attempts to map these probabilities into a lower-dimensional space, often revealing patterns and clusters.

**Key Steps**:
- Standardize the dataset.
- Apply t-SNE to reduce the data to 2 dimensions.
- Visualize the reduced data.

```python
def apply_tsne(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(data_scaled)
    return reduced
```

### 4. Truncated Singular Value Decomposition (SVD)**
SVD is a matrix factorization method used for dimensionality reduction. It decomposes a matrix into three matrices, and the components are then used to reconstruct the data with fewer dimensions. SVD is particularly useful for sparse datasets.

**Key Steps**:
- Standardize the dataset.
- Apply Truncated SVD to reduce the dimensions.
- Visualize the reduced data.
  
```python
def apply_svd(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    svd = TruncatedSVD(n_components=2)
    reduced = svd.fit_transform(data_scaled)
    return reduced
```

### 5. Multi-Dimensional Scaling (MDS)*8
MDS is a non-linear technique used to visualize the similarity or dissimilarity between data points. It seeks to represent the pairwise distances between data points in a lower-dimensional space, while minimizing the stress (error) between the original distances and the reduced distances.

**Key Steps**:
- Standardize the dataset.
- Apply MDS to reduce the data to 2 dimensions.
- Visualize the resulting reduced data.

```python
def apply_mds(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    mds = MDS(n_components=2, random_state=42)
    reduced = mds.fit_transform(data_scaled)
    return reduced
```

### Usage**

- Input: The user is prompted to input the path to the dataset (in CSV format) and select the desired dimensionality reduction technique.
- Output: A 2D scatter plot representing the reduced data will be displayed.
- Data: The dataset can either have a target column (for supervised techniques like LDA) or not (for unsupervised techniques like PCA, t-SNE, and SVD).
- Data Sampling: For performance reasons, only a random sample of 5000 entries from the dataset will be used for processing.
```python
Example:
Load_and_reduce()
# This function will prompt for a file path, allow you to choose a method (PCA, LDA, t-SNE, SVD, or MDS), and visualize the result.
```

### Conclusion and Future Work**

The implementation provides a solid foundation for understanding and applying key dimensionality reduction techniques in data analysis. While we have gained a deeper understanding of these methods and their applications, we recognize that this is only the beginning. There are many other dimensionality reduction techniques, such as Isomap, Autoencoders, and t-SNE with perplexity tuning, which could be incorporated to further enhance this toolkit.

We are eager to expand my knowledge of more advanced techniques and their optimizations, and to continue exploring how these methods can be leveraged in more complex and diverse datasets. We are excited to apply these techniques in a variety of contexts and learn more about the nuanced trade-offs involved in selecting the best dimensionality reduction method for different use cases.

### Requirements**

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
You can install the required dependencies using pip:
```python
pip install pandas numpy matplotlib scikit-learn
```
