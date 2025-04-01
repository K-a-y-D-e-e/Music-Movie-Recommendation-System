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

### 1. ** Linear Discriminant Analysis (LDA)**
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



