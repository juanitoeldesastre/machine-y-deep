from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# 1. Cargar el conjunto de datos MNIST reducido
digits = load_digits()
X = digits.data  # Imágenes en formato de matriz aplanada
y = digits.target  # Etiquetas reales (solo para evaluar, no usadas en K-Means)

# 2. Preprocesamiento: Reducir la dimensionalidad para visualización
pca = PCA(2)  # Reducción a 2 dimensiones
X_pca = pca.fit_transform(X)

# 3. Crear el modelo K-Means con 10 clusters
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X)  # Entrenar el modelo con las imágenes aplanadas
clusters = kmeans.predict(X)  # Obtener las etiquetas de los clusters

# 4. Visualización de los clusters en 2D
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=10)
plt.colorbar(scatter, label='Cluster')
plt.title('Clustering de Dígitos con K-Means (Reducido a 2D con PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# 5. Mostrar ejemplos de cada cluster
for i in range(10):
    cluster_indices = np.where(clusters == i)[0]
    examples = X[cluster_indices[:10]]  # Tomar los primeros 10 ejemplos de cada cluster
    plt.figure(figsize=(10, 1))
    for j, example in enumerate(examples):
        plt.subplot(1, 10, j + 1)
        plt.imshow(example.reshape(8, 8), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Cluster {i} - Ejemplos de Imágenes')
    plt.show()
