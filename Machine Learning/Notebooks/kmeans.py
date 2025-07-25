# -*- coding: utf-8 -*-
"""Kmeans.ipynb


# Modelo de KMeans

---


[documentação](https://scikit-learn.org/stable/index.html) <br>
---

>[Modelo de KMeans](#scrollTo=QoBv84MIUa-h)

>>[documentação](#scrollTo=DqMh54wYK1UQ)

>>[1. Vamos começar por importar os packages e o dataset](#scrollTo=HO6jdFDmldaU)

>>[2. Vamos explorar o dataset](#scrollTo=2S0UrVDEf8E-)

>>[3. Vamos aplicar o modelo de Kmeans](#scrollTo=BKzodGb8R7t9)

>>[4. Vamos calcular as inércias e aplicar o elbow method](#scrollTo=fUSKDMyr_7R_)

>>[5. Vamos avaliar o Silhouette score](#scrollTo=sHAG-N7uAsFi)

## 1.&nbsp;Vamos começar por importar os packages e o dataset
"""

# packages gerais
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine

# vamos importar o dataset das iris
wine = load_wine()
X = pd.DataFrame(wine.data, columns = wine.feature_names)

"""## 2.&nbsp;Vamos explorar o dataset"""

# exploração inicial
X.head()
# X.info()
# X.shape
# X.describe()

"""## 3.&nbsp;Vamos aplicar o modelo de Kmeans"""

# importamos o modelo
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, n_init = 10, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

"""## 4.&nbsp;Vamos calcular as inércias e aplicar o elbow method"""

# vamos aplicar o elbow method
inertias = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init = 10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# vamos visualizar os resultados
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

"""## 5.&nbsp;Vamos avaliar o Silhouette score"""

from sklearn.metrics import silhouette_score, silhouette_samples

# Vamos escolher o número óptimo de clusters
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init = 10)
y_kmeans = kmeans.fit_predict(X)

# vamos avaliar o silhouette score
silhouette_avg = silhouette_score(X, y_kmeans)
print(f'Silhouette Score for K={optimal_k}: {silhouette_avg}')

# vamos visualizar os resultados
silhouette_vals = silhouette_samples(X, y_kmeans)

plt.figure(figsize=(10, 6))
y_lower, y_upper = 0, 0
yticks = []

for i in range(optimal_k):
    cluster_silhouette_vals = silhouette_vals[y_kmeans == i]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0)
    yticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_vals)

plt.axvline(silhouette_avg, color='red', linestyle='--')
plt.yticks(yticks, [f'Cluster {i+1}' for i in range(optimal_k)])
plt.ylabel('Cluster')
plt.xlabel('Silhouette Coefficient')
plt.title(f'Silhouette Plot for K={optimal_k}')
plt.grid(True)
plt.show()