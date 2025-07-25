# -*- coding: utf-8 -*-
"""AP_11.ipynb


# Vamos aplicar o que aprendemos sobre Kmeans
*   Complete com o código em falta
*   Sempre que necessário, **substitua** ___

Corra a primeira célula de código para obter o dataframe com que vamos trabalhar
"""

# faça o upload do ficheiro csv associado à atividade

# vamos importar a biblioteca
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# vamos importar o dataset das iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns = iris.feature_names)

# veja as 5 primeiras linhas do dataframe
X.head()
# veja a forma do dataframe: quantas linhas, quantas colunas?
X.shape
# veja a informação sobre o dataframe
X.info()
# veja a descrição das variáveis numéricas
X.describe()

# aplique o modelo de KMeans com 6 clusters
from ___ import ___
kmeans = ___(___, n_init = 10, random_state=42)
___
labels = ___

# verifique o melhor K com o elbow method
inertias = []
K = range(1, 11)

for k in K:
    kmeans = ___
    ___
    inertias.append(___)

# faça plot do elbow method
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

# Escolha o número óptimo de clusters
optimal_k = ___
kmeans = ___(___, random_state=42, n_init = 10)
y_kmeans = ___

# avalie o silhouette score
from ___
silhouette_avg = ___
print(f'Silhouette Score for K={optimal_k}: {silhouette_avg}')

# faça plot dos resultados
___
silhouette_vals = ___

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