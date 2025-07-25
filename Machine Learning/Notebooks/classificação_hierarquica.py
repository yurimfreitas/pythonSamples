# -*- coding: utf-8 -*-
"""classificação hierarquica.ipynb


# Modelo de Classificação Hierárquica

---


[documentação](https://scikit-learn.org/stable/index.html) <br>
[documentação](https://docs.scipy.org/doc/scipy/index.html) <br>
dataset: [fonte](https://archive.ics.uci.edu/ml/datasets/wine)

---

>[Modelo de Classificação Hierárquica](#scrollTo=QoBv84MIUa-h)

>>[1. Vamos começar por importar os packages e o dataset](#scrollTo=HO6jdFDmldaU)

>>[2. Vamos explorar o dataset](#scrollTo=2S0UrVDEf8E-)

>>[3. Vamos visualizar a informação](#scrollTo=lYnaR0LYO-Sg)

>>[4. Vamos aplicar o modelo de Classificação Hierárquica às variáveis](#scrollTo=BKzodGb8R7t9)

>>[5. Vamos aplicar o modelo de Classificação Hierárquica às observações](#scrollTo=VN3ei0tZ9W0o)

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
# X.head()
# X.info()
# X.shape
X.describe()

"""## 3.&nbsp;Vamos visualizar a informação

"""

# importamos o matplotlib.pyplot
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#definimos as variáveis que queremos visualizar (magnesium e ash)
x_variable = X["magnesium"]
y_variable = X["ash"]
plt.scatter(x_variable, y_variable, c = wine.target)
plt.xlabel("Magnesium")
plt.ylabel('Ash')
plt.show()

"""## 4.&nbsp;Vamos aplicar o modelo de Classificação Hierárquica às variáveis"""

# importamos o modelo
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# vamos utilizar a transposta (para termos as variáveis como linhas)
X_t = X.values.transpose()

# aplicamos o modelo, escolhendo o método e a medida de distância
clusters_sl = linkage(X_t, method = 'single', metric = 'correlation')
dendrogram(clusters_sl,
           labels = X.columns,
           leaf_rotation = 90,
           leaf_font_size = 6)
plt.show()

clusters_cl = linkage(X_t, method = 'complete', metric = 'correlation')
dendrogram(clusters_cl,
           labels = X.columns,
           leaf_rotation = 90,
           leaf_font_size = 6)
plt.show()

clusters_avg = linkage(X_t, method = 'average', metric = 'correlation')
dendrogram(clusters_avg,
           labels = X.columns,
           leaf_rotation = 90,
           leaf_font_size = 6)
plt.show()

# Se quisermos ver as labels
labels = fcluster(clusters_avg, 0.8, criterion = 'distance')
labels

"""## 5.&nbsp;Vamos aplicar o modelo de Classificação Hierárquica às observações"""

# importamos o modelo
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# aplicamos o modelo, escolhendo o método e a medida de distância
clusters_sl = linkage(X, method = 'single', metric = 'euclidean')
dendrogram(clusters_sl,
           labels = X.index,
           leaf_rotation = 90,
           leaf_font_size = 6)
plt.show()

clusters_cl = linkage(X, method = 'complete', metric = 'euclidean')
dendrogram(clusters_cl,
           labels = X.index,
           leaf_rotation = 90,
           leaf_font_size = 6)
plt.show()

clusters_avg = linkage(X, method = 'average', metric = 'euclidean')
dendrogram(clusters_avg,
           labels = X.index,
           leaf_rotation = 90,
           leaf_font_size = 6)
plt.show()

# Se quisermos ver as labels
labels = fcluster(clusters_avg, 350, criterion = 'distance')
labels

# vamos comparar com as classes
df_labels = pd.DataFrame({'labels': labels,
                         'classes': wine.target})
pd.crosstab(df_labels['labels'], df_labels['classes'])