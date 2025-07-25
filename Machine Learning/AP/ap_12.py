# -*- coding: utf-8 -*-
"""AP_12.ipynb


# Vamos aplicar o que aprendemos sobre a classificação hierárquica
*   Complete com o código em falta
*   Sempre que necessário, **substitua** ___

---


[documentação](https://scikit-learn.org/stable/index.html) <br>
dataset: [fonte](https://www.kaggle.com/datasets/dongeorge/seed-from-uci?select=Seed_Data.csv)

---

>[Vamos aplicar o que aprendemos sobre a classificação hierárquica](#scrollTo=7UqvAzOuK_SN)

>[1. Corra a primeira célula de código para obter o dataframe com que vamos trabalhar](#scrollTo=25DtwghMIQqJ)

>>[1.1. analise o dataframe](#scrollTo=0CBwwjoI0v0l)

>[2. Aplique o algoritmo de classificação hierárquica às variáveis](#scrollTo=glt7JqnNdHxk)

>[3. Aplique o algoritmo de classificação hierárquica às observações](#scrollTo=Rq7QOxUC9Y4u)

#1.&nbsp;Corra a primeira célula de código para obter o dataframe com que vamos trabalhar
"""

# faça o upload do ficheiro csv associado à atividade

# vamos importar a biblioteca
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# vamos importar o dataframe do ficheiro .csv
df_seed = pd.read_csv("Seed_Data.csv")

"""## 1.1.&nbsp;analise o dataframe"""

# veja as 5 primeiras linhas do dataframe
df_seed.head()
# veja a forma do dataframe: quantas linhas, quantas colunas?
df_seed.shape
# veja a informação sobre o dataframe
df_seed.info()
# veja a descrição das variáveis numéricas
df_seed.describe()

# remova a última coluna do dataframe (target)
target_variable = "target"
X = df_seed.drop([target_variable], axis = 1)

"""#2.&nbsp;Aplique o algoritmo de classificação hierárquica às variáveis"""

# importe o modelo
from ___ import ___, ___, ___

# utilize a transposta (para termos as variáveis como linhas)
X_t = ___

# aplique o modelo, escolhendo o método single e a medida de distância "correlação"
clusters_sl =___
dendrogram(___,
           ___ = X.columns,
           ___ = 90,
           ___ = 6)
plt.show()

# aplique o modelo, escolhendo o método complete e a medida de distância "correlação"
clusters_cl = ___
dendrogram(___)
plt.show()

# aplique o modelo, escolhendo o método average e a medida de distância "correlação"
clusters_avg =___
dendrogram(___)
plt.show()

# veja as labels para o complete e com um corte a 0.6
labels = fcluster(___, criterion = 'distance')
labels

"""#3.&nbsp;Aplique o algoritmo de classificação hierárquica às observações"""

# importe o modelo
from ___

# aplique o modelo, escolhendo o método single e a medida de distância "euclidiana"
clusters_sl = ___
dendrogram(___,
           ___,
           ___ = 90,
           ___ = 6)
plt.show()

# aplique o modelo, escolhendo o método complete e a medida de distância "euclidiana"
clusters_cl = ___
dendrogram(___,
           ___ = X.index,
           ___ = 90,
           ___ = 6)
plt.show()

# aplique o modelo, escolhendo o método average e a medida de distância "euclidiana"
clusters_avg = ___
dendrogram(___)
plt.show()

# veja as labels para o complete e com um corte a 8
labels = ___
print(labels)

# vamos comparar com as classes
df_labels = pd.DataFrame({'labels': labels,
                         'classes': df_seed[target_variable]})
df_labels_ctab = pd.crosstab(df_labels['labels'], df_labels['classes'])
df_labels_ctab