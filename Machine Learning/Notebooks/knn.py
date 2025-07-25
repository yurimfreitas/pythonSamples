# -*- coding: utf-8 -*-
"""KNN.ipynb

# Modelo de KNN

---


[documentação](https://scikit-learn.org/stable/index.html) <br>
dataset: [fonte](https://www.kaggle.com/datasets/dinhanhx/studentgradepassorfailprediction)

---

>[Modelo de KNN](#scrollTo=QoBv84MIUa-h)

>>[1. Vamos começar por importar os packages e o dataset](#scrollTo=HO6jdFDmldaU)

>>[2. Vamos explorar o dataset](#scrollTo=2S0UrVDEf8E-)

>>[3. Vamos remover 10 elementos para teste](#scrollTo=VlyljBsUgD8D)

>>[4. Vamos aplicar o modelo KNN](#scrollTo=rBbqqMS5hXRe)

>>[5. Vamos dividir em treino e teste com a ajuda do scikit-learn](#scrollTo=zXdiWEUjmzkX)

## 1.&nbsp;Vamos começar por importar os packages e o dataset
"""

# packages gerais
import pandas as pd
import numpy as np

# dataset
df_students = pd.read_csv("student-mat-pass-or-fail.csv")

"""## 2.&nbsp;Vamos explorar o dataset"""

# exploração inicial
df_students.head()
df_students.info()
df_students.shape
df_students.describe()

"""## 3.&nbsp;Vamos remover 10 elementos para teste"""

# definimos a variável alvo
target_variable = "pass"

# train_test split manual
test_elements = 10
X_train = (df_students
           .drop(["G3", target_variable], axis = 1)
           .iloc[:-test_elements,:])
y_train = df_students[target_variable].iloc[:-test_elements]
X_test = (df_students
           .drop(["G3", target_variable], axis = 1)
           .iloc[-test_elements:,:])
y_test = df_students[target_variable].iloc[-test_elements:]

"""## 4.&nbsp;Vamos aplicar o modelo KNN"""

# importamos o modelo
from sklearn.neighbors import KNeighborsClassifier

# instanciar o objeto que contém o modelo KNN e atribuir K = 7
knn = KNeighborsClassifier(n_neighbors = 7)
# knn = KNeighborsClassifier(n_neighbors = 7, p = 1)

# vamos aplicar o método .fit para "ajustar" o modelo aos nossos dados
knn.fit(X_train, y_train)

"""Notas:
*   Entregamos ao método .fit dois argumentos -> as características e os rótulos;
*   O scikit-learn apenas recebe NumPy arrays ou dataframes;
*   As características têm de conter valores contínuos;
*   Não podem existir valores em falta;


"""

# Vamos aplicar o modelo e prever para os dados de teste (método .predict())
y_pred = knn.predict(X_test)

#Vamos ver os resultados
print('Previsões: {}'.format(y_pred))
print('Reais:     {}'.format(y_test.values))

# para verificar a accuracy do modelo aplicamos a função .score
knn.score(X_test, y_test)

"""## 5.&nbsp;Vamos dividir em treino e teste com a ajuda do scikit-learn"""

# train_test split usando a função train_test_split
X = df_students.drop(["G3", target_variable], axis = 1)
y = df_students[target_variable]
y.sum()/len(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 12,
                                                    stratify = y)

# importamos o modelo
from sklearn.neighbors import KNeighborsClassifier

# instanciar o objeto que contém o modelo KNN e atribuir K = 7
knn = KNeighborsClassifier(n_neighbors = 7)
# knn = KNeighborsClassifier(n_neighbors = 7, p = 1)

# vamos aplicar o método .fit para "ajustar" o modelo aos nossos dados
knn.fit(X_train, y_train)

# Vamos aplicar o modelo e prever para os dados de teste (método .predict())
y_pred = knn.predict(X_test)

#Vamos ver os resultados
print('Previsões: {}'.format(y_pred))
print('Reais:     {}'.format(y_test.values))

# para verificar a accuracy do modelo aplicamos a função .score
knn.score(X_test, y_test)