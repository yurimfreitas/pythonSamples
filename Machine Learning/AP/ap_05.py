# -*- coding: utf-8 -*-
"""AP_05.ipynb

# Vamos aplicar o que aprendemos sobre o modelo de KNN

*   Complete com o código em falta
*   Sempre que necessário, **substitua** ___

---

dataset: [fonte](https://www.kaggle.com/datasets/charleyhuang1022/contract-renewal-prediction?select=South_China.csv)

---

Corra a primeira célula de código para obter o dataframe com que vamos trabalhar
"""

# faça o upload do ficheiro csv associado à atividade

# vamos importar a biblioteca
import pandas as pd

# vamos importar o dataframe do ficheiro .csv
df_contract_renewal = pd.read_csv("South_China.csv")

# veja as 5 primeiras linhas do dataframe
df_contract_renewal.___

# veja a forma do dataframe: quantas linhas, quantas colunas?
df_contract_renewal.___

# veja a informação sobre o dataframe
df_contract_renewal.___

# veja a descrição das variáveis numéricas
df_contract_renewal.___

# defina a variável alvo como sendo a coluna "Renewal"
target_variable = ___

# separe manualmente os dados de teste e de treino
test_elements = 10

X_train = (df_contract_renewal
           .drop([___, "ID"], axis = 1)
           .iloc[:-test_elements,:])
y_train = df_contract_renewal[___].iloc[:-test_elements]*1
X_test = (df_contract_renewal
           .drop([___, "ID"], axis = 1)
           .iloc[-test_elements:,:])
y_test = df_contract_renewal[___].iloc[-test_elements:]*1

# importe o modelo de KNeighborsClassifier
from sklearn.neighbors import ___

# instancie o objeto que contém o modelo KNN e atribuir K = 7
knn = ___

# aplique o método .fit para "ajustar" o modelo aos nossos dados
knn.fit(___, ___)

# aplique o modelo e prever para os dados de teste (método .predict())
y_pred = knn.predict(___)

#Vamos ver os resultados
print('Previsões: {}'.format(y_pred))
print('Reais:     {}'.format(y_test.values))

# defina a variável alvo como sendo a coluna "Renewal"
target_variable = ____

# train_test split usando a função train_test_split
X = df_contract_renewal.drop(["ID", ___], axis = 1)
y = ___[___]*1

# verifique o grau de desequilibrio
print(___ / ___)

# importe a função train_test_split e defina X_train, X_test, y_train, y_test
from sklearn.model_selection import ___

X_train, ___, ___, ___ = ___(___, ___,
                             ___ = 0.2,
                             random_state = 12,
                             stratify = ___)

# instancie o objeto que contém o modelo KNN e atribuir K = 7
knn = KNeighborsClassifier(___ = 7)

# aplique o método .fit para "ajustar" o modelo aos nossos dados
knn.fit(___, ___)

# aplique o modelo e prever para os dados de teste (método .predict())
y_pred = knn.predict(___)

# vamos ver os resultados
print('Previsões: {}'.format(y_pred))
print('Reais:     {}'.format(y_test.values))

# verifique a accuracy do modelo
knn.score(___, ___)