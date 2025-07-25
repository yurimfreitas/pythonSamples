# -*- coding: utf-8 -*-
"""AP_05_solucoes.ipynb

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
df_contract_renewal.head()
# veja a forma do dataframe: quantas linhas, quantas colunas?
df_contract_renewal.shape
# veja a informação sobre o dataframe
df_contract_renewal.info()
# veja a descrição das variáveis numéricas
df_contract_renewal.describe()

# defina a variável alvo como sendo a coluna "Renewal"
target_variable = "Renewal"

# separe manualmente os dados de teste e de treino
test_elements = 10

X_train = (df_contract_renewal
           .drop([target_variable, "ID"], axis = 1)
           .iloc[:-test_elements,:])
y_train = df_contract_renewal[target_variable].iloc[:-test_elements]*1
X_test = (df_contract_renewal
           .drop([target_variable, "ID"], axis = 1)
           .iloc[-test_elements:,:])
y_test = df_contract_renewal[target_variable].iloc[-test_elements:]*1

# importe o modelo de KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# instancie o objeto que contém o modelo KNN e atribuir K = 7
knn = KNeighborsClassifier(n_neighbors = 7)

# aplique o método .fit para "ajustar" o modelo aos nossos dados
knn.fit(X_train, y_train)

# aplique o modelo e prever para os dados de teste (método .predict())
y_pred = knn.predict(X_test)

#Vamos ver os resultados
print('Previsões: {}'.format(y_pred))
print('Reais:     {}'.format(y_test.values))

# defina a variável alvo como sendo a coluna "Renewal"
target_variable = "Renewal"

# train_test split usando a função train_test_split
X = df_contract_renewal.drop(["ID", target_variable], axis = 1)
y = df_contract_renewal[target_variable]*1

# verifique o grau de desequilibrio
print(y.sum()/len(y))

# importe a função train_test_split e defina X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 12,
                                                    stratify = y)

# instancie o objeto que contém o modelo KNN e atribuir K = 7
knn = KNeighborsClassifier(n_neighbors = 7)

# aplique o método .fit para "ajustar" o modelo aos nossos dados
knn.fit(X_train, y_train)

# aplique o modelo e prever para os dados de teste (método .predict())
y_pred = knn.predict(X_test)

# vamos ver os resultados
print('Previsões: {}'.format(y_pred))
print('Reais:     {}'.format(y_test.values))

# verifique a accuracy do modelo
knn.score(X_test, y_test)