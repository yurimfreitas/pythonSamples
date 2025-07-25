# -*- coding: utf-8 -*-
"""AP_09.ipynb

# Vamos aplicar o que aprendemos sobre o modelo de Regressão Logística

*   Complete com o código em falta
*   Sempre que necessário, **substitua** ___

---

dataset: [fonte](https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset)

---

Corra a primeira célula de código para obter o dataframe com que vamos trabalhar
"""

# faça o upload do ficheiro csv associado à atividade

# vamos importar a biblioteca
import pandas as pd

# vamos importar o dataframe do ficheiro .csv
df_contract_renewal = pd.read_csv("Employee.csv")

# veja as 5 primeiras linhas do dataframe
df_contract_renewal.head()
# veja a forma do dataframe: quantas linhas, quantas colunas?
df_contract_renewal.shape
# veja a informação sobre o dataframe
df_contract_renewal.info()
# veja a descrição das variáveis numéricas
df_contract_renewal.describe()

# vamos fazer o drop das colunas object
df_contract_renewal.___(___)

# defina a variável alvo como sendo a coluna "LeaveOrNot"
target_variable = ___

# train_test split usando a função train_test_split
X = df_contract_renewal.___(___)
y = df_contract_renewal[___]*1

# verifique o grau de desequilibrio
print(y.sum()/len(y))

# importe a função train_test_split e defina X_train, X_test, y_train, y_test
___
___ = ___(___, ___, ___ = 0.2, ___ = 12, ___)

# importe o modelo de DecisionTreeClassifier
from ___ import ___
clf = ___(___ = 'entropy')

# aplique o método .fit para "ajustar" o modelo aos dados
___(___, ___)

# aplique o modelo e preveja para os dados de teste (método .predict())
y_pred = ___(___)

# veja os resultados
print('Previsões: {}'.format(___))
print('Reais:     {}'.format(___))

# verifique a accuracy do modelo
___(___, ___)

# faça o upload do ficheiro csv associado à atividade

# vamos importar a biblioteca
import pandas as pd
import numpy as np

# vamos importar o dataframe do ficheiro .csv
df_real_estate = (pd
                  .read_csv("Real estate.csv")
                  .drop(["No",
                         "X1 transaction date",
                         "X4 number of convenience stores"], axis = 1))

# veja as 5 primeiras linhas do dataframe
df_real_estate.head()
# veja a forma do dataframe: quantas linhas, quantas colunas?
df_real_estate.shape
# veja a informação sobre o dataframe
df_real_estate.info()
# veja a descrição das variáveis numéricas
df_real_estate.describe()

# importe o matplotlib.pyplot
import matplotlib.pyplot as plt
plt.style.use('ggplot')
x_variable = df_real_estate["X5 latitude"]
y_variable = df_real_estate["Y house price of unit area"]
plt.scatter(x_variable, y_variable, color = 'b')
plt.ylabel("preço do m2 de habitação")
plt.xlabel('latitude')
plt.show()

# vamos codificar o nome das variáveis
columns_names = ["x1", "x2", "x3", "x4", "y"]
df_real_estate.columns = columns_names

# defina a variável alvo ("y")
target_variable = ___

# train_test split usando a função train_test_split
X = ___
y = ___

___
___ = ___(___, ___, ___ = 0.3, ___ = 12)

# importe o modelo de regressão linear
from ___ import ___
rf = ___()
rf.___(___, ___)
y_pred = ___

# para verificar a accuracy do modelo aplique a função .score
rf.___

# considere 500 árvores e um máximo de features = sqrt
rf = ___(___ = 500, ___ = 'sqrt')
rf.___(___, ___)
y_pred_2 = rf.___(___)

# para verificar a accuracy do modelo aplique a função .score
rf.___

#visualize as diferenças
plt.scatter(___, ___, color = 'b')
plt.scatter(___, y_pred_2, color = 'r')
plt.xlabel('valores reais')
plt.ylabel("valores previstos")
plt.show()