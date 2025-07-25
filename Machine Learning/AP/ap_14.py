# -*- coding: utf-8 -*-
"""AP_14.ipynb

# Vamos aplicar o que aprendemos sobre pipelines
*   Complete com o código em falta
*   Sempre que necessário, **substitua** ___

---


[documentação](https://scikit-learn.org/stable/index.html) <br>
dataset: [fonte](https://www.kaggle.com/datasets/mirichoi0218/insurance)

---

>[Vamos aplicar o que aprendemos sobre pipelines](#scrollTo=7UqvAzOuK_SN)

>[1. Corra a primeira célula de código para obter o dataframe com que vamos trabalhar](#scrollTo=25DtwghMIQqJ)

>[2. Trate os dados em falta](#scrollTo=glt7JqnNdHxk)

>[3. Faça train_test_split](#scrollTo=Rq7QOxUC9Y4u)

>[4. Aplique o pipeline](#scrollTo=wM4XJr63vizy)

#1.&nbsp;Corra a primeira célula de código para obter o dataframe com que vamos trabalhar
"""

# faça o upload do ficheiro csv associado à atividade

# vamos importar a biblioteca
import pandas as pd
import numpy as np

# vamos importar o dataframe do ficheiro .csv
df = pd.read_csv("insurance.csv")

# veja as 5 primeiras linhas do dataframe
df.head()
# veja a forma do dataframe: quantas linhas, quantas colunas?
df.shape
# veja a informação sobre o dataframe
df.info()
# veja a descrição das variáveis numéricas
df.describe()

"""#2.&nbsp;Trate os dados em falta"""

# remova a coluna quase sem informação
df.___

# remova os dois casos simples
df.___

"""#3.&nbsp;Faça train_test_split"""

# defina a variável alvo
target_variable = ___

# train_test split usando a função train_test_split
X = df.___
y = df___

from ___
___(___,___,
                                                    test_size = 0.3,
                                                    random_state = 12)

"""#4.&nbsp;Aplique o pipeline"""

# importe os transformers
from sklearn.___ import ___
from sklearn.___ import ___
from sklearn.___ import ___
from sklearn.___ import ___
from sklearn.___ import ___

# defina o processamento das variáveis numéricas
numeric_features = ___
num_transformer = ___(___=[
    ('imputer', ___),
    ('scaler', ___)
])

# defina o processamento das variáveis categóricas
categorical_features =___
categorical_transformer = ___(___=[
    ('encoder', ___(drop='first', handle_unknown='ignore'))
])

# combine os processos
preprocessor = ___(___=[
    ('num',___),
    ('cat',___)
    ],
    remainder='passthrough'
)

# crie o pipeline com processamento e modelação
pipeline = ___(___=[
    ('preprocessor', ___),
    ('regressor', ___)
])


# importe a grid search
from sklearn.model_selection import ___

# defina os dados para a grid
param_grid = {
    'regressor__n_estimators': [25, 50, 100, 200],
    'regressor__max_depth': [2, 5, 10, 20]
}

# defina as métricas de scoring
scoring = {
    'mse': 'neg_mean_squared_error',
    'r2': 'r2'
}

# construa a grid search
grid_search = ___(___)

# faça o fit da grid search
grid_search.___(___, ___)

# veja os melhores parameters
print("Best parameters found:")
print(grid_search.___)

# obtenha o melhor modelo
best_model = grid_search.___

# faça predict usando o melhor modelo
y_pred = best_model.___(___)

# visualize os resíduos
from sklearn.___ import ___

display = PredictionErrorDisplay(y_true = ___, y_pred = ___)
display.plot()

# importe a lista de métricas
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

# MAE
mean_absolute_error(___)

# MSE
mean_squared_error(___)

# RMSE
np.sqrt(mean_squared_error(___))

# MAPE
mean_absolute_percentage_error(___)

# R2
r2_score(___)