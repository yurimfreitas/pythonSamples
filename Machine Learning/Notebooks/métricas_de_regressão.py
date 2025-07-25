# -*- coding: utf-8 -*-
"""métricas de regressão.ipynb

# Métricas de desempenho

---


[documentação](https://scikit-learn.org/stable/index.html) <br>
dataset: [fonte](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction?select=CarPrice_Assignment.csv)

---

>[Métricas de desempenho](#scrollTo=QoBv84MIUa-h)

>>[1. Vamos começar por importar os packages e o dataset](#scrollTo=HO6jdFDmldaU)

>>[2. Vamos explorar o dataset](#scrollTo=2S0UrVDEf8E-)

>>[3. Vamos visualizar a informação](#scrollTo=lYnaR0LYO-Sg)

>>[4. Vamos aplicar o modelo de regressão linear](#scrollTo=4CJL3vzRZ88m)

>>[5. Vamos aplicar as diferentes métricas](#scrollTo=nfZM1HZtrxGt)

## 1.&nbsp;Vamos começar por importar os packages e o dataset
"""

# packages gerais
import pandas as pd
import numpy as np

# dataset
df_car_price = pd.read_csv("CarPrice_Assignment.csv")

"""## 2.&nbsp;Vamos explorar o dataset"""

# exploração inicial
df_car_price.head()
# df_car_price.info()
# df_car_price.shape
# df_car_price.describe()

"""## 3.&nbsp;Vamos visualizar a informação

"""

# importamos o matplotlib.pyplot
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#definimos as variáveis que queremos visualizar (enginesize e price)
x_variable = df_car_price["enginesize"]
y_variable = df_car_price["price"]
plt.scatter(x_variable, y_variable, color = 'b')
plt.ylabel("preço do veículo")
plt.xlabel('dimensão do motor')
plt.show()

"""## 4.&nbsp;Vamos aplicar o modelo de regressão linear"""

# definimos a variável alvo
target_variable = "price"

# train_test split usando a função train_test_split
# -> não consideramos stratification

X = df_car_price.drop([target_variable], axis = 1)
y = df_car_price[target_variable]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 12)

# importamos o modelo
from sklearn.linear_model import LinearRegression
l_reg = LinearRegression()
l_reg.fit(X_train, y_train)
y_pred_lr = l_reg.predict(X_test)

# para verificar a accuracy do modelo aplicamos a função .score
l_reg.score(X_test, y_test) #R2

"""## 5.&nbsp;Vamos aplicar as diferentes métricas"""

# Vamos visualizar os resíduos
from sklearn.metrics import PredictionErrorDisplay

display = PredictionErrorDisplay(y_true = y_test, y_pred = y_pred_lr)
display.plot()

# importamos a lista de métricas
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

# vamos agora aplicar as métricas
# MAE
mean_absolute_error(y_test, y_pred_lr)

# MSE
mean_squared_error(y_test, y_pred_lr)

# RMSE
np.sqrt(mean_squared_error(y_test, y_pred_lr))

# MAPE
mean_absolute_percentage_error(y_test, y_pred_lr)

# R2
r2_score(y_test, y_pred_lr)