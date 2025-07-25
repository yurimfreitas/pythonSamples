# -*- coding: utf-8 -*-
"""AP_10_solucoes.ipynb


# Vamos aplicar o que aprendemos sobre regressões lineares
*   Complete com o código em falta
*   Sempre que necessário, **substitua** ___

---

dataset: [fonte](https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction)

---

Corra a primeira célula de código para obter o dataframe com que vamos trabalhar
"""

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

# defina a variável alvo ("Y house price of unit area")
target_variable = "Y house price of unit area"

# train_test split usando a função train_test_split
X = df_real_estate.drop([target_variable], axis = 1)
y = df_real_estate[target_variable]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 12)

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# defina o cv com 10 folds
kf = KFold(n_splits=10)

# crie o modelo de random forest
rf = RandomForestRegressor(random_state=42)

# defina a grid dos hyperparameters considerando
# n_estimators -> [50, 100, 200, 400]
# max_depth -> [None, 10, 20, 30]
# min_samples_split -> [2, 5, 10]

param_grid_rf = {
    'n_estimators': [50, 100, 200, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# crie o dict de métricas (rmse, r2)
scoring = {
    'rmse': 'neg_root_mean_squared_error',
    'r2': 'r2'
}

# aplique a grid search ao modelo de Random Forest (faça fit para o rmse)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=kf, scoring=scoring, refit='rmse')
grid_search_rf.fit(X_train, y_train)

# obtenha os melhores hyperparameters
best_params_rf = grid_search_rf.best_params_
print("Best Parameters for Random Forest Regressor:", best_params_rf)

# obtenha o melhor score
best_score_rf = grid_search_rf.best_score_
print("Best Cross-Validation RMSE Score for Random Forest Regressor:", -best_score_rf)

# avalie os resultados para os diferentes k_folds
pd.DataFrame(grid_search_rf.cv_results_)

# obtenha o melhor modelo
best_rf = grid_search_rf.best_estimator_

# faça as previsões
y_pred_rf = best_rf.predict(X_test)

# avalie as métricas finais
print("\nRandom Forest Regressor Evaluation")
print("Root Mean Squared Error:", mean_squared_error(y_test, y_pred_rf)**(1/2))
print("R2 Score:", r2_score(y_test, y_pred_rf))