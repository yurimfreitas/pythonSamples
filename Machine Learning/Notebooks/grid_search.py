# -*- coding: utf-8 -*-
"""Grid Search.ipynb


# Grid-Search

---


[documentação](https://scikit-learn.org/stable/index.html) <br>
dataset: [fonte](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction?select=CarPrice_Assignment.csv)

---

>[Grid-Search](#scrollTo=QoBv84MIUa-h)

>>[1.  vamos importar o dataset](#scrollTo=_78JL1jFVQST)

>>[2.  vamos fazer o train test split](#scrollTo=bP8vEZsNl-gh)

>>[3.  vamos aplicar o GridSearchCV](#scrollTo=Z_3KDut4mKXy)

## 1.&nbsp; classificação

### 1.1.&nbsp; vamos importar o dataset
"""

# vamos importar as bibliotecas
import pandas as pd
import numpy as np
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

"""### 1.2.&nbsp; vamos fazer o train test split"""

# defina a variável alvo
target_variable = 'Renewal'

# train_test split usando a função train_test_split
X = df_contract_renewal.drop(["ID", target_variable], axis = 1)
y = df_contract_renewal[target_variable]*1

# importe a função train_test_split e defina X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 12,
                                                    stratify = y)

"""### 1.3.&nbsp; vamos aplicar o GridSearchCV"""

# Vamos importar as bibliotecas
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    make_scorer
)

# Vamos definir o cv
skf = StratifiedKFold(n_splits=5)

# Vamos criar o modelo de KNN
knn = KNeighborsClassifier()

# Vamos definir a grid dos hyperparameter
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Vamos definir as métricas de scoring
scoring = {
    'accuracy': 'accuracy',
    'roc_auc': 'roc_auc'
}

# Vamos aplicar a grid search ao modelo de KNN
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=skf, scoring=scoring, refit='roc_auc')
grid_search_knn.fit(X_train, y_train)

# Vamos ver os melhores hyperparameters
best_params_knn = grid_search_knn.best_params_
best_score_knn = grid_search_knn.best_score_
print("Best Parameters for KNN:", best_params_knn)
print("Best Cross-Validation AUC for KNN:", best_score_knn)

# Vamos ver os resultados para cada combinação
pd.DataFrame(grid_search_knn.cv_results_)

# Vamos definir o modelo de Random Forest
rf = RandomForestClassifier(random_state=42)

# Vamos definir a grid para os hyperparameters
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Vamos aplicar a grid search ao modelo de Random Forest
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=skf, scoring=scoring, refit='roc_auc')
grid_search_rf.fit(X_train, y_train)

# Vamos ver os melhores hyperparameters
best_params_rf = grid_search_rf.best_params_
best_score_rf = grid_search_rf.best_score_
print("Best Parameters for Random Forest:", best_params_rf)
print("Best Cross-Validation AUC for Random Forest:", best_score_rf)

# Vamos ver os resultados para cada combinação
pd.DataFrame(grid_search_rf.cv_results_)

# Vamos fazer o fit do melhor modelo de KNN
best_knn = grid_search_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test)
y_pred_knn_proba = best_knn.predict_proba(X_test)[:, 1]

# Vamos avaliar as métricas finais
print("\nKNN Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_knn_proba))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

# Vamos fazer o fit do melhor modelo de Random Forest
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_pred_rf_proba = best_rf.predict_proba(X_test)[:, 1]

# Vamos avaliar as métricas finais
print("\nRandom Forest Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_rf_proba))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

"""## 2.&nbsp; regressão

### 2.1.&nbsp; vamos importar o dataset
"""

# packages gerais
import pandas as pd
import numpy as np

# dataset
df_car_price = pd.read_csv("CarPrice_Assignment.csv")

# exploração inicial
df_car_price.head()
# df_car_price.info()
# df_car_price.shape
# df_car_price.describe()

"""### 2.2.&nbsp; vamos fazer o train test split"""

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

"""### 2.3.&nbsp; vamos aplicar o GridSearchCV"""

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

# Vamos definir o cv
kf = KFold(n_splits=5)

# Vamos criar o modelo de random forest
rf = RandomForestRegressor(random_state=42)

# Vamos definir a grid para os hyperparameters
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Vamos definir as métricas de scoring
scoring = {
    'mse': 'neg_mean_squared_error',
    'r2': 'r2'
}

# Vamos aplicar a grid search ao modelo de Random Forest
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=kf, scoring=scoring, refit='r2')
grid_search_rf.fit(X_train, y_train)

# Vamos ver os melhores hyperparameters
best_params_rf = grid_search_rf.best_params_
best_score_rf = grid_search_rf.best_score_
print("Best Parameters for Random Forest Regressor:", best_params_rf)
print("Best Cross-Validation R2 Score for Random Forest Regressor:", best_score_rf)

# Vamos ver os resultados para cada combinação
pd.DataFrame(grid_search_rf.cv_results_)

# Vamos fazer o fit do melhor modelo de Random Forest
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Vamos avaliar as métricas finais
print("\nRandom Forest Regressor Evaluation")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))