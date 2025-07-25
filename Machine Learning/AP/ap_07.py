# -*- coding: utf-8 -*-
"""AP_07.ipynb

# Vamos aplicar o que aprendemos sobre Cross-Validation

*   Complete com o código em falta
*   Sempre que necessário, **substitua** ___

---

dataset: [fonte](https://www.kaggle.com/datasets/charleyhuang1022/contract-renewal-prediction?select=South_China.csv)

---
"""

# faça o upload do ficheiro csv associado à atividade

# vamos importar a biblioteca
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

# defina a variável alvo
target_variable = ___

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

# importe o modelo de KNeighborsClassifier e o de LogisticRegression
from sklearn.neighbors import ___
from sklearn.linear_model import ___

# instancie os objetos
knn = KNeighborsClassifier(___ = 7)
log_reg = LogisticRegression( max_iter = 250)

# importe o KFold
from sklearn.model_selection import KFold

# importe as métricas de precision e recall
from sklearn.metrics import precision_score, recall_score

# defina 5 folds
kfold = KFold(n_splits=___)

# crie listas vazias para os scores
precision_scores_knn = ___
recall_scores_knn = ___
precision_scores_log_reg = ___
recall_scores_log_reg = ___

# faça cross-validation usando o ciclo for para o modelo de knn
for train_index, val_index in kfold.split(___):
    X_train_fold, X_val_fold = X_train.iloc[___], X_train.iloc[___]
    y_train_fold, y_val_fold = y_train.iloc[___], y_train.iloc[___]

    knn.___(___, ___)
    y_pred_knn = ___.___(___)
    precision_knn = ___(___, y_pred_knn)
    recall_knn = ___(___, y_pred_knn)

    precision_scores_knn.append(precision_knn)
    recall_scores_knn.append(recall_knn)

# faça print para visualizar os resultados
print("KNN - Mean Precision:", np.mean(___).round(2))
print("KNN - Mean Recall:", np.mean(___).round(2))

# importe o modelo
from sklearn.model_selection import cross_val_score

# utilize o cross_val_score para fazer cross-validation
precision_scores_knn = cross_val_score(knn, ___, ___, cv=___, scoring='precision')
recall_scores_knn = cross_val_score(knn, ___, ___, cv=___, scoring='recall')

# faça print dos resultados
print("KNN - Mean Precision:", np.mean(___).round(2))
print("KNN - Mean Recall:", np.mean(___).round(2))

# importamos o modelo
from sklearn.model_selection import cross_validate

# defina as suas métricas de score ('precision', 'recall', 'neg_log_loss')
scoring = ___

# utilize a função de cross-validate
cv_results_log_reg = cross_validate(log_reg, ___, ___, cv=___, scoring=___)

# faça print dos resultados
print("Logistic Regression - Mean Precision:", np.mean(___).round(2))
print("Logistic Regression - Mean Recall:", np.mean(___).round(2))
print("Logistic Regression - Mean Log Loss:", -np.mean(___).round(2))