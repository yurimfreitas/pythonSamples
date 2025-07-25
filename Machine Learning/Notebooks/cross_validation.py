# -*- coding: utf-8 -*-
"""cross-validation.ipynb

# Cross-Validation

---


[documentação](https://scikit-learn.org/stable/index.html) <br>
dataset: [fonte](https://www.kaggle.com/datasets/dinhanhx/studentgradepassorfailprediction)

---

>[Cross-Validation](#scrollTo=QoBv84MIUa-h)

>>[1. Vamos começar por importar os packages e o dataset](#scrollTo=HO6jdFDmldaU)

>>[3. Vamos dividir em treino e teste com a ajuda do scikit-learn](#scrollTo=VlyljBsUgD8D)

>>[4. Vamos aplicar o modelo KNN](#scrollTo=rBbqqMS5hXRe)

>>[5. Vamos aplicar o modelo de regressão logística](#scrollTo=Z-_Mfy5StNhM)

>>[6. Vamos aplicar o Cross-Validation](#scrollTo=VvyvAnaJqBo9)

>>>[6.1 K-Fold](#scrollTo=J-PpbmBgs3Rt)

>>>[6.2 Stratified K-Fold](#scrollTo=rNRDHTcGvuD1)

>>>[6.3 Vamos simplificar usando funções do scikit-learn](#scrollTo=t2nxybAP0SSP)

## 1.&nbsp;Vamos começar por importar os packages e o dataset
"""

# packages gerais
import pandas as pd
import numpy as np

# dataset
df_students = pd.read_csv("student-mat-pass-or-fail.csv")

"""## 3.&nbsp;Vamos dividir em treino e teste com a ajuda do scikit-learn"""

# definimos a variável alvo
target_variable = "pass"

# train_test split usando a função train_test_split
X = df_students.drop(["G3", target_variable], axis = 1)
y = df_students[target_variable]
y.sum()/len(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 12,
                                                    stratify = y)

"""## 4.&nbsp;Vamos aplicar o modelo KNN"""

# importamos o modelo
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# para verificar a accuracy do modelo aplicamos a função .score
knn.score(X_test, y_test)

"""## 5.&nbsp;Vamos aplicar o modelo de regressão logística"""

# importamos o modelo
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression( max_iter = 250)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# para verificar a accuracy do modelo aplicamos a função .score
log_reg.score(X_test, y_test)

"""## 6.&nbsp;Vamos aplicar o Cross-Validation"""

# importamos os modelos de CV e as métricas
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
    RepeatedKFold
)
from sklearn.metrics import precision_score, recall_score

"""### 6.1&nbsp;K-Fold"""

#importamos o modelo
kfold = KFold(n_splits=5)

precision_scores_knn = []
recall_scores_knn = []
precision_scores_log_reg = []
recall_scores_log_reg = []

for train_index, val_index in kfold.split(X_train):

    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    knn.fit(X_train_fold, y_train_fold)
    y_pred_knn = knn.predict(X_val_fold)
    precision_knn = precision_score(y_val_fold, y_pred_knn)
    recall_knn = recall_score(y_val_fold, y_pred_knn)
    precision_scores_knn.append(precision_knn)
    recall_scores_knn.append(recall_knn)

    log_reg.fit(X_train_fold, y_train_fold)
    y_pred_log_reg = log_reg.predict(X_val_fold)
    precision_log_reg = precision_score(y_val_fold, y_pred_log_reg)
    recall_log_reg = recall_score(y_val_fold, y_pred_log_reg)
    precision_scores_log_reg.append(precision_log_reg)
    recall_scores_log_reg.append(recall_log_reg)

# vemos os resultados
print("Precision scores for KNN:", np.mean(precision_scores_knn).round(2))
print("Precision scores for Logistic Regression:", np.mean(precision_scores_log_reg).round(2))
print("Recall scores for KNN:", np.mean(recall_scores_knn).round(2))
print("Recall scores for Logistic Regression:", np.mean(recall_scores_log_reg).round(2))

"""### 6.2&nbsp;Stratified K-Fold"""

#importamos o modelo
stratified_kfold = StratifiedKFold(n_splits=5)

precision_scores_knn = []
recall_scores_knn = []
precision_scores_log_reg = []
recall_scores_log_reg = []

for train_index, val_index in stratified_kfold.split(X_train, y_train):

    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    knn.fit(X_train_fold, y_train_fold)
    y_pred_knn = knn.predict(X_val_fold)
    precision_knn = precision_score(y_val_fold, y_pred_knn)
    recall_knn = recall_score(y_val_fold, y_pred_knn)
    precision_scores_knn.append(precision_knn)
    recall_scores_knn.append(recall_knn)

    log_reg.fit(X_train_fold, y_train_fold)
    y_pred_log_reg = log_reg.predict(X_val_fold)
    precision_log_reg = precision_score(y_val_fold, y_pred_log_reg)
    recall_log_reg = recall_score(y_val_fold, y_pred_log_reg)
    precision_scores_log_reg.append(precision_log_reg)
    recall_scores_log_reg.append(recall_log_reg)

# vemos os resultados
print("Precision scores for KNN:", np.mean(precision_scores_knn).round(2))
print("Precision scores for Logistic Regression:", np.mean(precision_scores_log_reg).round(2))
print("Recall scores for KNN:", np.mean(recall_scores_knn).round(2))
print("Recall scores for Logistic Regression:", np.mean(recall_scores_log_reg).round(2))

"""### 6.3&nbsp;Vamos simplificar usando funções do scikit-learn"""

from sklearn.metrics import get_scorer_names
get_scorer_names()

# importamos o modelo
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
precision_scores_knn = cross_val_score(knn, X_train, y_train, cv=kfold, scoring='precision')
recall_scores_knn = cross_val_score(knn, X_train, y_train, cv=kfold, scoring='recall')
precision_scores_log_reg = cross_val_score(log_reg, X_train, y_train, cv=kfold, scoring='precision')
recall_scores_log_reg = cross_val_score(log_reg, X_train, y_train, cv=kfold, scoring='recall')

# vemos os resultados
print("Precision scores for KNN:", np.mean(precision_scores_knn).round(2))
print("Recall scores for KNN:", np.mean(recall_scores_knn).round(2))
print("Precision scores for Logistic Regression:", np.mean(precision_scores_log_reg).round(2))
print("Recall scores for Logistic Regression:", np.mean(recall_scores_log_reg).round(2))

# importamos o modelo
from sklearn.model_selection import cross_validate

scoring = ['precision', 'recall', 'neg_log_loss']
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results_knn = cross_validate(knn, X_train, y_train, cv=kfold, scoring=scoring)

# vemos os resultados
print("Precision scores for KNN:", np.mean(cv_results_knn['test_precision']).round(2))
print("Recall scores for KNN:", np.mean(cv_results_knn['test_recall']).round(2))

scoring = ['precision', 'recall', 'neg_log_loss']
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results_log_reg = cross_validate(log_reg, X_train, y_train, cv=kfold, scoring=scoring)

print("Precision scores for Logistic Regression:", np.mean(cv_results_log_reg['test_precision']).round(2))
print("Recall scores for Logistic Regression:", np.mean(cv_results_log_reg['test_recall']).round(2))
print("Log loss for Logistic Regression:", - np.mean(cv_results_log_reg['test_neg_log_loss']).round(2))