# -*- coding: utf-8 -*-
"""Pipelines.ipynb

# Pipelines

---


[documentação](https://scikit-learn.org/stable/index.html) <br>


---

>[Pipelines](#scrollTo=QoBv84MIUa-h)

>>[1. Vamos começar por importar os packages e o dataset](#scrollTo=HO6jdFDmldaU)

>>[2. Vamos explorar o dataset](#scrollTo=2S0UrVDEf8E-)

>>[3. Vamos aplicar um processamento genérico](#scrollTo=BKzodGb8R7t9)

>>>[3.1.  Removemos as colunas com pouca informação](#scrollTo=rEuiN5R7Przw)

>>>[3.2.  Removemos as linhas de embarked sem valor](#scrollTo=c4ZBqVliAQ0V)

>>[4. Vamos fazer o train test split](#scrollTo=vevBCPmAm2K9)

>>[5. Vamos definir o nosso pipeline](#scrollTo=iIFgSiGFuxbl)

>>[6. Aplicamos o pipeline](#scrollTo=73M-ivqdxRkQ)

>>[7. Avaliamos os resultados](#scrollTo=Uczo3UrRxozn)

>>[8. E a gridSearch?](#scrollTo=NAlheW90yP5b)

## 1.&nbsp;Vamos começar por importar os packages e o dataset
"""

# packages gerais
import pandas as pd
import numpy as np

# vamos importar o dataset do titanic
df = pd.read_csv("titanic.csv")

"""## 2.&nbsp;Vamos explorar o dataset"""

# vamos avaliar os dados em falta
df.isna().sum()
# df[df.isna().sum(axis = 1) > 1]

# exploração inicial
# df.head()
# df.info()
# df.shape
# df.describe()

"""## 3.&nbsp;Vamos aplicar um processamento genérico

### 3.1.&nbsp; Removemos as colunas com pouca informação
"""

# fazemos um simples drop
df.drop('Cabin', axis=1, inplace=True)

"""### 3.2.&nbsp; Removemos as linhas de embarked sem valor"""

# vamos fazer dropna para remover os 2 casos que faltam
df.dropna(subset=['Embarked'], inplace=True)

df.head()

"""## 4.&nbsp;Vamos fazer o train test split"""

# definimos a variável alvo
target_variable = "Survived"

# train_test split usando a função train_test_split
X = df.drop(["PassengerId", "Name", "Ticket", target_variable], axis = 1)
y = df[target_variable]
y.sum()/len(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 12,
                                                    stratify = y)

"""## 5.&nbsp;Vamos definir o nosso pipeline"""

# vamos importar os transformers
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# vamos definir o processamento das variáveis numéricas
numeric_features = ["Age", "Fare"]
age_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

fare_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('power_transformer', PowerTransformer())
])

# vamos definir o processamento das variáveis categóricas
categorical_features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# vamos combinar os processos
preprocessor = ColumnTransformer(transformers=[
    ('age', age_transformer, ["Age"]),
    ('fare', fare_transformer, ["Fare"]),
    ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# vamos crear o pipeline com processamento e modelação
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=250))
])

"""## 6.&nbsp;Aplicamos o pipeline"""

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# fazemos o fit do pipeline
pipeline.fit(X_train, y_train)

# prevemos os valores de teste
y_pred = pipeline.predict(X_test)

"""## 7.&nbsp;Avaliamos os resultados"""

# avaliamos o modelo
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['morreu', 'sobreviveu']))

# ROC curve and AUC
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
FPR, TPR, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(FPR, TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC')
plt.show()

plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
plt.plot(thresholds[1:], TPR[1:])
plt.plot(thresholds[1:], FPR[1:])
plt.plot([0.5, 0.5], [0, 1.1], 'k--')
plt.xlabel('threshold')
plt.ylabel('rates')
plt.legend(['TPR', 'FPR'])
plt.title('Curva ROC')
plt.show()

print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

"""## 8.&nbsp;E a gridSearch?"""

# importamos a grid search
from sklearn.model_selection import GridSearchCV

# temos de definir os dados para a grid
param_grid = {
    'classifier__C': [1, 10, 100, 1000],
    'classifier__penalty': [None, 'l2']
}

# creamos a grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

# fazemos o fit da grid search
grid_search.fit(X_train, y_train)

# vamos ver os melhores parameters
print("Best parameters found:")
print(grid_search.best_params_)

# obtemos o melhor modelo
best_model = grid_search.best_estimator_

# vamos prever usando o melhor modelo
y_pred = best_model.predict(X_test)

# avaliamos o modelo
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['morreu', 'sobreviveu']))

# ROC curve and AUC
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
FPR, TPR, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(FPR, TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC')
plt.show()

plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
plt.plot(thresholds[1:], TPR[1:])
plt.plot(thresholds[1:], FPR[1:])
plt.plot([0.5, 0.5], [0, 1.1], 'k--')
plt.xlabel('threshold')
plt.ylabel('rates')
plt.legend(['TPR', 'FPR'])
plt.title('Curva ROC')
plt.show()

print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))