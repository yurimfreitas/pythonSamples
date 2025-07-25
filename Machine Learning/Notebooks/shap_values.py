# -*- coding: utf-8 -*-
"""SHAP_Values.ipynb

# SHAP Values

---


[documentação](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) <br>


---

>[SHAP Values](#scrollTo=QoBv84MIUa-h)

>>[1. Vamos começar por importar os packages e o dataset](#scrollTo=HO6jdFDmldaU)

>>[2. Vamos explorar o dataset](#scrollTo=2S0UrVDEf8E-)

>>[3. Vamos aplicar um processamento genérico](#scrollTo=BKzodGb8R7t9)

>>[4. Vamos fazer o train test split](#scrollTo=vevBCPmAm2K9)

>>[5. Vamos definir o nosso pipeline](#scrollTo=iIFgSiGFuxbl)

>>[6. Vamos avaliar as features com recurso aos SHAP Values](#scrollTo=D3MqReI-tYOW)

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

"""## 3.&nbsp;Vamos aplicar um processamento genérico"""

# fazemos um simples drop
df.drop('Cabin', axis=1, inplace=True)

# vamos fazer dropna para remover os 2 casos que faltam
df.dropna(subset=['Embarked'], inplace=True)

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
from sklearn.model_selection import GridSearchCV

# vamos definir o processamento das variáveis numéricas
numeric_features = ["Age", "Fare"]
age_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler().set_output(transform="pandas"))
])

fare_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median').set_output(transform="pandas")),
    ('power_transformer', PowerTransformer().set_output(transform="pandas"))
])

# vamos definir o processamento das variáveis categóricas
categorical_features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore').set_output(transform="pandas"))
])

# vamos combinar os processos
preprocessor = ColumnTransformer(transformers=[
    ('age', age_transformer, ["Age"]),
    ('fare', fare_transformer, ["Fare"]),
    ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
).set_output(transform="pandas")

# vamos crear o pipeline com processamento e modelação
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=250))
])

# temos de definir os dados para a grid
param_grid = {
    'classifier__C': [0.1, 1, 10, 100, 1000],
    'classifier__penalty': ['l2']
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

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

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

"""## 6.&nbsp;Vamos avaliar as features com recurso aos SHAP Values"""

# temos de instalar o shap no colab
# ! pip install shap

# vamos importar o package
import shap

# Vamos extrair o classificador
classifier = best_model.named_steps['classifier']

# Vamos transformar as eatures de teste
preprocessed_X_test = best_model.named_steps['preprocessor'].transform(X_test)

# criamos o explainer
explainer = shap.LinearExplainer(classifier, preprocessed_X_test)

# calculamos os SHAP values
shap_values = explainer.shap_values(preprocessed_X_test)

# vamos visualizar os SHAP values
shap.summary_plot(shap_values, preprocessed_X_test)

# vamos tratar os nomes das variáveis
preprocessed_X_test.columns = [name.split('__')[-1] for name in preprocessed_X_test.columns]

# vamos visualizar os SHAP values outa vez
shap.summary_plot(shap_values, preprocessed_X_test)

# vamos ver a variação dos SHAP values para a feature de Fare
shap.dependence_plot("Fare", shap_values, preprocessed_X_test,interaction_index="Age")

# vamos inverter o processo de escala
preprocessed_X_test[["Age"]] = best_model.named_steps['preprocessor'].named_transformers_['age'].named_steps['scaler'].inverse_transform(preprocessed_X_test[["Age"]])
preprocessed_X_test[["Fare"]] = best_model.named_steps['preprocessor'].named_transformers_['fare'].named_steps['power_transformer'].inverse_transform(preprocessed_X_test[["Fare"]])

# vamos ver a variação dos SHAP values para a feature de Fare
shap.dependence_plot("Fare", shap_values, preprocessed_X_test,interaction_index="Age")

# vamos ver a variação dos SHAP values para a feature da Age
shap.dependence_plot("Age", shap_values, preprocessed_X_test,interaction_index="Sex_male")

# vamos ver que peso tiveram cada uma das features no resultado de cada elemento do test
shap.plots.force(explainer.expected_value, shap_values[0,:], preprocessed_X_test.iloc[0, :], matplotlib = True)