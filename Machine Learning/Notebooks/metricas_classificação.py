# -*- coding: utf-8 -*-
"""metricas_classificação.ipynb

# Métricas de modelos de classificação

---


[documentação](https://scikit-learn.org/stable/index.html) <br>
dataset: [fonte](https://www.kaggle.com/datasets/dinhanhx/studentgradepassorfailprediction)

---

>[Métricas de modelos de classificação](#scrollTo=QoBv84MIUa-h)

>>[1. Vamos começar por importar os packages e o dataset](#scrollTo=HO6jdFDmldaU)

>>[2. Vamos explorar o dataset](#scrollTo=2S0UrVDEf8E-)

>>[3. Vamos dividir em treino e teste com a ajuda do scikit-learn](#scrollTo=VlyljBsUgD8D)

>>[4. Vamos aplicar o modelo KNN](#scrollTo=rBbqqMS5hXRe)

>>[5. Vamos aplicar o modelo de regressão logística](#scrollTo=Z-_Mfy5StNhM)

>>[6. Vamos obter as métricas de classificação](#scrollTo=q1AhrqhK6Aea)

>>>[6.1. matriz de confusão](#scrollTo=bt4CbYHc7h3U)

>>>[6.2. curva ROC e métrica AUC](#scrollTo=kRans4wO7fTR)

## 1.&nbsp;Vamos começar por importar os packages e o dataset
"""

# packages gerais
import pandas as pd
import numpy as np

# dataset
df_students = pd.read_csv("student-mat-pass-or-fail.csv")

"""## 2.&nbsp;Vamos explorar o dataset"""

# exploração inicial
df_students.head()
# df_students.info()
# df_students.shape
# df_students.describe()

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

"""## 6.&nbsp;Vamos obter as métricas de classificação

### 6.1.&nbsp;matriz de confusão

Para relembrar: <br>
[TP FN <br>
 FP TN]

Accuracy -> (𝑇𝑃+𝑇𝑁)/(𝑃+𝑁)<br>
Precision -> 𝑇𝑃/(𝑇𝑃+𝐹𝑃)<br>
Recall, Hit rate ou TPR -> 𝑇𝑃/(𝑇𝑃+𝐹𝑁)<br>
Fall-out ou FPR -> 𝐹𝑃/(𝐹𝑃+𝑇𝑁)<br>
"""

# importamos os módulos que precisamos
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# vamos ver as métricas para o KNN
rótulos = ['chumbou', 'passou']
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn, target_names = rótulos))
# vamos ver as métricas para a regressão logística
print(confusion_matrix(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg, target_names = rótulos))

# vamos analisar melhor os resultados
n_positives = y_test.sum()
n_negatives = len(y_test) - y_test.sum()
print(n_positives, n_negatives)

"""### 6.2.&nbsp;curva ROC e métrica AUC"""

# importamos a curva roc
from sklearn.metrics import roc_curve

# Calculamos as probabilidades previstas (.predict_proba)
log_reg.predict_proba(X_test)
y_pred_log_reg_prob = log_reg.predict_proba(X_test)[:,1] # escolhemos a segunda coluna

# Vamos gerar a curva ROC
FPR, TPR, thresholds = roc_curve(y_test, y_pred_log_reg_prob)

# Vamos visualizar a curva
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(FPR, TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC')
plt.show()

# vamos visualisar a influência do threshold
plt.xlim(0,1.1)
plt.ylim(0,1.1)
plt.plot(thresholds[1:], TPR[1:])
plt.plot(thresholds[1:], FPR[1:])
plt.plot([0.5, 0.5], [0, 1.1], 'k--')
plt.xlabel('threshold')
plt.ylabel('rates')
plt.legend(['TPR', 'FPR'])
plt.title('Curva ROC')
plt.show()

# vamos calcular a AUC
from sklearn.metrics import roc_auc_score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_log_reg_prob)))