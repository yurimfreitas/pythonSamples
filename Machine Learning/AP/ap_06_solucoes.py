# -*- coding: utf-8 -*-
"""AP_06_solucoes.ipynb

# Vamos aplicar o que aprendemos sobre métricas
*   Complete com o código em falta
*   Sempre que necessário, **substitua** ___

---

dataset: [fonte](https://www.kaggle.com/datasets/charleyhuang1022/contract-renewal-prediction?select=South_China.csv)

---

Corra a primeira célula de código para obter o dataframe com que vamos trabalhar
"""

# faça o upload do ficheiro csv associado à atividade

# vamos importar a biblioteca
import pandas as pd

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

# defina a variável alvo como sendo a coluna "Renewal"
target_variable = "Renewal"

# train_test split usando a função train_test_split
X = df_contract_renewal.drop(["ID", target_variable], axis = 1)
y = df_contract_renewal[target_variable]*1

# verifique o grau de desequilibrio
print(y.sum()/len(y))

# importe a função train_test_split e defina X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.4,
                                                    random_state = 12,
                                                    stratify = y)

# aplique o modelo de KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# verifique a accuracy do modelo
knn.score(X_test, y_test)

# aplique o modelo de regressão logística
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression( max_iter = 250)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# verifique a accuracy do modelo
log_reg.score(X_test, y_test)

labels = ['cancelou', 'renovou']

# faça import do classification_report e da confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# veja a matriz de confusão dos modelos
print(confusion_matrix(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_log_reg))

# veja as métricas através do report
print(classification_report(y_test, y_pred_knn, target_names = labels))
print(classification_report(y_test, y_pred_log_reg, target_names = labels))

# importe a curva roc
from sklearn.metrics import roc_curve

# Calcule as probabilidades previstas (.predict_proba)
y_pred_log_reg_prob = log_reg.predict_proba(X_test)[:,1]

# gere a curva ROC
FPR, TPR, thresholds = roc_curve(y_test, y_pred_log_reg_prob)

# visualize a curva ROC
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(FPR, TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC')
plt.show()

# visualize a influência do threshold
plt.xlim(0,1.1)
plt.ylim(0,1.1)
plt.plot(thresholds[1:], TPR[1:])
plt.plot(thresholds[1:], FPR[1:])
plt.plot([0.5, 0.5], [0, 1.1], 'k--')
plt.xlabel('threshold')
plt.ylabel('rates')
plt.legend(['TPR', 'FPR'])
plt.title('rates vs (TPR e FPR)')
plt.show()

# calcule a AUC
from sklearn.metrics import roc_auc_score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_log_reg_prob)))