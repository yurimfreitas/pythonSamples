# -*- coding: utf-8 -*-
"""AP_06.ipynb

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
target_variable = ___

# train_test split usando a função train_test_split
X = df_contract_renewal.drop(["ID", ___], axis = 1)
y = df_contract_renewal[___]*1

# importe a função train_test_split e defina X_train, X_test, y_train, y_test
from sklearn.model_selection import ___
___, ___, ___, ___ = ___(___)

# aplique o modelo de KNN
from sklearn.neighbors import ___
knn = ___(n_neighbors = 7)
knn.___(___, ___)
y_pred_knn = knn.___(___)

# verifique a accuracy do modelo
knn.___(___, ___)

# aplique o modelo de regressão logística
from sklearn.linear_model import ___
log_reg = ___()
log_reg.___(___, ___)
y_pred_log_reg = log_reg.___(___)

# verifique a accuracy do modelo
log_reg.___(___, ___)

labels = ['cancelou', 'renovou']

# faça import do classification_report e da confusion_matrix
from sklearn.metrics import ___
from sklearn.metrics import ___

# veja a matriz de confusão dos modelos
print(___(___, ___))
print(___(___, ___))

# veja as métricas através do report
print(___(___, ___, ___ = labels))
print(___(___, ___, ___ = labels))

# importe a curva roc
from sklearn.metrics import ___

# calcule as probabilidades previstas (.predict_proba)
y_pred_log_reg_prob = log_reg.___(___)[:,1]

# gere a curva ROC
FPR, TPR, thresholds = ___(___, ___)

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
from sklearn.metrics import ___
print("AUC: {}".format(___(___, ___)))