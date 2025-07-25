# -*- coding: utf-8 -*-
"""AP_13.ipynb

# Vamos aplicar o que aprendemos sobre processamento dos dados
*   Complete com o código em falta
*   Sempre que necessário, **substitua** ___

---


[documentação](https://scikit-learn.org/stable/index.html) <br>
dataset: [fonte](https://www.kaggle.com/datasets/mirichoi0218/insurance)

---

>[Vamos aplicar o que aprendemos sobre processamento dos dados](#scrollTo=7UqvAzOuK_SN)

>[1. Corra a primeira célula de código para obter o dataframe com que vamos trabalhar](#scrollTo=25DtwghMIQqJ)

>[2. Trate os dados em falta](#scrollTo=glt7JqnNdHxk)

>[3. Trate as variáveis categóricas](#scrollTo=Rq7QOxUC9Y4u)

>[4. Escale as variáveis](#scrollTo=_ZSGbtOHGQ8_)

#1.&nbsp;Corra a primeira célula de código para obter o dataframe com que vamos trabalhar
"""

# faça o upload do ficheiro csv associado à atividade

# vamos importar a biblioteca
import pandas as pd
import numpy as np

# vamos importar o dataframe do ficheiro .csv
df_insurance = pd.read_csv("insurance.csv")

# veja as 5 primeiras linhas do dataframe
df_insurance.head()
# veja a forma do dataframe: quantas linhas, quantas colunas?
df_insurance.shape
# veja a informação sobre o dataframe
df_insurance.info()
# veja a descrição das variáveis numéricas
df_insurance.describe()

"""#2.&nbsp;Trate os dados em falta"""

# remova a coluna quase sem informação
df_insurance_cleaned = ___

# avalie a distribuição dos dados da coluna numérica
df_insurance_cleaned.age.___

# importe o modelo de impute
from sklearn.___ import ___

# preencha os dados em falta
si_age = ___(strategy = 'median')
df_insurance_cleaned['age'] = si_age.___(___)

# remova os restantes casos
df_insurance_cleaned.___

"""#3.&nbsp;Trate as variáveis categóricas"""

# importe o package
from sklearn.___ import ___

# defina a lista de colunas a passar ao modelo
ohe_list = ___

# faça One-Hot Encoding
ohe = ___(sparse_output=False, drop='first')
ohe.___(___) # entrega em array
ohe.___()

df_ohe = pd.DataFrame(
    data=___,
    columns=___
)

df_insurance_cleaned = pd.concat([df_insurance_cleaned.reset_index(drop = True), df_ohe.reset_index(drop = True)], axis = 1)
df_insurance_cleaned.drop(___)

"""# 4.&nbsp;Escale as variáveis"""

# importe o package
from ___ import ___

# utilize o StandardScaler no imc
std_scaler = ___
df_insurance_cleaned['bmi'] = std_scaler.___

# utilize a transformação de yeo-johnson na idade
pow_scaler = ___
df_insurance_cleaned['age'] = pow_scaler.___