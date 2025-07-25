# -*- coding: utf-8 -*-
"""AP_13_solucoes.ipynb

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
df_insurance_cleaned = df_insurance.drop(columns='region', axis=1)

# avalie a distribuição dos dados da coluna numérica
df_insurance_cleaned.age.plot(kind='box')

# importe o modelo de impute
from sklearn.impute import SimpleImputer

# preencha os dados em falta
si_age = SimpleImputer(strategy = 'median')
df_insurance_cleaned['age'] = si_age.fit_transform(df_insurance_cleaned[['age']])

# remova os restantes casos
df_insurance_cleaned.dropna(inplace = True)

"""#3.&nbsp;Trate as variáveis categóricas"""

# importe o package
from sklearn.preprocessing import OneHotEncoder

# defina a lista de colunas a passar ao modelo
ohe_list = ['sex', 'children']

# faça One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False, drop='first')
ohe.fit_transform(df_insurance_cleaned[ohe_list]) # entrega em array
ohe.get_feature_names_out()

df_ohe = pd.DataFrame(
    data=ohe.transform(df_insurance_cleaned[ohe_list]),
    columns=ohe.get_feature_names_out()
)

df_insurance_cleaned = pd.concat([df_insurance_cleaned.reset_index(drop = True), df_ohe.reset_index(drop = True)], axis = 1)
df_insurance_cleaned.drop(ohe_list, inplace=True, axis=1)

"""# 4.&nbsp;Escale as variáveis"""

# importe o package
from sklearn.preprocessing import StandardScaler, PowerTransformer

# utilize o StandardScaler no imc
std_scaler = StandardScaler()
df_insurance_cleaned['bmi'] = std_scaler.fit_transform(df_insurance_cleaned[['bmi']])

# utilize a transformação de yeo-johnson na idade
pow_scaler = PowerTransformer()
df_insurance_cleaned['age'] = pow_scaler.fit_transform(df_insurance_cleaned[['age']])