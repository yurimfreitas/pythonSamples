# -*- coding: utf-8 -*-
"""Processamento dos dados.ipynb

# Processamento dos dados

---


[documentação](https://scikit-learn.org/stable/index.html) <br>


---

>[Processamento dos dados](#scrollTo=QoBv84MIUa-h)

>>[1. Vamos começar por importar os packages e o dataset](#scrollTo=HO6jdFDmldaU)

>>[2. Vamos explorar o dataset](#scrollTo=2S0UrVDEf8E-)

>>[3. Vamos tratar os dados em falta](#scrollTo=BKzodGb8R7t9)

>>>[3.1.  Removemos as colunas com pouca informação](#scrollTo=rEuiN5R7Przw)

>>>[3.2.  Imputamos valores](#scrollTo=PEYghDvyQAdi)

>>>[3.3.  Removemos todas as linhas com elementos com valor em falta](#scrollTo=c4ZBqVliAQ0V)

>>[4. Vamos passar ao tratamento das variáveis categóricas](#scrollTo=Zr_lOzPEVUOS)

>>[5. Vamos por último escalar as variáveis](#scrollTo=7sdPNoqx9UYX)

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
df[df.isna().sum(axis = 1) > 1]

# exploração inicial
# df.head()
# df.info()
df.shape
# df.describe()

"""## 3.&nbsp;Vamos tratar os dados em falta

### 3.1.&nbsp; Removemos as colunas com pouca informação
"""

# fazemos um simples drop
df_cleaned = df.drop('Cabin', axis=1)

"""### 3.2.&nbsp; Imputamos valores"""

df.Age.plot(kind='box')

# vamos importar o modelo de impute
from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy = 'median')
df_cleaned['Age'] = si.fit_transform(df_cleaned[['Age']])

# vamos ver o dataframe final
df_cleaned.head(10)
df.head(10)

"""### 3.3.&nbsp; Removemos todas as linhas com elementos com valor em falta"""

# vamos fazer dropna para remover os 2 casos que faltam
df_cleaned.dropna(inplace = True)

df_cleaned.info()

"""## 4.&nbsp;Vamos passar ao tratamento das variáveis categóricas"""

# vamos primeiro examinar o conteúdo das variáveis
df_cleaned.select_dtypes(include='object')

# vamos remover o nome e o bilhete
df_cleaned.drop(['Name', 'Ticket'], inplace=True, axis=1)

df_cleaned.select_dtypes(include='object')

# vamos importar o package
from sklearn.preprocessing import OneHotEncoder

# vamos criar a lista de colunas a passar ao modelo
ohe_list = df_cleaned.select_dtypes(include='object').columns.to_list()

# One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False, drop='first')
ohe.fit_transform(df_cleaned[ohe_list]) # entrega em array
ohe.get_feature_names_out()

# vamos criar um df
df_ohe = pd.DataFrame(
    data=ohe.transform(df_cleaned[ohe_list]),
    columns=ohe.get_feature_names_out()
)

# vamos juntar tudo
df_cleaned = pd.concat([df_cleaned.reset_index(drop = True), df_ohe.reset_index(drop = True)], axis = 1)
df_cleaned.drop(ohe_list, inplace=True, axis=1)

df_cleaned

"""## 5.&nbsp;Vamos por último escalar as variáveis"""

df_cleaned[['Age', 'Fare']].hist()

# vamos importar o package
from sklearn.preprocessing import StandardScaler, PowerTransformer

# vamos utilizar o StandardScaler na idade
std_scaler = StandardScaler()
df_cleaned['Age'] = std_scaler.fit_transform(df_cleaned[['Age']]) # entrega em array

# vamos usar a transformação de yeo-johnson na tarifa
pow_scaler = PowerTransformer()
df_cleaned['Fare'] = pow_scaler.fit_transform(df_cleaned[['Fare']]) # entrega em array

df_cleaned[['Age', 'Fare']].hist()