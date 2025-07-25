# -*- coding: utf-8 -*-
"""Introdução_ao_pandas(I).ipynb

#Introdução ao pandas

---


[documentação](https://pandas.pydata.org/docs/index.html) <br>
dataset: [fonte](https://www.kaggle.com/datasets/crawford/80-cereals)

---

>[Introdução ao pandas](#scrollTo=Z2Ur5SspUJlO)

>>[1. Vamos começar por importar a biblioteca e o dataset](#scrollTo=ZxiRWYNLgXq-)

>>[2. Vamos explorar o dataframe](#scrollTo=Ydxw6tPwf-TF)

>>[3. Vamos ordenar e criar subconjuntos](#scrollTo=_o03LNFpg4li)

>>[4. Criar novas colunas](#scrollTo=lUulh_Fl0T9B)

## 1.&nbsp;Vamos começar por importar a biblioteca e o dataset
"""

# importar a biblioteca
import pandas as pd

# importar o dataset que está em csv
df_cereals = pd.read_csv('cereal.csv')
print(df_cereals)

"""## 2.&nbsp;Vamos explorar o dataframe"""

# vamos ver as primeiras 5 linhas do dataframe
df_cereals.head()
# vamos ver as primeiras 10 linhas do dataframe
df_cereals.head(10)

# vamos ver as últimas 5 linhas do dataframe
df_cereals.tail()

# vamos ver a "ficha" do dataframe
df_cereals.info() #method

# vamos ver a forma do dataframe
df_cereals.shape #atribute

# podemos sempre ver estatísticas gerais
df_cereals.describe()

# um dataframe é composto por (index, columns e values)
df_cereals.index
df_cereals.columns
df_cereals.values

"""## 3.&nbsp;Vamos ordenar e criar subconjuntos"""

# para ordenar o dataframe usamos a função sort_values()
# vamos ordenar pela coluna de rating
df_cereals.sort_values("rating")

# vamos agora ordenar em sentido descencente
df_cereals.sort_values("rating", ascending = False)

# agora vamos querer ordenar pelos ratings e pelo açucar
df_cereals.sort_values(["rating", "sugars"])

# o que queremos é ter o rating e sentido descendente e o açucar no ascendente
df_cereals.sort_values(["rating", "sugars"], ascending = [False, True])

# vamos ver como extrair uma das colunas
df_cereals["rating"]
df_cereals["rating"].columns
# df_cereals[["rating"]]
# df_cereals[["rating"]].columns

# agora queremos obter o subconjunto com as colunas do açucar e do rating
df_cereals[["sugars", "rating"]]

# para obter o subconjunto de linhas, podemos usar uma condição
rating_condition_series = df_cereals["rating"] > 60
df_cereals[rating_condition_series]

# rating_condition_df = df_cereals[["rating"]] > 60
# df_cereals[rating_condition_df]

# podemos usar mais do que uma condição
df_cereals[(df_cereals["rating"] > 60) & (df_cereals["mfr"] == "N")]

# ou
condition_1 = df_cereals["rating"] > 60
condition_2 = df_cereals["mfr"] == "N"
df_cereals[condition_1 & condition_2]

"""## 4.&nbsp;Criar novas colunas"""

# a coluna peso (weight) está em ounces; vamos alterá-la para gramas
# 1 ounce = 28.3495231 g
df_cereals["weight_g"] = df_cereals["weight"] * 28.3495231
df_cereals.sort_values("weight", ascending = False)

# vamos agora criar o rácio do peso/chávena
df_cereals["ratio"] = df_cereals["weight_g"] / df_cereals["cups"]
df_cereals.sort_values("ratio", ascending = False)

# podemos agora começar a juntar o que aprendemos e a explorar o df
# vamos ver os cereais que:
# têm um ratio < 50
# são da marca K
# por ordem descendente em termos de açucar
# só queremos as colunas do nome e do açucar

final_condition_1 = df_cereals["ratio"] < 50
final_condition_2 = df_cereals["mfr"] == "K"
df_cereals_subset = df_cereals[final_condition_1 & final_condition_2]
df_cereals_subset_sort = df_cereals_subset.sort_values("sugars",
                                                       ascending = False)
df_cereals_subset_sort

df_cereals_final = df_cereals_subset_sort[["name", "sugars"]]
df_cereals_final