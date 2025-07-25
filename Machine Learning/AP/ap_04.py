# -*- coding: utf-8 -*-
"""AP_04.ipynb

# Vamos aplicar o que aprendemos sobre os pandas dataframes


*   Complete com o código em falta
*   Sempre que necessário, **substitua** ___

---

dataset: [fonte](https://www.kaggle.com/datasets/crawford/80-cereals)

---

Corra a primeira célula de código para obter o dataframe com que vamos trabalhar
"""

# faça o upload do ficheiro csv associado à atividade

# vamos importar a biblioteca
import pandas as pd

# vamos importar o dataframe do ficheiro .csv
df_cereals = pd.read_csv("cereal.csv")

# veja as 5 primeiras linhas do dataframe
df_cereals.___

# veja a forma do dataframe: quantas linhas, quantas colunas?
df_cereals.___

# veja a informação sobre o dataframe
df_cereals.___

# veja a descrição das variáveis numéricas
df_cereals.___

# encontre a média de sodio (sodium)
df_cereals["___"].___

# encontre a mediana de proteína e de gordura (protein, fat)
df_cereals[["___", "___"]].___

# encontre o mínimo e máximo do valor de fibras (sodium, fiber)
df_cereals___

# conte o número de cereais vendidos por marca
df_cereals___

# conte o número de cereais por tipo (type) em ordem ascendente e em freq. rel.
df_cereals[___].value_counts(___, ___, ___)

# agrupe os cerais por marca (mfr) e veja a média de calorias (calories)
df_cereals_cal_by_mfr = df_cereals.___

# use a função de sort_values no grupo anterior para ordenar em sentido desc.
df_cereals_cal_by_mfr.___

# agrupe pelo tipo (type) e marca (mfr) e veja o valor min e max de sodio (sodium)
df_cereals.___

# faça uma tabela pivô com a média de calorias (calories) por marca (mfr)
df_cereals.pivot_table(values = "___", index = "___")

# faça a mesma tabela pivô, mas com o máximo das gorduras (fat)
df_cereals.pivot_table___

# faça a mesma tabela pivô, mas com o max, a média e o min do açúcar (sugars)
df_cereals.pivot_table___

# faça a mesma tabela pivô mas por marca (mfr) e tipo (type)
df_cereals.pivot_table___

# faça uma tabela pivô do máximo de rating por marca (mfr) e tipo (type)
df_cereals.pivot_table___

# faça a mesma tabela pivô mas com os totais visíveis
df_cereals.pivot_table___

# faça a mesma tabela usando ainda o argumento fill_value = 0
df_cereals.pivot_table___