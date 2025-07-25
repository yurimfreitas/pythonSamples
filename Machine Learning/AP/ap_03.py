# -*- coding: utf-8 -*-
"""AP_03.ipynb

# Vamos aplicar o que aprendemos sobre os pandas dataframes


*   Complete com o código em falta
*   Sempre que necessário, **substitua** ___

---

dataset: [fonte](https://www.kaggle.com/datasets/starbucks/starbucks-menu?select=starbucks_drinkMenu_expanded.csv)

---

Corra a primeira célula de código para obter o dataframe com que vamos trabalhar
"""

# faça o upload do ficheiro csv associado à atividade

# vamos importar a biblioteca
import pandas as pd

# vamos importar o dataframe do ficheiro .csv
df_starbucks = pd.read_csv("starbucks_drinkMenu_expanded.csv")

# veja as 5 primeiras linhas do dataframe
df_starbucks.___

# veja as 10 primeiras linhas do dataframe
df_starbucks.___

# veja as últimas 10 linhas do dataframe
df_starbucks.___

# veja a forma do dataframe: quantas linhas, quantas colunas?
df_starbucks.___

# veja a informação sobre o dataframe
df_starbucks.___

# veja a descrição das variáveis numéricas
df_starbucks.___

# veja as três partes que constituem o dataframe
# não se esqueça de que são atributos
# não se esqueça de que um deles é "estranho"
df_starbucks.___
df_starbucks.___
df_starbucks.___

# ordene as bebidas pela suas calorias (Calories) -> ascendente
df_starbucks.___("___")

# ordene as bebidas pela suas calorias (Calories) -> descendente
df_starbucks.___("___", ___)

# ordene as bebidas pelas calorias-> desc e pelas proteínas ( Protein (g))-> asc
df_starbucks.___

# crie uma serie com a coluna das calorias (Calories)
calories_series = ___

# crie um dataframe com a mesma coluna
calories_df = ___

# crie um df com a categoria das bebidas (Beverage_category) e as calorias
df_beverage_cg_and_calories = ___

# ordene este novo df pelas categorias-> asc e pelas calorias-> desc
df_beverage_cg_and_calories.___

# filtre as bebidas com calorias acima de 400
condition = ___
df_starbucks[condition]

# filtre as bebidas com calorias acima de 400
# mas que não sejam "Signature Espresso Drinks"
condition_1 = ___
condition_2 = ___
df_starbucks[condition_1 & condition_2]

# queremos saber -> açucar/carbs*100
df_starbucks["Sugar_Carbs_ratio"] = ___

# queremos apenas ficar com as colunas:
#(Beverage, Calories, Sugar_Carbs_ratio, Caffeine (mg))
list_of_columns = ["Beverage", "Calories", "Sugar_Carbs_ratio", "Caffeine (mg)"]
df_starbucks_subset = ___

# ordene pelas calorias -> descendente
df_starbucks_subset_sort = ___