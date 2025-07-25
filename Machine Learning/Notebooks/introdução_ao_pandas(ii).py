# -*- coding: utf-8 -*-
"""Introdução_ao_pandas(II).ipynb

#Introdução ao pandas

---


[documentação](https://pandas.pydata.org/docs/index.html) <br>
dataset: [fonte](https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales?select=supermarket_sales+-+Sheet1.csv)

---

>[Introdução ao pandas](#scrollTo=Z2Ur5SspUJlO)

>>[1. Vamos começar por importar a biblioteca e o dataset](#scrollTo=ZxiRWYNLgXq-)

>>[2. Vamos explorar o dataframe](#scrollTo=Ydxw6tPwf-TF)

>>[3. Vamos sumarizar os dados numéricos](#scrollTo=_o03LNFpg4li)

>>[4. Contar](#scrollTo=lUulh_Fl0T9B)

>>[5. Agrupar](#scrollTo=w1de_2_1pcdj)

>>[6. Tabelas Pivot](#scrollTo=cfZmo-fOrQ6R)

## 1.&nbsp;Vamos começar por importar a biblioteca e o dataset
"""

# importar a biblioteca
import pandas as pd

# importar o dataset que está em csv
df_sales = pd.read_csv('supermarket_sales.csv')
print(df_sales)

"""## 2.&nbsp;Vamos explorar o dataframe"""

# vamos ver as primeiras 5 linhas do dataframe
df_sales.head()

# vamos ver a "ficha" do dataframe
df_sales.info() #method

# vamos ver a forma do dataframe
df_sales.shape #atribute

# podemos sempre ver estatísticas gerais
df_sales.describe()

"""## 3.&nbsp;Vamos sumarizar os dados numéricos"""

# calcular a média do Total
df_sales["Total"].mean() # .min(); .max(); .std()

# calcular a soma do Total
df_sales["Total"].sum()

# calcular a transação mais recente
df_sales["Date"].max()

# para criar estatísticas personalizadas -> .agg()
# vamos definir uma função
def my_median(column):
  return column.median()

# vamos aplicar a nossa função a uma coluna
df_sales["Quantity"].agg(my_median)

# posso aplicar a mais do que uma coluna
df_sales[["Quantity", "Total"]].agg(my_median)

# posso aplicar várias funções à mesma coluna
def my_mean(column):
  return column.mean()
df_sales[["Quantity", "Total"]].agg([my_median, my_mean])

"""## 4.&nbsp;Contar"""

# vamos eliminar duplicados considerando apenas a coluna branch
df_sales_unique = df_sales.drop_duplicates(subset = ["Branch"])

# vamos contar quantas linhas temos por "Branch"
df_sales["Branch"].value_counts()

# vamos contar quantas linhas temos por invoice
df_sales["Invoice ID"].value_counts()

# vamos ordenar as contagens de linha de produtos
df_sales["Product line"].value_counts(sort = True) # descendente por default

# podemos ver este valor por frequencia relativa
df_sales["Product line"].value_counts(normalize = True)

"""## 5.&nbsp;Agrupar"""

# vamos ver a média de vendas por cidade
df_sales.groupby("City")["Total"].mean()

# vamos ver o máximo de quantidade vendida por linha de produto
df_sales.groupby("Product line")["Quantity"].sum()

# podemos ter mais do que uma função e variável
df_sales.groupby("Product line")[["Quantity", "Total"]].agg(["min", "max"])

# podemos ter mais do que um elemento em cada uma das opções
df_sales.groupby(["City", "Product line"])[["Quantity", "Total"]].agg(["min", "max"])

"""## 6.&nbsp;Tabelas Pivot"""

# vamos ver como fazer uma tabela pivô
df_sales.pivot_table(values = "Total", index = "City")

# agora queremos a soma e não a média
df_sales.pivot_table(values = "Total", index = "City", aggfunc = "sum")

# podemos usar funções do numpy
import numpy as np
df_sales.pivot_table(values = "Total", index = "City", aggfunc = np.sum)

# podemos ter mais do que uma função
df_sales.pivot_table(values = "Total", index = "City", aggfunc = [np.sum, np.mean])

# vamos agora ver a informação, mas por cidade e linha de produto
df_sales.pivot_table(values = "Total",
                     index = "City",
                     columns = "Product line",
                     aggfunc = [np.sum])

# vamos agora adicionar os totais
df_sales.pivot_table(values = "Total",
                     index = "City",
                     columns = "Product line",
                     aggfunc = [np.sum],
                     margins = True)