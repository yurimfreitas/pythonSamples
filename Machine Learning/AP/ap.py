# -*- coding: utf-8 -*-
"""AP.ipynb

# Previsão do valor de mercado das casas na Califórnia

Explore o dataset e construa um modelo de regressão capaz de prever o valor de mercado do imobiliário californiano.

* Divida em treino e teste, com 30% dos dados para teste;
* sempre que necessário, use random_state = 12;
"""

from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
# Load the data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
data