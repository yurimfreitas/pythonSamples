import numpy as np
import scipy.stats as st

filhos=[1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 7, 8, 10]

print(filhos)

desvio_padrao = np.std(filhos)

print("Desvio padrao diferente do R", desvio_padrao)

print("Intervalo de Confianca para precisao de 90%: ", st.t.interval(0.90, df=19, loc=np.mean(filhos), scale=st.sem(filhos)))

print("Intervalo de Confianca para precisao de 99%: ", st.t.interval(0.99, df=19, loc=np.mean(filhos), scale=st.sem(filhos)))