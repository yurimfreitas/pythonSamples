import pandas as pd

test = pd.read_csv('C:/Users/yfreitas/Documents/pythonSamples/ficheiro.csv')

print(test.to_string)

print(test.loc[[0,1]])