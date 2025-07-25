

arquivo_csv = open('C:/Users/yfreitas/Documents/pythonSamples/ficheiro.csv', 'r')
# Crie um leitor CSV
leitor_csv = arquivo_csv.readlines()
print(leitor_csv)
# Itere sobre as linhas do arquivo CSV
for linha in leitor_csv:
    # Cada linha é uma lista de valores separados por vírgula
    print(linha)

arquivo2 = open('C:/Users/yfreitas/Documents/pythonSamples/ficheiro.csv', 'a+')

arquivo2.write('Juca,33,M\n')

print(leitor_csv)