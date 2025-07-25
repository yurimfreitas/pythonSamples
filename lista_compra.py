lista_compra = []
fim = ''
while fim != 'N':

    lista_compra.append(input("Digite o produto e quantidade separado por virgula: "))

    fim = input("Deseja adicionar mais produtos (S/N) ?")


for elemento in lista_compra:
   print('Produto: ' + elemento.split(',')[0]+' quantidade: '+ elemento.split(',')[1])