numero = int(input("Digite um numero para descobrir seus divisores: "))

i = 1
lista_divisores=[]

while i <= numero:
   if numero % i == 0:
      lista_divisores.append(i)
   
   i+=1

print(lista_divisores)