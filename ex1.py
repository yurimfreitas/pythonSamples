numeros = [1,2,3,4,5]
soma = 0

for num in numeros:
    if num%2==0:
        soma += num
    elif num%3==0:
        continue
    else:
        soma -= num

while soma < 10:
    soma +=1

print (soma)