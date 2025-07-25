numeros = [2,5,7,9,10,13]
soma = 0

for num in numeros:
    if num%3==0:
        soma -= num
    elif num%2==0:
        soma += num

#if soma < 3:
 #   while soma < 5:
  #      soma += 1
#else:
 #   for num in range(2):
  #     soma -= 1

print(soma)