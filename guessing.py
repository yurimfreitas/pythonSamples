import random

rand=random.randint(1, 10)
guessing=0
errado=0

while guessing != rand:
   print("Batota:", rand)
   guessing = int(input("Digite um numero: "))

   if rand == guessing:
      print('Acertou misseravel.')
   else:
      
      if guessing > rand:
         print('Digitaste um numero alto!')
      else:
         print('Digitaste um numero baixo!')

      print('Errou, tente novamente.')
      
   
   errado+=1

print("tentou: ",errado)
