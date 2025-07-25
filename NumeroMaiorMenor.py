import random

guessing = int(input("Digite um numero:"))
rand=random.randint(1, 10)
1
if guessing > rand:
   print("Numero muito alto.")
elif guessing < rand:
   print("Numero muito pequeno.")
elif guessing == rand:
   print("Acertou miseravel.")

