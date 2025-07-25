import random

# Generate a random integer between 1 and 10
print(random.randint(1, 6))

dice = int(random.random()*6+1)


while (dice>8):
    dice = int(random.random()*6+1)
    print(dice)


