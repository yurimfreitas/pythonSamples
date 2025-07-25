

def counter_func(text="This is a test"):
    list_string = text.split(" ")
    counter = 0
    for i in list_string:
        counter += 1
    
    return counter

print(counter_func("James Bond is 007."))
print(counter_func("When the moon hits your eye like a big pizza pie, that's amore!"))
print(counter_func("Anyway, like I was sayin', shrimp is the fruit of the sea. You can barbecue it, boil it, broil it, bake it, \
saute it. Dey's uh, shrimp-kabobs, shrimp creole, shrimp gumbo. Pan fried, deep fried, stir-fried. There's pineapple \
shrimp, lemon shrimp, coconut shrimp, pepper shrimp, shrimp soup, shrimp stew, shrimp salad, shrimp and potatoes, \
shrimp burger, shrimp sandwich. That- that's about it."))