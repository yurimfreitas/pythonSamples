def retangule_are(length, width, height):
    return length*width*height


length_user = int(input("Input length: "))
width_user = int(input("\nInput width: "))
height_user = int(input("\nInput height: "))

print("The area is: "+str(retangule_are(length_user, width_user, height_user)))