my_dict = {"Queen": "Bohemian Rhapsody", 
           "Bee Gees": "Stayin' Alive", 
           "U2": "One", 
           "Michael Jackson": "Billie Jean", 
           "The Beatles": "Hey Jude", 
           "Bob Dylan": "Like A Rolling Stone"}

print(len(my_dict))

for key in my_dict.keys():
    print(key)

for value in my_dict.values():
    print(value)

for key, value in my_dict.items():
    print(key, value)

print(my_dict.get("Promise of the Real", "This key does not exists"))