internet_celebrities = {"DrDisrespect": "YouTube", "ZLaner": "Facebook", "Ninja": "Mixer"}
another_one = {"shroud": "Twitch"}

internet_celebrities.update(another_one)

print(internet_celebrities)

my_copy_dict = internet_celebrities.copy()

print(my_copy_dict)

internet_celebrities.clear()

print(internet_celebrities)