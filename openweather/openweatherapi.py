#!/usr/bin/env python
import os

import requests

API_KEY = 'a8fbefbbd69002c31368cd8b732fa91f'
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def get_weather(latitude, longitude):
    params = {
        'lat': latitude,
        'lon': longitude,
        'appid': API_KEY
    }
    #https://api.openweathermap.org/data/2.5/weather?lat=41.18153135487473&lon=-8.693420558573623&appid=a8fbefbbd69002c31368cd8b732fa91f
    r = requests.get(BASE_URL, params=params)
    return r.json()

def kelvin_to_celsius(temp, rounding_decimals=1):
        return round(temp - 273.15, rounding_decimals)

list_of_locations = [
        (41.18153135487473, -8.693420558573623),
        (41.124739030442505, -8.612817457006539)
]



for location in list_of_locations:
    w = get_weather(*location)
    print(w)
    location_name = w['name']
    weather_desc = w['weather'][0]['description']
    temp = kelvin_to_celsius(w['main']['temp'])
    print(f'Nome: {location_name}\n'
          f'{weather_desc}\n'
          f'{temp} ºC\n'
          '\n'
          )

