import requests
import os

API_KEY = "8ac3cf4daa51a20ecee043de223c5392"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def get_weather(latitude, longitude, api_key):
    params = {
        'lat': latitude,
        'lon': longitude,
        'appid': api_key  # Use a chave da API fornecida como argumento
    }
    r = requests.get(BASE_URL, params=params)
    return r.json()

def kelvin_to_celsius(temp, rounding_decimals=1):
        return round(temp - 273.15, rounding_decimals)

list_of_locations = [
        (41.18153135487473, -8.693420558573623),
        (41.124739030442505, -8.612817457006539)
]

for location in list_of_locations:
    w = get_weather(*location, API_KEY)
    location_name = w.get('name', 'N/A') 
    weather_desc = w.get('weather', [{'description': 'N/A'}])[0]['description']
    temp = kelvin_to_celsius(w.get('main', {}).get('temp', 0))  
    print(f'Nome: {location_name}\n'
          f'{weather_desc}\n'
          f'{temp} ºC\n'
          '\n'
          )