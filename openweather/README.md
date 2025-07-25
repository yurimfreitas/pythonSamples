# OpenWeatherAPI

Este é um exemplo de como extrair dados de uma API HTTP. Neste caso, estamos
a usar a [OpenWeatherAPI](https://openweathermap.org/api).

Podes criar a tua própria conta [aqui](https://home.openweathermap.org/users/sign_up)
para obter uma chave de API grátis.

## Instalação das dependências

Instalar a library `requests` com este comando:

```
pip install -r requirements.txt
```

## Uso

Definir a *environment variable* `OPENWEATHER_API_KEY`.

Em Linux ou Mac:

```
$ export OPENWEATHER_API_KEY=<a-tua-chave-aqui>
```

Em Windows:

```
> setx OPENWEATHER_API_KEY "<a-tua-chave-aqui>"
```

Correr o script com o seguinte comando:

```
$ python openweatherapi.py
```

Podes editar o script para obter informações sobre diferentes coordenadas ou
extrair diferentes atributos da resposta dada pela API.
