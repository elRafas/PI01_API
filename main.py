from fastapi import FastAPI

import numpy as np
import pandas as pd

app = FastAPI()

df = pd.read_csv("scr/movies_dataset_ETL_toAPI.csv")

#http://127.0.0.1:8000
@app.get("/")
def index():
    return "https://elrafas-proyecto1.onrender.com/docs"

@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes:str):
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes historicamente'''

    mes = str.lower(mes)
    m_dict = {
        'enero': 1,
        'febrero': 2,
        'marzo': 3,
        'abril': 4,
        'mayo': 5,
        'junio': 6,
        'julio': 7,
        'agosto': 8,
        'septiembre': 9,
        'octubre': 10,
        'noviembre': 11,
        'diciembre': 12
        }

    df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d', errors='coerce')

    count = df[df['release_month'] == m_dict[mes]].shape[0]
    
    return {"mes":mes, "cantidad_filmaciones":count}


@app.get('/cantidad_filmaciones_dia{dia}')
def cantidad_filmaciones_dia(dia:str):
    '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrenaron ese dia historicamente'''

    dia = str.lower(dia)
    d_dict = {
        'lunes': 0,
        'martes': 1,
        'miercoles': 2,
        'jueves': 3,
        'viernes': 4,
        'sabado': 5,
        'domingo': 6,
        }
    
    df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d', errors='coerce')

    dff = df[df['status'] == 'Released']

    count = dff[dff['release_weekday'] == d_dict[dia]].shape[0]
    
    return {"dia":dia, "cantidad_filmaciones": count}



@app.get('/score_titulo/{titulo}')
def score_titulo(titulo:str):
    '''Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score'''

    titulo = str.lower(titulo)
    
    dff = df[df['title'].apply(str.lower) == titulo].sort_values('release_year',ascending=False)
    
    dff.rename(columns={'title': 'titulo', 'release_year': 'anio_estreno','popularity': 'popularidad'}, inplace=True)
    dff = dff[['titulo', 'anio_estreno', 'popularidad']]
    
    return dff.iloc[0].to_dict()

@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo:str):
    '''Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones. 
    La misma variable deberá de contar con al menos 2000 valoraciones, 
    caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.'''

    titulo = str.lower(titulo)

    dff = df[df['title'].apply(str.lower) == titulo].sort_values('release_year',ascending=False)
    
    dff.rename(columns={'title': 'titulo', 'release_year': 'anio','vote_count': 'voto_total', 'vote_average': 'voto_promedio'}, inplace=True)
    dff = dff[['titulo', 'anio', 'voto_total', 'voto_promedio']]


    count = dff['voto_total'].sum()
    avg = dff['voto_promedio'].mean()
    year = dff['anio'].iloc[0]
    
    if count >= 2000:
        return {"titulo":str(titulo), 'anio':int(year), "voto_total":count,"voto_promedio":avg}
    else:
        return {"titulo":str(titulo), 'mensaje': 'No es posible un resultado ya que el título seleccionado tiene pocas valoraciones.'}

@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor:str):
    '''Se ingresa el nombre de un actor que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
    Además, la cantidad de películas que en las que ha participado y el promedio de retorno'''

    nombre_actor = str.lower(nombre_actor)

    mask = df['cast'].apply(str.lower).str.find(nombre_actor) != -1
    dff = df[mask]

    count = dff.shape[0]
    retorno_tot = dff['return'].sum()
    retorno_prom = dff['return'].mean()

    return {"actor": str(nombre_actor), "cantidad_filmaciones": count,"retorno_total": retorno_tot, 'retorno_promedio': retorno_prom}

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):
    ''' Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
    Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.'''

    nombre_director = str.lower(nombre_director)

    dff = df.dropna(subset='director')
    mask = dff['director'].apply(str.lower).str.find(nombre_director) != -1
    dff = dff[mask]  
    
    retorno = dff['return'].sum()

    peliculas = []

    for i in range(dff.shape[0]):
        dic = {}
        dic['title'] = str(dff['title'].iloc[i])
        dic['release_year'] = str(dff['release_year'].iloc[i])
        dic['return'] = str(dff['return'].iloc[i])
        dic['budget'] = str(dff['budget'].iloc[i])
        dic['revenue'] = str(dff['revenue'].iloc[i])
        peliculas.append(dic)

    #return {"director":str(nombre_director), "retorno_promedio_por_filmacion":str(retorno),"peliculas":peliculas}

    return {'director':str(nombre_director), 'retorno_total_director':retorno, 
    'peliculas':dff['title'].tolist(), 'anio':dff['release_year'].tolist(), 'retorno_pelicula':dff['return'].tolist(), 
    'budget_pelicula':dff['budget'].tolist(), 'revenue_pelicula':dff['revenue'].tolist()}



# ML
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    import re
    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    data = pd.read_csv('scr/movies_dataset_ETL_toML.csv', usecols=['id', 'title', 'overview', 'genres'])

    data['tags'] = data['overview'].astype(str) + ". " + data['genres'].astype(str)
    data.drop(columns=['overview', 'genres'])

    data['tags'] = data['tags'].str.lower()

    
    data["tags"] = data["tags"].apply(lambda x: re.sub("[^a-zA-Z]"," ",str(x)))

    
    ps = PorterStemmer()

    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)

    data["tags"] = data["tags"].apply(stem)
    
    cv = CountVectorizer(max_features=1500, stop_words="english")
    vectors = cv.fit_transform(data["tags"]).toarray()

    similares = cosine_similarity(vectors)

    def recomendacion(movie):
        movie = str.lower(movie)
        #busca el index del titulo en minusculas
        movie_index = data[data["title"].apply(str.lower) == movie].index[0]

        distances = similares[movie_index]
        movie_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6]

        out = []

        for i in movie_list:
            out.append(data.iloc[i[0]].title)
        return out

    return {'lista recomendada': recomendacion(titulo)}