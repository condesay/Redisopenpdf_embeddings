import os
import streamlit as st
import PyPDF2
import pandas as pd
import openai
import redis
import numpy as np

# Définition des constantes
EMBEDDING_MODEL = "text-embedding-ada-002"
VECTOR_DIM = 768
DISTANCE_METRIC = "COSINE"
INDEX_NAME = "embeddings-index"
PREFIX = "doc"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# Définition des champs RediSearch
text = TextField(name="text")
embedding_field = VectorField(
    "embedding",
    "FLAT",
    {"TYPE": "FLOAT", "SIZE": VECTOR_DIM, "DISTANCE_METRIC": DISTANCE_METRIC},
)
fields = [text, embedding_field]

# Connexion à Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

# Fonction pour extraire le texte d'un fichier PDF et créer un dataframe pandas
def create_dataframe_from_pdf(pdf_file):
    with open(pdf_file, "rb") as f:
        reader = PyPDF2.PdfFileReader(f)
        pages = [reader.getPage(i) for i in range(reader.getNumPages())]
        texts = [page.extractText() for page in pages]
    df = pd.DataFrame({"text": texts})
    df.drop_duplicates(subset=["text"], keep="first", inplace=True)
    return df.reset_index(drop=True)

# Fonction pour calculer les embeddings des données textuelles
def calculate_embeddings(df):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    embeddings = df["text"].apply(
        lambda x: openai.Embedding.create(input=x, model=EMBEDDING_MODEL)["data"][0]["embedding"]
    )
    df["embedding"] = embeddings
    return df

# Fonction pour créer un index RediSearch et indexer les embeddings
def create_index(df):
    try:
        redis_client.ft(INDEX_NAME).info()
    except redis.exceptions.ResponseError:
        redis_client.ft_create(
            INDEX_NAME,
            *fields,
            prefix=PREFIX,
            index_type=IndexType.HASH,
        )

    for i, row in df.iterrows():
        redis_client.hset(f"{PREFIX}:{i}", mapping=row.to_dict())

# Fonction pour effectuer une recherche RediSearch et retourner les résultats
def search(query, k=10):
    embedded_query = openai.Embedding.create(input=query, model=EMBEDDING_MODEL)["data"][0]["embedding"]
    results = redis_client.ft_search(INDEX_NAME, f"@embedding:[{','.join(map(str, embedded_query))}]",
                                     limit=k, with_scores=True)
    return results

