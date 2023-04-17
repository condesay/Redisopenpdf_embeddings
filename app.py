from PyPDF2 import PdfReader
import pandas as pd
import os
import openai
from io import BytesIO
from PyPDF2 import PdfReader
import pandas as pdpd
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import os
import requests
import redis
from ast import literal_eval
import numpy as np
from typing import List, Iterator


# Ignore unclosed SSL socket warnings - optional in case you get these errors
import warnings

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

pdf = PdfReader(open("partp34.pdf", "rb"))
number_of_pages = len(pdf.pages)

pdf_text = []
for i in range(number_of_pages):
    page = pdf.pages[i]
    text = page.extract_text()
    if text is not None:
        pdf_text.append({'text': text})
        
def paper_df(pdf):
    filtered_pdf = []
    for row in pdf:
        if len(row['text']) < 30:
            continue
        filtered_pdf.append(row)
    df = pd.DataFrame(filtered_pdf)
    return df

df = paper_df(pdf_text)
print(df)

def calculate_embeddings(df):
    openai.api_key = os.getenv("OPENAI_API_KEY","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")
    embedding_model = "text-embedding-ada-002"
    embeddings = df.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
    df["embeddings"] = embeddings
    print('Done calculating embeddings')
    return df

df = paper_df(pdf_text)
df = calculate_embeddings(df)
print(df)

# Read vectors from strings back into a list
#df['embeddings']=df.embeddings.apply(literal_eval)

from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType,
)
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TextField,
    VectorField,
)

REDIS_HOST ="127.0.0.1"
REDIS_PORT ="6379"
REDIS_PASSWORD = "" # default for passwordless Redis

# Connect to Redis
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD
)


# Constants
VECTOR_DIM = len(df['embeddings'][0]) # length of the vectors
VECTOR_NUMBER = len(df)                 # initial number of vectors
INDEX_NAME = "embeddings-index2"                 # name of the search index
PREFIX2 = "doc2"                                  # prefix for the document keys
DISTANCE_METRIC = "COSINE"                      # distance metric for the vectors (ex. COSINE, IP, L2)

# Define RediSearch fields for each of the columns in the dataset

text = TextField(name="text")
title_embedding = VectorField("embeddings",
    "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER,
    }
)
fields = [text, title_embedding]
try:
    redis_client.ft(INDEX_NAME).info()
    print("Index already exists")
except:
    # Create RediSearch Index
    redis_client.ft(INDEX_NAME).create_index(
        fields = fields,
        definition = IndexDefinition(prefix=[PREFIX2], index_type=IndexType.HASH)
    )

def index_documents(client: redis.Redis, prefix: str, documents: pd.DataFrame):
    records = documents.to_dict("records")
    for doc2 in records:
        key = f"{prefix}:{str(doc2['id'])}"

        # create byte vectors for title and content
        title_embedding = np.array(doc2["embeddings"], dtype=np.float32).tobytes()

        # replace list of floats with byte vectors
        doc2["embeddings"] = title_embedding

        client.hset(key, mapping = doc2)

index_documents(redis_client, PREFIX2, df)
print(f"Loaded {redis_client.info()['db0']['keys']} documents in Redis search index with name: {INDEX_NAME}")


def search_redis(
    redis_client: redis.Redis,
    user_query: str,
    index_name: str = "embeddings-index2",
    vector_field: str = "embeddings",
    return_fields: list = ["text", "vector_score"],
    hybrid_fields = "*",
    k: int = 20,
) -> List[dict]:
    EMBEDDING_MODEL = "text-embedding-ada-002"

    # Creates embedding vector from user query
    embedded_query = openai.Embedding.create(input=user_query,
                                            model=EMBEDDING_MODEL,
                                            )["data"][0]['embedding']

    # Prepare the Query
    base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'
    query = (
        Query(base_query)
         .return_fields(*return_fields)
         .sort_by("vector_score")
         .paging(0, k)
         .dialect(2)
    )
    params_dict = {"vector": np.array(embedded_query).astype(dtype=np.float32).tobytes()}

    # perform vector search
    results = redis_client.ft(index_name).search(query, params_dict)
    for i, Undoc in enumerate(results.docs):
        score = 1 - float(Undoc.vector_score)
        print(f"{i}. {Undoc.title} (Score: {round(score ,3) })")
    return results.docs 
