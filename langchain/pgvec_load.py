import os

from dotenv import load_dotenv

load_dotenv()


from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector

loader = PyPDFLoader("data/budget_speech.pdf")
pages = loader.load_and_split()


embeddings = OpenAIEmbeddings()


CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "localhost"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "pgvec_local"),
    user=os.environ.get("PGVECTOR_USER", "pj_local"),
    password=os.environ.get("PGVECTOR_PASSWORD", "a4209e80538b5a0f2ae7"),
)

db = PGVector.from_documents(
    embedding=embeddings,
    documents=pages,
    collection_name="budget_2023",
    connection_string=CONNECTION_STRING,
)
