import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector

embeddings = OpenAIEmbeddings()

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "localhost"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "pgvec_local"),
    user=os.environ.get("PGVECTOR_USER", "pj_local"),
    password=os.environ.get("PGVECTOR_PASSWORD", "a4209e80538b5a0f2ae7"),
)


store = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name="budget_2023",
)

retriever = store.as_retriever()

query = "how is angel tax applicable to investors?"
docs = retriever.get_relevant_documents(query)

chat = OpenAI()


chain = load_qa_chain(llm=chat, chain_type="stuff")
response = chain.run(input_documents=docs, question=query)
print(response)
