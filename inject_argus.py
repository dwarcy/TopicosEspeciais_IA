import os
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from docs import docs

print("=== INICIANDO INGESTÃO NO PINECONE ===")

# 1. Conexão com o Pinecone
os.environ["PINECONE_API_KEY"] = "pcsk_7Amgef_RcM74ZD1P8v7sQM78WNsXtomMX2v9VcS5W8LzMH2TAgcYdfoXNMqudAJgqqqRjd"
INDEX_NAME = "argus-index-recursivesecond"
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if INDEX_NAME not in pc.list_indexes().names():
    print(f"-> Criando índice '{INDEX_NAME}' na nuvem...")
    pc.create_index(
        name=INDEX_NAME, 
        dimension=384, 
        metric="cosine", 
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"-> Índice '{INDEX_NAME}' encontrado.")

# 2. Carrega o modelo de embedding local
print("-> Carregando modelo de embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Prepara os fatiadores (Splitters)
# splitter_macro = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=["\n\n", "\n", ". ", " ", ""])
# splitter_micro = CharacterTextSplitter(chunk_size=150, chunk_overlap=20, separator=" ")

splitter_macro = CharacterTextSplitter(chunk_size=150, chunk_overlap=20, separator=" ")
splitter_micro = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=["\n\n", "\n", ". ", " ", ""])

docs_macro = []
docs_micro = []

print("-> Fatiando os documentos de mitologia...")
for doc in docs:
    chunks_macro = splitter_macro.create_documents([doc["text"]], metadatas=[{"doc_id": doc["id"]}])
    docs_macro.extend(chunks_macro)
    
    chunks_micro = splitter_micro.create_documents([doc["text"]], metadatas=[{"doc_id": doc["id"]}])
    docs_micro.extend(chunks_micro)

print(f"   Gerados {len(docs_macro)} chunks MACRO e {len(docs_micro)} chunks MICRO.")

# 4. Envia os dados para o Pinecone
print("-> Enviando vetores para o Pinecone...")
PineconeVectorStore.from_documents(docs_macro, embeddings, index_name=INDEX_NAME, namespace="macro")
PineconeVectorStore.from_documents(docs_micro, embeddings, index_name=INDEX_NAME, namespace="micro")

print("\n=== INGESTÃO CONCLUÍDA COM SUCESSO! ===")