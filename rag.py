import json
import os
import warnings
from functools import lru_cache
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
import config

os.environ.update({
    'ANONYMIZED_TELEMETRY': 'False',
    'CHROMA_SERVER_HOST': 'localhost'
})
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")

embeddings = OllamaEmbeddings(
    model=config.EMBEDDING_MODEL,
    base_url=config.OLLAMA_URL
)

@lru_cache(maxsize=1)
def load_ux_rules():
    with open(config.DATA_DIR / "ux_rules_optimized.json", 'r', encoding='utf-8') as f:
        return json.load(f)



def get_vectorstore():
    return Chroma(
        persist_directory=config.CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="ux_rules"
    )

def setup_rag():
    try:
        vectorstore = get_vectorstore()
        if len(vectorstore.get()['ids']) > 0:
            print(f"Base existante trouvée avec {len(vectorstore.get()['ids'])} règles")
            return vectorstore
    except:
        pass
    
    rules = load_ux_rules()
    texts = []
    metadatas = []
    ids = []
    
    for rule in rules:
        if rule['content'].strip():
            texts.append(rule['content'])
            metadatas.append(rule)
            ids.append(str(rule['id']))
    
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        embedding=embeddings,
        persist_directory=config.CHROMA_PATH,
        collection_name="ux_rules"
    )
    print(f"{len(rules)} règles ajoutées dans Chroma")
    return vectorstore

def search_rules(query):
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=config.TOP_K)
    return [doc.metadata for doc in docs]