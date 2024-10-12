from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(
        model="nemotron-mini"
    )
    return embeddings
