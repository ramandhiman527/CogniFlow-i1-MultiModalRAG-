from langchain_ollama import OllamaEmbeddings
#from sentence_transformers import SentenceTransformer


def get_embedding_function():
    """
    Use the nomic-embed-text model for embeddings.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

#simple embedding function for demonstration purpose
