import faiss
import numpy as np

def create_vector_db(code_embeddings):
    """
    Create a FAISS vector database from code embeddings.
    
    Args:
    code_embeddings (list): List of embedding vectors
    
    Returns:
    faiss.IndexFlatL2: The created vector database
    """
    dimension = code_embeddings[0].shape[0]
    vector_db = faiss.IndexFlatL2(dimension)
    vectors = np.array([embedding for embedding in code_embeddings])
    vector_db.add(vectors)
    return vector_db