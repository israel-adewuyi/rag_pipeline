import numpy as np
from config import TOP_K

def augment_query(query, vector_db, functions_with_sig, get_code_embedding, model, tokenizer):
    """
    Augment a query with similar function contexts.
    
    Args:
    query (str): The input query
    vector_db: The FAISS vector database
    functions_with_sig (list): List of (function, signature) tuples
    get_code_embedding: Function to get code embedding
    model: The embedding model
    tokenizer: The tokenizer for the embedding model
    
    Returns:
    list: List of augmented contexts
    """
    query_embedding = get_code_embedding(query, model, tokenizer)
    query_embedding = query_embedding.reshape(1, -1)
    
    distances, I = vector_db.search(query_embedding, TOP_K)

    distances = distances[0]
    I = I[0]

    if distances[0] == 0.0:
        I = I[:-1]
    else:
        I = I[1:]
    
    augmented_contexts = [functions_with_sig[i][0] for i in I]
    return augmented_contexts