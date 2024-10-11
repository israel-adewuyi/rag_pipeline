import torch
import numpy as np
from transformers import RobertaModel

# def get_code_embedding(piece, model, tokenizer):
#     """
#     Generate embedding for a given piece of code.
    
#     Args:
#     piece (str): The code piece to embed
#     model: The embedding model
#     tokenizer: The tokenizer for the embedding model
    
#     Returns:
#     numpy.ndarray: The embedding vector
#     """
#     tokens = tokenizer(piece, return_tensors='pt', truncation=True, padding=True)
#     tokens = {k: v.to(model.device) for k, v in tokens.items()}
#     with torch.no_grad():
#         outputs = model(**tokens)
#     return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def get_code_embedding(piece, model, tokenizer):
    """
    Generate embedding for a given piece of code.

    Args:
    piece (str): The code piece to embed
    model: The embedding model
    tokenizer: The tokenizer for the embedding model

    Returns:
    numpy.ndarray: The embedding vector
    """
    tokens = tokenizer(piece, return_tensors='pt', truncation=True, padding=True)

    if isinstance(model, RobertaModel):
        # Handle RobertaModel or similar models that expect **tokens
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    else:
        # Handle custom models like SentenceEmbeddingModel
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            embeddings = model(input_ids, attention_mask)

        return embeddings.cpu().numpy().squeeze()


def generate_embeddings(functions_with_sig, model, tokenizer):
    """
    Generate embeddings for a list of functions.
    
    Args:
    functions_with_sig (list): List of (function, signature) tuples
    model: The embedding model
    tokenizer: The tokenizer for the embedding model
    
    Returns:
    list: List of embedding vectors
    """
    return [get_code_embedding(sig, model, tokenizer) for _, sig in functions_with_sig]