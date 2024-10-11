import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from data_preparation import load_function_data, load_benchmark
from model_loader import load_embedding_model, load_inference_model, load_CL_embedding_model
from embedding import generate_embeddings, get_code_embedding
from vector_db import create_vector_db
from query_augmentation import augment_query
from prompt_generation import generate_prompt
from inference import run_inference
from utils import save_predictions, process_directory
from config import ORACLE_DIR, OUTPUT_DIR
# from config import ORACLE_DIR, OUTPUT_DIR

def main(device):

    torch.cuda.set_device(device)
    
    # Load data
    print("Loading function data and benchmark...")
    functions_with_sig = load_function_data()
    benchmark_df = load_benchmark()

    # # Load models
    print("Loading embedding and inference models...")
    embedding_model, embedding_tokenizer = load_CL_embedding_model(device)
    inference_model, inference_tokenizer = load_inference_model(device)

    # Generate embeddings and create vector DB
    print("Generating embeddings and creating vector DB...")
    code_embeddings = generate_embeddings(functions_with_sig, embedding_model, embedding_tokenizer)

    # assert isinstance(code_embeddings[0], numpy.ndarray)
    # print(code_embeddings[0].shape)

    vector_db = create_vector_db(code_embeddings)

    # # Define augment_query function with loaded models and vector_db
    def augment_query_func(query):
        return augment_query(query, vector_db, functions_with_sig, get_code_embedding, embedding_model, embedding_tokenizer)

    # # Define inference function with loaded models
    def inference_func(sample, context):
        return run_inference(sample, context, inference_model, inference_tokenizer, generate_prompt, augment_query_func)

    # Process each directory
    print("Processing directories...")
    oracle_dir = Path(ORACLE_DIR)

    # repos_to_examine = [repo_dir for repo_dir in oracle_dir.iterdir() if repo_dir.is_dir()][2:3]
    for repo_dir in oracle_dir.iterdir():
        if repo_dir.is_dir():
            process_directory(repo_dir, benchmark_df, inference_func, save_predictions)

    print("RAG pipeline execution completed.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        device = torch.device(f"cuda:{sys.argv[1]}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device)