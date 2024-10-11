from tqdm import tqdm
import pandas as pd
import os
from config import OUTPUT_DIR

def save_predictions(code_, save_path):
    """
    Save predictions to a CSV file.
    
    Args:
    code_ (dict): Dictionary of predictions
    save_path (str): Path to save the CSV file
    """
    df_to_save = pd.DataFrame.from_dict(code_, orient='index', columns=[f'{i}' for i in range(10)])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_to_save.to_csv(save_path)
def process_directory(repo_dir, df, inference_func, save_predictions_func):
    """
    Process a directory of files.
    
    Args:
    repo_dir (str): Path to the repository directory
    df (pandas.DataFrame): Benchmark DataFrame
    inference_func: Function to run inference
    save_predictions_func: Function to save predictions
    """
    code = {}
    print(f"Processing {repo_dir}")
    save_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(repo_dir)}_predictions.csv")
    
    if os.path.exists(save_path):
        saved_df = pd.read_csv(save_path)
        code = saved_df.set_index('Unnamed: 0').T.to_dict(orient='list')
    
    file_list = [f for f in os.listdir(repo_dir) if os.path.isfile(os.path.join(repo_dir, f))]
    for file_path in tqdm(file_list, desc=f"Files in {os.path.basename(repo_dir)}"):
        index = os.path.splitext(file_path)[0]
        # print(index)
        if index in code:
            print(f"{index} is already processed")
            continue
        
        with open(os.path.join(repo_dir, file_path), 'r') as f:
            context = f.read()
        # print("getting sample")
        sample = df.loc[df.index == index].iloc[0]
        # print("doing inference")
        code[index] = inference_func(sample, context)
        # print("Finished this iteration")
        save_predictions_func(code, save_path)
    
    save_predictions_func(code, save_path)