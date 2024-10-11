import json
import pandas as pd

from config import JSON_DATA_PATH

def load_function_data():
    """
    Load function data from a JSON file.
    
    Returns:
    list: A list of tuples containing (function, signature)
    """
    with open(JSON_DATA_PATH, 'r') as f:
        data = json.load(f)
    
    # return [(item['function'], item['signature']) for item in data]
    return data

def load_benchmark():
    """
    Load benchmark data from a CSV file.
    
    Returns:
    pandas.DataFrame: The loaded benchmark data
    """
    return pd.read_csv('bench-v0.6_with_signature.csv', index_col=0)