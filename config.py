import torch
from transformers import BitsAndBytesConfig, GenerationConfig

# Data Configuration
JSON_DATA_PATH = "/.../.../.../Israel/function_extraction/functions.json"

# Model Configuration
EMBEDDING_MODEL_NAME = "microsoft/codebert-base"
INFERENCE_MODEL_NAME = "mistralai/Codestral-22B-v0.1"
HF_TOKEN = "your_hugging_face_token"

# Vector DB Configuration
TOP_K = 3

# Max length of input string
MAX_CONTEXT_LEN = 12000

# Path to Contrastive Learning Model
CL_MODEL = "/.../.../.../Israel/CLCodeBertModel_02.pth"

# Quantization Configuration
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

ORACLE_DIR = "oracle/oracle"

# Output Configuration
OUTPUT_DIR = "output"