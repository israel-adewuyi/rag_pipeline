import torch
from torch import nn
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaModel, RobertaTokenizer, BertTokenizer, BertModel
from config import EMBEDDING_MODEL_NAME, INFERENCE_MODEL_NAME, HF_TOKEN, QUANTIZATION_CONFIG, CL_MODEL

def load_embedding_model(device):
    """
    Load the embedding model and tokenizer.
    
    Returns:
    tuple: (model, tokenizer)
    """
    # Original RAG Setup
    tokenizer = RobertaTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = RobertaModel.from_pretrained(EMBEDDING_MODEL_NAME, device_map=device)
    return model, tokenizer

def load_inference_model(device):
    """
    Load the inference model and tokenizer.
    
    Returns:
    tuple: (model, tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(
        INFERENCE_MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=QUANTIZATION_CONFIG,
        device_map=device
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        INFERENCE_MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    tokenizer.use_default_system_prompt = False
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_CL_embedding_model(device):
    """
    Load the CL embedding model and tokenizer.

    Returns:
    tuple: (model, tokenizer)
    """
    cfg = Config()
    model = SentenceEmbeddingModel(cfg, device).to(device)
    model.load_state_dict(torch.load(CL_MODEL, map_location=device))
    tokenizer = RobertaTokenizer.from_pretrained(cfg.model_name)
    return model, tokenizer

@dataclass
class Config:
    model_name: str = "microsoft/codebert-base" # Replace with the intended embedding model
    projection_dim: int = 128
    
class SentenceEmbeddingModel(nn.Module):
    def __init__(self, cfg: Config, device: torch.device):
        super(SentenceEmbeddingModel, self).__init__()
        self.cfg = cfg
        self.device = device  # Add device attribute
        self.embeddingmodel = RobertaModel.from_pretrained(self.cfg.model_name).to(self.device)
        # self.embeddingmodel = BertModel.from_pretrained(self.cfg.model_name).to(self.device)
        self.projection = nn.Linear(self.embeddingmodel.config.hidden_size, self.cfg.projection_dim).to(self.device)

        
    def forward(self, input_ids, attention_mask):
        outputs = self.embeddingmodel(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected_embedding = self.projection(cls_embedding)

        return projected_embedding