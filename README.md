# RAG-PIPELINE

This code base was uploaded to show what I have been working on for the past ~2 months, as part of my application to Palisade Research and is not intended to be used outside the box. 

- In `config.py`, change the model name as well as the base-tokenizer. If there is a trained embedding model, you can also include it. Model name should be how it's represented on HF.

- run `main.py` with `python main.py 0` where 0 is the cuda card to load the embedding and inference model and tokenizer on.
