import torch
from transformers import GenerationConfig

def run_inference(sample, context, model, tokenizer, generate_prompt, augment_query):
    """
    Run inference using the loaded model.
    
    Args:
    sample: A sample from the benchmark data
    context (str): Additional context for the prompt
    model: The inference model
    tokenizer: The tokenizer for the inference model
    generate_prompt: Function to generate the prompt
    augment_query: Function to augment the query
    
    Returns:
    list: Generated outputs
    """
    GENERATION_CONFIG = GenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        num_beams=10,
        num_beam_groups=10,
        diversity_penalty=1.0,
        num_return_sequences=10,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    with torch.no_grad():
        prompt = generate_prompt(sample, context, augment_query)
        inputs = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        torch.cuda.empty_cache()

        outputs = model.generate(
            inputs,
            generation_config=GENERATION_CONFIG,
        )

        torch.cuda.empty_cache()
        
    # print("got output in inference")
    return [tokenizer.decode(
        output[len(inputs[0]) :], skip_special_tokens=True
    ).strip() for output in outputs]