from config import MAX_CONTEXT_LEN
def generate_prompt(row, context, augment_query):
    """
    Generate a prompt for the inference model.
    
    Args:
    row: A row from the benchmark DataFrame
    context (str): Additional context for the prompt
    augment_query: Function to augment the query
    
    Returns:
    list: A list of message dictionaries for the chat model
    """
    lang = "C++" if row["repository"] not in ["openssl", "redis"] else "C"
    func_name = " ".join(row["fname"].replace("\n", " ").split())
    function_description = row["doc"]

    # system_messamge = {
    #     "role": "system", 
    #     "content": '\nGenerate a function on {lang} programming language.\n You are also provided other similar functions that may or may not be helpful. Feel free to use or discard them, as you see fit.'   
    # }

    system_message = {
        "role": "system",
        "content": f"\nGenerate a function on {lang} programming language.\n You are provided with the function name, and some short description. You are also provided other similar functions that may or may not be helpful. Feel free to use or discard them, as you see fit. Assume every other dependency or external functions or variables have been declared, so generate only a single function. Assume all the preprocessor directives have also been included. Again, generate only a single function.",
    }
    
    # cont = f"\nUse this context:\n {context}" if context else ""
    cont = ""
    
    list_of_similar_functions = augment_query(func_name)
    similar_functions = 'Here are the provided functions that you could consider \n' + '\n'.join(list_of_similar_functions)
    
    cont += similar_functions

    cont = cont[: MAX_CONTEXT_LEN]
    
    prompt = {
        "role": "user",
        "content": f'Function name `{func_name}`.\nFunction description: "{function_description}".{cont}',
    }

    return [system_message, prompt]