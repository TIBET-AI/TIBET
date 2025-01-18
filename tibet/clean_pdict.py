def cleanup_dict(p_dict):

    # sometimes, GPT will return a dict with a key that need to be removed, instead of a list of counterfactuals
    if len(p_dict['result'].keys()) <= 2:

        if 'axes' in p_dict['result']:
            p_dict['result'] = p_dict['result']['axes']

        if 'Counterfactual Prompts' in p_dict['result']:
            p_dict['result'] = p_dict['result']['Counterfactual Prompts']

        if 'counterfactual prompts' in p_dict['result']:
            p_dict['result'] = p_dict['result']['counterfactual prompts']

        if 'counterfactual_prompts' in p_dict['result']:
            p_dict['result'] = p_dict['result']['counterfactual_prompts']
            
        if 'prompts_with_biases' in p_dict['result']:
            p_dict['result'] = p_dict['result']['prompts_with_biases']

        if 'prompt_biases' in p_dict['result']:
            p_dict['result'] = p_dict['result']['prompt_biases']

        if 'biases' in p_dict['result']:
            p_dict['result'] = p_dict['result']['biases']

        if 'Biases' in p_dict['result']:
            p_dict['result'] = p_dict['result']['Biases']

    # if everything is parsed correctly, return the dictionary
    if isinstance(p_dict['result'], dict) and len(p_dict['result'].keys()) > 2:
        for key in p_dict['result'].keys():
            if isinstance(p_dict['result'][key], list):
                if len(p_dict['result'][key]) >= 2 and isinstance(p_dict['result'][key][0], str):
                    print("Successfully parsed bias axes and counterfactuals")
                    return p_dict
    else:
        print("Failed to parse bias axes or counterfactuals")
        return None
    
def setup_prompt_dict(p_dict, ID):

    p_dict = cleanup_dict(p_dict)
    
    # format result keys so that a key like 'Gender biases' becomes 'gender_biases'
    result = p_dict['result']
    new_result = {}
    for key in result.keys():
        new_result[key.lower().replace(' ', '_')] = result[key]
    p_dict['result'] = new_result
    p_dict['id'] = ID
    print("Successfully set up prompt dict")

    return p_dict

    




