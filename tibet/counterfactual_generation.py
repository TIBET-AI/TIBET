import os
from openai import OpenAI
import json
import time
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

def get_response(messages, client, get_json=False, gpt_4=False):
    response_format = {'type': 'json_object'} if get_json else None
    gpt_model = "gpt-4-1106-preview" if gpt_4 else 'gpt-3.5-turbo-1106'

    response = client.chat.completions.create(
        model=gpt_model,
        messages=messages,
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format=response_format
    )
    return response


def chain_of_prompting(initial_prompt, client, change_prompt=False, gpt_4=False):

    messages = []

    # Prompt 1:
    prompt1 = "For the image generation prompt \""+initial_prompt.strip()+"\", what are some of the axes where the prompt may lead to biases in the image?"
    messages.append({"role": "user", "content":prompt1})
    res1 = get_response(messages, client, gpt_4=gpt_4)
    messages.append({'role': res1.choices[0].message.role, 'content': res1.choices[0].message.content})

    # Prompt 2:
    if change_prompt:
        prompt2 = "Generate many counterfactuals for each axis. Create counterfactuals for all diverse alternatives for an axis. Each counterfactual should look exactly like the original prompt, with only one concept changed at a time."
    else:
        prompt2 = "Generate many counterfactuals for each axis. Create counterfactuals for all diverse alternatives for an axis. Each counterfactual should look similar to the original prompt, with only one concept changed, removed, or added at a time."
    messages.append({"role": "user", "content":prompt2})
    res2 = get_response(messages, client, gpt_4=gpt_4)
    messages.append({'role': res2.choices[0].message.role, 'content': res2.choices[0].message.content})

    # Prompt 3:
    prompt3 = "Convert these to a json dictionary where the axes are the keys and the counterfactuals are list for each key. Only return json."
    messages.append({"role": "user", "content":prompt3})
    res3 = get_response(messages, client, get_json=True, gpt_4=gpt_4)
    
    # Parse Json
    json_out = res3.choices[0].message.content
    try:
        data = json.loads(json_out)
    except:
        res3 = get_response(messages, client, gpt_4=gpt_4)
        json_out = res3.choices[0].message.content
        try:
            data = json.loads(json_out)
        except:
            data = None

    messages.append({'role': res3.choices[0].message.role, 'content': res3.choices[0].message.content})

    return messages, data


def get_counterfactuals(initial_prompt, client, gpt4=False):

    done = False
    count_fail = 0
    change_prompt = False
    
    # Retry until successful, upto 10 times
    while not done:
        messages, data = chain_of_prompting(initial_prompt, client, change_prompt=change_prompt, gpt_4=gpt4)
        if data is not None:
            input_tokens = word_tokenize(initial_prompt)
            sample_axis = [key for key in data.keys()][0]
            sample_tokens = word_tokenize(data[sample_axis][0])
            if len(sample_tokens) < (len(input_tokens) + 5) and len(sample_tokens) > (len(input_tokens) - 5):
                done = True
            elif count_fail > 2:
                change_prompt = True
                print('Retrying in 5 seconds, with alternative prompt')
                time.sleep(5)
            elif count_fail > 10:
                done = True
                data = None
            else:
                count_fail += 1
                change_prompt = False
                print('Retrying in 5 seconds')
                time.sleep(5)

    if data is not None:
        p_dict = {
            'initial_prompt': initial_prompt,
            'gpt_response': messages,
            'result': data
        }
        print("Generated counterfactuals for prompt: "+initial_prompt)
        return p_dict
    else:
        print("Failed: "+initial_prompt)
        return None

    