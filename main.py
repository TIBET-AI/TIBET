
import os
import json
from openai import OpenAI

from tibet.counterfactual_generation import get_counterfactuals
from tibet.clean_pdict import setup_prompt_dict
from tibet.gen_questions import generate_questions
from tibet.gen_images import generate_images
from tibet.minigptv2 import get_concepts
from tibet.metrics.plotting import plot_BAV_and_CAS, get_CAS_data


# Step 1: Configurations
NUM_IMAGES = 2 # Number of images to generate per prompt and counterfactural prompt
IMG_DIR = 'images'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

PROMPT = "A photo of a computer programmer"
PROMPT_ID = 'P_1'

client = OpenAI(api_key=OPENAI_API_KEY)
save_path = PROMPT_ID+'.json'

# Step 2: Discover Biases and Counterfactuals using GPT
p_dict = get_counterfactuals(PROMPT, client, gpt4=False)
p_dict = setup_prompt_dict(p_dict, PROMPT_ID)

# Step 3: Generate Questions
p_dict = generate_questions(p_dict)

with open(save_path, 'w') as f:
    json.dump(p_dict, f, indent=4)

# Step 4: Generate Images (using Stable Diffusion 2.1)
p_dict = generate_images(p_dict, IMG_DIR, NUM_IMAGES)

# Step 5: Extract Concepts using VQA with MiniGPT-v2
p_dict = get_concepts(p_dict, IMG_DIR, NUM_IMAGES)

with open(save_path, 'w') as f:
    json.dump(p_dict, f, indent=4)

# Step 6: Compute Bias and Variance
variance_data, CAS_data, CAS_labels = get_CAS_data(p_dict, debug=True)
plot_BAV_and_CAS(variance_data, CAS_data, CAS_labels, PROMPT_ID+'_BAV_and_CAS.png')