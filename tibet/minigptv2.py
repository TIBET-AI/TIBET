#############
# THIS FILE IS A MODIFICATION TO THE `demo_v2.py` FILE IN THIS REPO: https://github.com/Vision-CAIR/MiniGPT-4
############

import argparse
import os
import random
from tqdm import tqdm

import numpy as np
from PIL import Image
import torch

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='vqa_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cudnn.benchmark = False
cudnn.deterministic = True

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

device = 'cuda:{}'.format(args.gpu_id)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
#bounding_box_size = 100

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

model = model.eval()

chat = Chat(model, vis_processor, device=device)

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

temperature = 0.6

def prepare_texts(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(
        conv.roles[0], '<Img><ImageHere></Img> {}'.format(text)) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts

def get_description(image_folder, mini_gpt_questions, num_images=8):
    # first for the initial prompts
    prompt_descriptions = []
    for i in tqdm(range(num_images)):

        image_url = image_folder +str(i)+'.jpeg'
        image = Image.open(image_url).convert('RGB')
        image = vis_processor(image)

        texts = prepare_texts(mini_gpt_questions, CONV_VISION)  # warp the texts with conversation template
        answers = model.generate(image.unsqueeze(0).repeat(len(mini_gpt_questions), 1, 1, 1), texts, max_new_tokens=50, do_sample=False)

        prompt_descriptions.append(answers)
        
    return prompt_descriptions

def build_mini_gpt_questions(questions):
    #mini_gpt_questions = ['Short description of the image: ']
    mini_gpt_questions = []
    for q in questions:
        update_q = '[vqa] '+q
        # if 'person' in update_q.split(' '):
        #     update_q = update_q + "If you don't know, just say 'no'."
        mini_gpt_questions.append(update_q)
    return mini_gpt_questions

def get_concepts(p_dict, image_folder, num_images=8):

    id = p_dict['id']
    image_base_url = os.path.join(image_folder,id)
    questions = build_mini_gpt_questions(p_dict['questions'])

    if not os.path.exists(image_base_url):
        print("Directory not found: ", image_base_url)
        return None
    else:
        num_image_per_set = len(os.listdir(image_base_url+'/initial_prompts/'))

    if 'concepts_initial' not in p_dict.keys():
        p_dict['concepts_initial'] = []

    if 'concepts_initial' in p_dict.keys() and len(p_dict['concepts_initial']) < num_image_per_set:
        initial_descriptions = get_description(image_base_url+'/initial_prompts/', questions, num_images=num_images)
        p_dict['concepts_initial'] = initial_descriptions
    else:
        print("Initial descriptions already present: ", id)

    if 'concepts_cf' not in p_dict.keys():
        p_dict['concepts_cf'] = {}

    if 'concepts_cf' in p_dict.keys():
        for key in p_dict['result'].keys():
            num_cf = len(p_dict['result'][key])

            if key not in p_dict['concepts_cf'].keys():
                p_dict['concepts_cf'][key] = [[] for _ in range(num_cf)]
            
            for icf in range(num_cf):
                
                if len(p_dict['concepts_cf'][key][icf]) > 1 and p_dict['concepts_cf'][key][icf][0] != '':
                    print("Skipping", id, key, icf)
                    continue
                
                # get descriptions of 48 images of the cf
                desc_cf = get_description(image_base_url+'/'+key+'/'+str(icf)+'/', questions, num_images=num_images)

                p_dict['concepts_cf'][key][icf] = desc_cf   

    print('Done generating concepts: ', id)
    return p_dict
