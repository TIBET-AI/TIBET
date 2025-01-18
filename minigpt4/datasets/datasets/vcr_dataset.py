import torch
from PIL import Image
import os
import json

from minigpt4.datasets.datasets.base_dataset import BaseDataset

class VCRData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

        self.ans_options = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]

        image_path = os.path.join(self.root_path, ann["img_fn"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        image_boxes_path = os.path.join(self.root_path, ann['metadata_fn'])
        with open(image_boxes_path) as f:
            boxes = json.load(f)

        boxes = self.convert_bb(boxes)
        base_question, box_list = self.replace_q_bboxes(ann['question'], self.convert_bb(boxes))

        options = []
        for answer in ann['answer_choices']:
            options.append(self.replace_q_bboxes(answer, self.convert_bb(boxes))[0])

        question = f'[vqa] Based on the image, pick the correct option. {base_question}'
        for iz, option in enumerate(options):
            question += ' \n '+self.ans_options[iz]+': '+option
        question += '\n Answer: '
        labels = 'true' if ann["label"] == 1 else 'false'

        return image, question, labels
    
    def convert_bb(boxes):
        width = boxes['width']
        height = boxes['height']
        box_list = boxes['boxes']

        new_boxes = []
        for box in box_list:
            new_box = "{<"+str(round(box[0]/width*100)) +"><"+str(round(box[1]/height*100))+"><"+str(round(box[2]/width*100))+"><"+str(round(box[3]/height*100))+">}"
            new_boxes.append(new_box)
        return new_boxes


    def replace_q_bboxes(question, boxes):
        box_list = []
        for i, part in enumerate(question):
            if isinstance(part, list):
                replacement_str = ''
                for idx in part:
                    replacement_str += boxes[idx]+' '
                    box_list.append(boxes[idx])
                question[i] = replacement_str.strip()

        return ' '.join(question), box_list