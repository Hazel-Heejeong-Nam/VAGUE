import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import json
import os
from tqdm import tqdm
from utils.util import get_prompt

def run_llava_15_vicuna_13b(args, data):
    model_id = "llava-hf/llava-1.5-13b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(args.device)

    processor = AutoProcessor.from_pretrained(model_id)
    answer_list = {}
    for d in tqdm(data):
        imgname = d['image_name'] + '_annot.jpg'
        mcq_prompt, is_image = get_prompt(args, d)
        if is_image : 
            image = Image.open(os.path.join(args.img_dir, imgname))
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": mcq_prompt},
                    {"type": "image"},
                    ],
                }]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt, return_tensors='pt').to(args.device, torch.float16)
        else :
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": mcq_prompt}
                    ],
                }]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=prompt, return_tensors="pt").to(args.device, torch.float16)
        
        output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
        answer = processor.decode(output[0][2:], skip_special_tokens=True).split('ASSISTANT:')[-1].strip()
        if args.answer_type=='mcq':
            answer_list[imgname] = {'model_output' : answer, 'true_answer' : d['mcq']['ordering'][0]}
        else : # da 
            answer_list[imgname] = {'model_output' : answer, 'true_answer' : d['mcq']['1_correct']}

    return answer_list