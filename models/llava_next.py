from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import json
import os
from tqdm import tqdm
from utils.util import get_prompt

CAP_PROMPT = "Describe the provided image in 2~3 sentences."

def run_llava_next_vicuna_13b(args, data):
    if args.format == "SM_gpt":
        with open('data/vague_SMgpt.json', 'r') as rebcap:
            gptcap = json.load(rebcap)

    
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to(args.device)
    answer_list = {}
    for d in tqdm(data):
        imgname = d['image_name'] + '_annot.jpg'
 ###########################
        if (args.format=="SM") or (args.format=="zeroshot_cot_SM"):
            image = Image.open(os.path.join(args.img_dir, imgname))
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": CAP_PROMPT},
                    {"type": "image"},
                    ],
                }]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(args.device)
            output = model.generate(**inputs, max_new_tokens=1000)
            caption = processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT:')[-1].strip()
            del d['meta']['caption']
            d['meta']['caption'] = caption
        if args.format == "SM_gpt":
            del d['meta']['caption']
            d['meta']['caption'] = gptcap[imgname]            
###############################    
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
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(args.device)
        else :
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": mcq_prompt}
                    ],
                }]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=prompt, return_tensors="pt").to(args.device)

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=1000)

        answer = processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT:')[-1].strip()
        if args.answer_type=='mcq':
            answer_list[imgname] = {'model_output' : answer, 'true_answer' : d['mcq']['ordering'][0]}
        else : # da 
            answer_list[imgname] = {'model_output' : answer, 'true_answer' : d['mcq']['1_correct']}

    return answer_list
