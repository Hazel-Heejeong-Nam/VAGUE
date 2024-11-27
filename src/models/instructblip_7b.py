from PIL import Image
import json
import os
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
from utils.util import get_prompt

def run_instructblip_7b(args, data):

    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    model.to(args.device)
        
    answer_list = {}
    for d in tqdm(data):
        imgname = d['image_name'] + '_annot.jpg'
        mcq_prompt, is_image = get_prompt(args, d)
        if is_image : 
            image = Image.open(os.path.join(args.img_dir, imgname))
            inputs = processor(images=image, text=mcq_prompt, return_tensors="pt").to(args.device)

        else :
            inputs = processor(images=None, text=mcq_prompt, return_tensors="pt").to(args.device)

        output = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=2048,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        answer = processor.decode(output[0], skip_special_tokens=True)
        if args.answer_type=='mcq':
            answer_list[imgname] = {'model_output' : answer, 'true_answer' : d['mcq']['ordering'][0]}
        else : # da 
            answer_list[imgname] = {'model_output' : answer, 'true_answer' : d['mcq']['1_correct']}

    return answer_list