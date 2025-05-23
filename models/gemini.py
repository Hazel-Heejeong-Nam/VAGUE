import torch
from PIL import Image
import json
import os
from tqdm import tqdm
from utils.util import get_prompt
import google.generativeai as genai
import io

CAP_PROMPT = "Describe the provided image in 2~3 sentences."
API_KEY = "enter_your_key"

def load_image(image_path):
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
    return Image.open(io.BytesIO(image_data))


    return response.text

def run_gemini(args, data):
    f = open(f'gemini_result_{args.format}_{args.answer_type}.txt', "a", encoding="utf-8") 

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro")

    answer_list = {}
    for i, d in enumerate(tqdm(data)):
        imgname = d['image_name'] + '_annot.jpg'

        if args.format == "SM":
            image = load_image(os.path.join(args.img_dir, imgname))
            caption = model.generate_content(
                [CAP_PROMPT, image],
                stream=False 
            )
            del d['meta']['caption']
            d['meta']['caption'] = caption
        mcq_prompt, is_image = get_prompt(args, d)
        if is_image : 
            image = load_image(os.path.join(args.img_dir, imgname))
            answer = model.generate_content(
                [mcq_prompt, image],
                stream=False 
            )
        else :
            answer =  model.generate_content(
                [mcq_prompt],
                stream=False 
            ) 
        
        answer = answer.text

        if args.answer_type=='mcq':
            answer_list[imgname] = {'model_output' : answer, 'true_answer' : d['mcq']['ordering'][0]}
        else : # da 
            answer_list[imgname] = {'model_output' : answer, 'true_answer' : d['mcq']['1_correct']}
        f.write(f'{imgname}, {answer}')
    return answer_list
