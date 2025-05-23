import torch
import json
import os
from tqdm import tqdm
from utils.util import get_prompt
from openai import OpenAI
from IPython.display import Image, display, Audio, Markdown
import base64
API = "enter_your_key"
CAP_PROMPT = "Describe the provided image in 2~3 sentences."

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def run_gpt(args, data):
    client = OpenAI(api_key=API)
    f = open(f'gpt_result_{args.format}_{args.answer_type}.txt', "a", encoding="utf-8") 

    answer_list = {}
    for d in tqdm(data):
        imgname = d['image_name'] + '_annot.jpg'

        if args.format == "SM":
            image = encode_image(os.path.join(args.img_dir, imgname))
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": CAP_PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image}"},
                            },
                        ],
                    },
                ],
                temperature=0.0,
            )
            caption = response.choices[0].message.content
            del d['meta']['caption']
            d['meta']['caption'] = caption
        mcq_prompt, is_image = get_prompt(args, d)
        if is_image : 
            image = encode_image(os.path.join(args.img_dir, imgname))
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": mcq_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image}"},
                            },
                        ],
                    },
                ],
                temperature=0.0,
            )
        else :
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": mcq_prompt
                            }
                        ],
                    },
                ],
                temperature=0.0,
            )
        # autoregressively complete prompt
        answer = response.choices[0].message.content
        if args.answer_type=='mcq':
            answer_list[imgname] = {'model_output' : answer, 'true_answer' : d['mcq']['ordering'][0]}
        else : # da 
            answer_list[imgname] = {'model_output' : answer, 'true_answer' : d['mcq']['1_correct']}
        f.write(f'{imgname}, {answer}\n')

    return answer_list
