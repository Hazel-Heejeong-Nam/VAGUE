from InternLM_XComposer.projects.ShareGPT4V.share4v.mm_utils import get_model_name_from_path
from InternLM_XComposer.projects.ShareGPT4V.share4v.eval.run_share4v import eval_model
from InternLM_XComposer.projects.ShareGPT4V.share4v.model.builder import load_pretrained_model
import torch
from InternLM_XComposer.projects.ShareGPT4V.share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from InternLM_XComposer.projects.ShareGPT4V.share4v.conversation import SeparatorStyle, conv_templates
from InternLM_XComposer.projects.ShareGPT4V.share4v.mm_utils import (KeywordsStoppingCriteria,get_model_name_from_path, tokenizer_image_token)
from InternLM_XComposer.projects.ShareGPT4V.share4v.model.builder import load_pretrained_model
from InternLM_XComposer.projects.ShareGPT4V.share4v.utils import disable_torch_init
import requests
from io import BytesIO
from PIL import Image
import os
from tqdm import tqdm
from utils.util import get_prompt


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def run_sharegpt4v_13b(args, data):

    disable_torch_init()
    model_path = "Lin-Chen/ShareGPT4V-13B"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=model_path,model_base=None, model_name=get_model_name_from_path(model_path), device_map=args.device, device=args.device)

    conv_mode = "share4v_v0"
    answer_list = {}
    for d in tqdm(data):
        imgname = d['image_name'] + '_annot.jpg'
        qs, is_image = get_prompt(args, d)
        if is_image : 
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image = load_image(os.path.join(args.img_dir, imgname))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(args.device)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)       
        
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
                
        else :
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)       
        
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=None,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
                

        input_token_len = input_ids.shape[1]            
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        answer = outputs.strip()
            
        if args.answer_type=='mcq':
            answer_list[imgname] = {'model_output' : answer, 'true_answer' : d['mcq']['ordering'][0]}
        else : # da 
            answer_list[imgname] = {'model_output' : answer, 'true_answer' : d['mcq']['1_correct']}

    return answer_list
            