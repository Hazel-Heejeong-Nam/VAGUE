from vllm import LLM, SamplingParams
import json
import os
from PIL import Image
from tqdm import tqdm
from utils.vllms import *
from utils.util import get_prompt

model_prompt_map = {
    "idefics3": run_idefics3,
    "internvl_chat": run_internvl,
    "llava-next": run_llava_next, 
    "llava-onevision": run_llava_onevision,
    "NVLM_D": run_nvlm_d,
    "phi3_v": run_phi3v, 
    "pixtral_hf": run_pixtral_hf, 
    "qwen2_5_vl": run_qwen2_5_vl,
}

model_load_map = {
    "idefics3": load_idefics3,
    "internvl_chat": load_internvl,
    "llava-next": load_llava_next,
    "llava-onevision": load_llava_onevision,
    "NVLM_D": load_nvlm_d,
    "phi3_v": load_phi3v,
    "pixtral_hf": load_pixtral_hf,
    "qwen2_5_vl": load_qwen2_5_vl,
}


CAP_PROMPT = "Describe the provided image in 2~3 sentences."


def run_vllm_models(args, data):
    if args.format == "SM_gpt":
        with open('data/vague_SMgpt.json', 'r') as rebcap:
            gptcap = json.load(rebcap)
    
    if args.model_type not in model_prompt_map:
        raise ValueError(f"Model type {args.model_type} is not supported.")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[-1]
    llm, tokenizer, stop_token_ids = model_load_map[args.model_type](args.model)
    sampling_params = SamplingParams(temperature=0.2,
                                max_tokens=1024,
                                stop_token_ids=stop_token_ids)
    answer_list = {}
    for d in tqdm(data):
        imgname = d['image_name'] + '_annot.jpg'
        
 ###########################
        if (args.format=="SM") or (args.format=="zeroshot_cot_SM"):
            data = Image.open(os.path.join(args.img_dir, imgname))
            prompt  = model_prompt_map[args.model_type](CAP_PROMPT, "image", tokenizer)
            inputs = {"prompt": prompt, "multi_modal_data": {"image": data}}
            o_caption = llm.generate(inputs, sampling_params=sampling_params)
            for oc in o_caption:
                caption = oc.outputs[0].text
            del d['meta']['caption']
            d['meta']['caption'] = caption
            # print(imgname, caption)
        if args.format == "SM_gpt":
            del d['meta']['caption']
            d['meta']['caption'] = gptcap[imgname]            
###############################
    
        mcq_prompt, is_image = get_prompt(args, d)
        if is_image:
            data = Image.open(os.path.join(args.img_dir, imgname))
            modality = "image"
            prompt  = model_prompt_map[args.model_type](mcq_prompt, modality, tokenizer)
            inputs = {"prompt": prompt, "multi_modal_data": {"image": data}}
        else: 
            modality = None

            prompt = model_prompt_map[args.model_type](mcq_prompt, modality, tokenizer)
            inputs = {"prompt": prompt}


        outputs = llm.generate(inputs, sampling_params=sampling_params)

        assert len(outputs)==1
        for o in outputs:
            generated_text = o.outputs[0].text
            
        if args.answer_type=='mcq':
            answer_list[imgname] = {'model_output' : generated_text, 'true_answer' : d['mcq']['ordering'][0]}
        else : # da 
            answer_list[imgname] = {'model_output' : generated_text, 'true_answer' : d['mcq']['1_correct']}
    return answer_list
