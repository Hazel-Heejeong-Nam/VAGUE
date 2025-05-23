import argparse
import json
import os
from models import *
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--img_dir", default='/mnt/vague/ego4d_final_annotated', type=str)
    # parser.add_argument("--img_dir", default='/mnt/vague/annotated_images', type=str)
    parser.add_argument("--data_path", default="data/ego4d_533_final.json", type=str)
    # parser.add_argument("--data_path", default="data/vcr_1144_final.json", type=str)
    parser.add_argument("--icl_path", default="icl.json", type=str)
    parser.add_argument("--out_dir", default='results', type=str)
    parser.add_argument("--answer_type", default='mcq', choices=[
        'mcq', 
        'da'])
    parser.add_argument("--format", default="SM_gpt", choices = [
        'VLM',
        'SM',
        'SM_gpt',
        'LLM',
        'zeroshot_cot',
        'zeroshot_cot_SM',
        'caption_icl_cot',
    ])
    parser.add_argument("--model", default='llava-hf/llava-v1.6-vicuna-13b-hf', choices=[
        'microsoft/Phi-3.5-vision-instruct', 
        'llava-hf/llava-onevision-qwen2-7b-ov-hf', 
        'llava-hf/llava-v1.6-vicuna-13b-hf', 
        'gemini-1.5-pro',
        'gpt-4o'
    ])
    parser.add_argument("--model_type", default=None)
    parser.add_argument("--vllm", action="store_true")
    
    args = parser.parse_args()

    return args

name_to_type = {
    'microsoft/Phi-3.5-vision-instruct': "phi3_v", 
    'llava-hf/llava-onevision-qwen2-7b-ov-hf': "llava-onevision",
    'llava-hf/llava-v1.6-vicuna-13b-hf': "llava-next",
}

def main_worker(args):
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    fname = f"{args.model.split('/')[-1]}_{args.format}_{args.answer_type}"
    print(fname)
    os.makedirs(args.out_dir, exist_ok=True)
    
    if os.path.isfile(os.path.join(args.out_dir, f'{fname}.json')):
        raise ValueError('Your result already exists. Please check your configuration again.')
    
    if args.vllm:
        args.model_type = name_to_type[args.model]
        answer_list = run_vllm_models(args,data)
    elif 'llava-v1.6' in args.model:
        answer_list = run_llava_next_vicuna_13b(args, data)
    elif 'gemini' in args.model:
        answer_list = run_gemini(args, data)
    elif 'gpt' in args.model:
        answer_list = run_gpt(args, data)
    with open(os.path.join(args.out_dir, f'{fname}.json'), 'w') as f:
        json.dump(answer_list, f, indent=4)

if __name__ =="__main__":
    args = parse_args()
    main_worker(args)

