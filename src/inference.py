import argparse
import json
from models import *
import os
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default='cuda:1', type=str)
    parser.add_argument("--img_dir", default='/mnt/vague/annotated_images', type=str)
    parser.add_argument("--data_path", default="data/vague_benchmark.json", type=str)
    parser.add_argument("--icl_path", default="data/vague_fewshot.json", type=str)
    parser.add_argument("--out_dir", default='results', type=str)
    parser.add_argument("--answer_type", default='mcq', choices=[
        'mcq', 
        'da'])
    parser.add_argument("--format", default="zeroshot_cot", choices = [
        'plain',
        'caption',
        'zeroshot_cot',
        'caption_icl', 
        'caption_icl_cot',
        'textonly'
    ])
    parser.add_argument("--model", default='internvl_25b', choices=[
        'llava_15_vicuna_13b',
        'llava_next_mistral_7b', 
        'instructblip_7b',
        'llava_next_vicuna_13b',
        'internvl_8b',
        'sharegpt4v_7b',
        'sharegpt4v_13b',
        'internvl_25b'
    ])
    args = parser.parse_args()

    return args

def main_worker(args):
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    fname = f"{args.model}_{args.format}_{args.answer_type}"
    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.isfile(os.path.join(args.out_dir, f'{fname}.json')):
        raise ValueError('Your result already exists. Please check your configuration again.')
    
    if args.model == 'instructblip_7b':
        answer_list = run_instructblip_7b(args, data)
    elif args.model == 'llava_next_mistral_7b':
        answer_list = run_llava_next_mistral_7b(args, data)
    elif args.model == 'llava_next_vicuna_13b':
        answer_list = run_llava_next_vicuna_13b(args, data)
    elif args.model == 'llava_15_vicuna_13b':
        answer_list = run_llava_15_vicuna_13b(args, data)
    elif args.model == 'internvl_25b':
        answer_list = run_internvl_25b(args, data)
    elif args.model == 'internvl_8b':
        answer_list = run_internvl_8b(args, data)
    # elif args.model == "llama_vision_11b":
    #     answer_list = run_llama_vision_11b(args, data)
    # elif args.model == "llava_gemma_llb":
    #     answer_list = run_llava_gemma_11b(args,data)
    # elif args.model == "sharegpt4v_7b":
    #     answer_list = run_sharegpt4v_7b(args, data)
    # elif args.model == "sharegpt4v_13b":
    #     answer_list = run_sharegpt4v_13b(args, data)
    else:
        raise ValueError(f'Invalid model encountered : {args.model}')
    
    
    with open(os.path.join(args.out_dir, f'{fname}.json'), 'w') as f:
        json.dump(answer_list, f, indent=4)

if __name__ =="__main__":
    args = parse_args()
    main_worker(args)