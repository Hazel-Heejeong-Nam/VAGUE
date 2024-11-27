from utils.exp_prompts import *
import json
def get_icl(path):
    with open(path, 'r') as f:
        data = json.load(f)
    icl_caps = [d['meta']['caption'] for d in data]
    icl_indirects = [d['indirect'] for d in data]
    icl_mcqs = []
    for d in data:
        mcq = [d['mcq']['1_correct'], d['mcq']['2_incorrect_fake_scene'], d['mcq']['3_incorrect_surface_understanding'], d['mcq']['4_incorrect_entity']]
        mcq_list = sorted([d['mcq']['ordering'][i] + ': ' + mcq[i] for i in range(4)])
        mcq_prompt = '\n'.join(mcq_list) 
        icl_mcqs.append(mcq_prompt)       
    icl_reasonings = [d['meta']['reasoning'] for d in data]
    icl_ans_c = [d['mcq']['ordering'][0] for d in data]
    icl_ans_full = [d['mcq']['1_correct'] for d in data]
    
    return icl_caps, icl_indirects, icl_mcqs, icl_reasonings, icl_ans_c, icl_ans_full

def get_prompt(args, d):
    mcq = [d['mcq']['1_correct'], d['mcq']['2_incorrect_fake_scene'], d['mcq']['3_incorrect_surface_understanding'], d['mcq']['4_incorrect_entity']]
    mcq_list = sorted([d['mcq']['ordering'][i] + ': ' + mcq[i] for i in range(4)])
    mcq_prompt = '\n'.join(mcq_list)
    p = d['solution'].split(',')[0][1:]
    if args.format == 'plain' and args.answer_type=='mcq':
        prompt = plain_mcq(d['indirect'], mcq_prompt)
        image = True
    elif args.format == 'textonly' and args.answer_type=='mcq':
        prompt = plain_mcq(d['indirect'], mcq_prompt)
        image = False
    elif args.format == 'caption' and args.answer_type=='mcq':
        prompt = caption_mcq(d['meta']['caption'], d['indirect'], mcq_prompt)
        image = False
    elif args.format == 'caption_icl' and args.answer_type=='mcq':
        icl_caps, icl_indirects, icl_mcqs, icl_reasonings, icl_ans_c, icl_ans_full = get_icl(args.icl_path)
        prompt = caption_icl_mcq(d['meta']['caption'], d['indirect'], mcq_prompt, icl_caps, icl_indirects, icl_mcqs, icl_ans_c)
        image = False
    elif args.format == 'zeroshot_cot' and args.answer_type=='mcq':
        prompt = zeroshot_cot_mcq(d['indirect'], mcq_prompt)
        image = True
    elif args.format == 'caption_icl_cot' and args.answer_type=='mcq':
        icl_caps, icl_indirects, icl_mcqs, icl_reasonings, icl_ans_c, icl_ans_full = get_icl(args.icl_path)
        prompt = caption_icl_cot_mcq(d['meta']['caption'], d['indirect'], mcq_prompt, icl_caps, icl_indirects, icl_mcqs, icl_reasonings, icl_ans_c)
        image = False
    elif args.format == 'plain' and args.answer_type=='da':
        prompt = plain_da(d['indirect'], p)
        image = True
    elif args.format == 'textonly' and args.answer_type=='da':
        prompt = plain_da(d['indirect'], p)
        image = False
    elif args.format == 'caption' and args.answer_type=='da':
        prompt = caption_da(d['meta']['caption'], d['indirect'], p)
        image = False
    elif args.format == 'caption_icl' and args.answer_type=='da':
        icl_caps, icl_indirects, icl_mcqs, icl_reasonings, icl_ans_c, icl_ans_full = get_icl(args.icl_path)
        prompt = caption_icl_da(d['meta']['caption'], d['indirect'], p, icl_caps, icl_indirects, icl_ans_full)
        image = False
    elif args.format == 'zeroshot_cot' and args.answer_type=='da':
        prompt = zeroshot_cot_da(d['indirect'], p)
        image = True
    elif args.format == 'caption_icl_cot' and args.answer_type=='da':
        icl_caps, icl_indirects, icl_mcqs, icl_reasonings, icl_ans_c, icl_ans_full = get_icl(args.icl_path)
        prompt = caption_icl_cot_da(d['meta']['caption'], d['indirect'], p, icl_caps, icl_indirects, icl_reasonings, icl_ans_full)
        image = False
    else :
        raise ValueError(f"format not found : {args.format}")
    
    return prompt, image