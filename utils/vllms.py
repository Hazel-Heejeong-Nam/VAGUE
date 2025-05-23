# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
import random

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser
import os
from PIL import Image
from tqdm import tqdm

# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

    
# Idefics3-8B-Llama3
def run_idefics3(question, modality, tokenizer):
    if modality =="image":
        prompt = (
            f"<|begin_of_text|>User:<image>{question}<end_of_utterance>\nAssistant:"
        )
    else :
        prompt = (
            f"<|begin_of_text|>User:{question}<end_of_utterance>\nAssistant:"
        )
    return prompt


# InternVL
def run_internvl(question, modality, tokenizer):
    if modality =="image":
        messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    else: 
        messages = [{'role': 'user', 'content': f"{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    return prompt



# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(question, modality, tokenizer):
    if modality =="image":
        prompt = f"[INST] <image>\n{question} [/INST]"
    else:
        prompt = f"[INST] {question} [/INST]"
    return prompt


# LLaVA-OneVision
def run_llava_onevision(question, modality, tokenizer):

    if modality == "image":
        prompt = f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user {question}<|im_end|> \
        <|im_start|>assistant\n"
    return prompt



# NVLM-D
def run_nvlm_d(question, modality, tokenizer):
    if modality =="image":
        messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    else :
        message =  [{'role': 'user', 'content': f"{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt




# Phi-3-Vision
def run_phi3v(question, modality, tokenizer):
    if modality =="image":
        prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
    else :
        prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
    return prompt


# Pixtral HF-format
def run_pixtral_hf(question, modality, tokenizer):
    if modality =="image":
        prompt = f"<s>[INST]{question}\n[IMG][/INST]"
    else :
        prompt = f"<s>[INST]{question}[/INST]"        
    return prompt



# Qwen2.5-VL
def run_qwen2_5_vl(question, modality, tokenizer):

    if modality == "image":
        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n")
    else :
        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                "<|im_start|>assistant\n")

    return prompt

def load_idefics3(model):
    llm = LLM(
        model=model,
        max_model_len=8192,
        max_num_seqs=2,
        enforce_eager=True,
        # if you are running out of memory, you can reduce the "longest_edge".
        # see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3#model-optimizations
        mm_processor_kwargs={
            "size": {
                "longest_edge": 3 * 364
            },
        },
        disable_mm_preprocessor_cache=False,
    )
    return llm, None, None

def load_internvl(model):
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=4096,
        disable_mm_preprocessor_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model,
                                              trust_remote_code=True)
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    
    return llm, tokenizer, stop_token_ids

def load_llava_next(model):
    llm = LLM(model=model,
              max_model_len=3000,
              disable_mm_preprocessor_cache=False)
    return llm, None, None

def load_llava_onevision(model):
    llm = LLM(model=model,
              dtype="float16",
              max_model_len=16384,
              disable_mm_preprocessor_cache=False)

    return llm, None, None

def load_nvlm_d(model):
        # Adjust this as necessary to fit in GPU
    llm= LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=4096,
        tensor_parallel_size=4,
        disable_mm_preprocessor_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model,
                                              trust_remote_code=True)
    return llm, tokenizer, None

def load_phi3v(model):
    llm= LLM(
        model=model,
        dtype="float16",
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={"num_crops": 16},
        disable_mm_preprocessor_cache=False,
    )
    return llm, None, None

def load_pixtral_hf(model):

    # NOTE: Need L40 (or equivalent) to avoid OOM
    llm = LLM(
        model=model,
        tokenizer_mode="mistral",
        max_model_len=8192,
        max_num_seqs=2,
        disable_mm_preprocessor_cache=False,
    )
    return llm, None, None

def load_qwen2_5_vl(model):
    llm = LLM(
        model=model,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        disable_mm_preprocessor_cache=False,
    )
    return llm, None, None