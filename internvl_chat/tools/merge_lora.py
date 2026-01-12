import argparse
import os
import random
import sys

sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])
import torch
from internvl.model.internvl_chat import InternVLChatModel
from transformers import AutoTokenizer

# argparse = argparse.ArgumentParser()
# argparse.add_argument('input_path', type=str,default='/home/jxzhang/paper_codes/InternVL/internvl_chat/shell/phi3_3_8b_dynamic/output/InternVL2B/question_aware/charthuman_vision/checkpoint-1000', help='Path to the input model')
# argparse.add_argument('output_path',default='/home/jxzhang/paper_codes/InternVL/internvl_chat/shell/phi3_3_8b_dynamic/output/InternVL2B/question_aware/charthuman_vision', type=str, help='Path to the output model')
# args = argparse.parse_args()
input_path = '/home/jxzhang/paper_codes/InternVL/internvl_chat/shell/phi3_3_8b_dynamic/output/InternVL2B/question_aware/docvqa_vision/checkpoint-3000'
output_path = '/home/jxzhang/paper_codes/InternVL/internvl_chat/shell/phi3_3_8b_dynamic/output/InternVL2B/question_aware/docvqa_vision'
print('Loading model...')
model = InternVLChatModel.from_pretrained(
    input_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)

if model.config.use_backbone_lora:
    model.vision_model.merge_and_unload()
    model.vision_model = model.vision_model.model
    model.config.use_backbone_lora = 0
if model.config.use_llm_lora:
    model.language_model.merge_and_unload()
    model.language_model = model.language_model.model
    model.config.use_llm_lora = 0

print('Saving model...')
model.save_pretrained(output_path)
print('Saving tokenizer...')
tokenizer.save_pretrained(output_path)
print('Done!')