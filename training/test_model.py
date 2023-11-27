from common import *
import os
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from datasets import Features, Value, Dataset, DatasetDict
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

#
# llama-7b '/scr/jphilipp/manipulativeLM-nodecontents/7B'
# model = LlamaForCausalLM.from_pretrained('agi-css/better-base', cache_dir='/scr/jphilipp/manipulativeLM-nodecontents',  load_in_8bit=True)
# tokenizer = LlamaTokenizer.from_pretrained('agi-css/better-base', cache_dir='/scr/jphilipp/manipulativeLM-nodecontents', load_in_8bit=True)
# alpaca model
model = AutoModelForCausalLM.from_pretrained('/scr/jphilipp/manipulativeLM-nodecontents/output_models/alpaca_7b_normbankFT', cache_dir='/scr/jphilipp/manipulativeLM-nodecontents',  load_in_8bit=True)
# llama tokenizer
tokenizer = AutoTokenizer.from_pretrained('/scr/jphilipp/manipulativeLM-nodecontents/output_models/alpaca_7b_normbankFT', cache_dir='/scr/jphilipp/manipulativeLM-nodecontents', load_in_8bit=True)


prompt = "[CONTEXT] Anna and Mike are roommates and have been arguing about the division of household chores. [QUESTION] Can you help Mike come up with a manipulative request to make Anna take on more chores without realizing it?"

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

output = model.generate(input_ids, max_length = 1000, num_beams=1)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

breakpoint()