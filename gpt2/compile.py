from transformers import AutoModelForCausalLM
import torch
from transformers_neuronx.module import save_pretrained_split
import os
from transformers_neuronx.gpt2.model import GPT2ForSampling

os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference'

def amp_callback(model, dtype):
    # cast attention and mlp to low precisions only; layernorms stay as f32
    for block in model.transformer.h:
        block.attn.to(dtype)
        block.mlp.to(dtype)
    model.lm_head.to(dtype)

def compile_gpt2(download=False):
    if download:
        model_cpu = AutoModelForCausalLM.from_pretrained('gpt2')
        amp_callback(model_cpu, torch.bfloat16)
        save_pretrained_split(model_cpu, './gpt2-split')
    model_neuron = GPT2ForSampling.from_pretrained(
        'gpt2-split', 
        batch_size=1,    # Must match batch size that's used during inference
        tp_degree=2,
        n_positions=512, # Maximum sequence length (input prompt len + generated tokens)
        amp='bf16', 
    )
    model_neuron.to_neuron()
    return model_neuron



