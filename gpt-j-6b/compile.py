from transformers import AutoModelForCausalLM
import torch
from transformers_neuronx.module import save_pretrained_split
import os
from transformers_neuronx.gptj.model import GPTJForSampling

os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference'

def amp_callback(model, dtype):
    # cast attention and mlp to low precisions only; layernorms stay as f32
    for block in model.transformer.h:
        block.attn.to(dtype)
        block.mlp.to(dtype)
    model.lm_head.to(dtype)

def compile_gptj(download=False):
    if download:
        model_cpu = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B')
        amp_callback(model_cpu, torch.bfloat16)
        save_pretrained_split(model_cpu, './gpt-j-6B')
    print('loading model')
    model_neuron = GPTJForSampling.from_pretrained(
        'gpt-j-6B', 
        batch_size=1,    # Must match batch size that's used during inference
        tp_degree=2,
        n_positions=512, # Maximum sequence length (input prompt len + generated tokens)
        amp='bf16', 
    )
    print('Converting to Neuron')
    model_neuron.to_neuron()
    print('Converted to Neuron')
    return model_neuron

if __name__ == "__main__":
    compile_gptj(download=False) #set to True for the very first run