from transformers import AutoModelForCausalLM
import torch
from transformers_neuronx.module import save_pretrained_split
import os
from transformers.models.opt import OPTForCausalLM
from transformers import AutoTokenizer
from transformers_neuronx.opt.model import OPTForSampling

os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference'

def amp_callback(model, dtype):
    # cast attention and mlp to low precision only; layernorms stay as f32
    for block in model.model.decoder.layers:
        block.self_attn.to(dtype)
        block.fc1.to(dtype)
        block.fc2.to(dtype)
    model.lm_head.to(dtype)

def compile_opt13b(download=False):
    if download:
        model_cpu = OPTForCausalLM.from_pretrained('facebook/opt-13b', low_cpu_mem_usage=True)
        amp_callback(model_cpu, torch.bfloat16)
        save_pretrained_split(model_cpu, './opt-13b-split')

    print('loading model')
    
    model_neuron = OPTForSampling.from_pretrained(
        './opt-13b-split', 
        batch_size=2, 
        tp_degree=2, 
        amp='f16')
    
    print('Converting to Neuron')
    model_neuron.to_neuron()
    print('Converted to Neuron')
    return model_neuron

if __name__ == "__main__":
    compile_opt13b(download=True) #set to True for the very first run