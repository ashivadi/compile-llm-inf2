import torch
from transformers_neuronx.module import save_pretrained_split
import os
import torch_neuronx
from transformers import AutoConfig, LlamaTokenizer, TextIteratorStreamer, AutoModelForCausalLM
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
from transformers_neuronx.llama.model import LlamaForSampling
from huggingface_hub import HfApi, snapshot_download, login, logout


os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference'

def create_directory_if_not_exists(path_str):
    """Creates a directory if it doesn't exist, and returns the directory path."""
    if os.path.isdir(path_str):
        return path_str
    elif input(f"{path_str} does not exist, create directory? [y/n]").lower() == "y":
        os.makedirs(path_str)
        return path_str
    else:
        raise NotADirectoryError(path_str)


model_name='meta-llama/Llama-2-13b-hf'
max_length = 100
tp_degree = 6
batch_size = 2
save_path=f"{model_name}-model_save_path"

def download_model(model_name, 
                    save_path,
                    api_key):
    #add HF API here
    huggingface_hub.login(token = api_key)
    model_cpu = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    save_path = create_directory_if_not_exists(save_path)
    save_pretrained_split(model_cpu, save_path)
    print(f"Files for '{model_name}' have been downloaded to '{save_path}'.")
    huggingface_hub.logout()
    

# Assumption is the model is downloaded post EULA / Huggingface login in a directory /opt/ml/model
def compile_llama2(download=False):
    if download:
        download_model(model_name, save_path, 'hf_XXX')

    # Load model config
    print('loading model')
    
    model_neuron = LlamaForSampling.from_pretrained(
            save_path, 
            batch_size=batch_size, 
            tp_degree=tp_degree
        )

    print('Compiling to Neuron')
    model_neuron.to_neuron()
    print('Compiling to Neuron complete')

    model_config = AutoConfig.from_pretrained(save_path)
    #hf_neuron_model = HuggingFaceGenerationModelAdapter(model_config, self.model)
    return model_neuron


 

# if __name__ == "__main__":
#     compile_llama2(download=True) #set to True for the very first run
