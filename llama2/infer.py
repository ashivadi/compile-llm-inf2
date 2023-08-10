from transformers import LlamaTokenizer
import time
import torch 
from compile import compile_llama2

model_name='meta-llama/Llama-2-13b-hf'
max_length = 100
tp_degree = 6
batch_size = 2
save_path=f"{model_name}-model_save_path"

def sample(model_neuron, tokenizer, batch_prompts, seqlen):

    input_ids = torch.as_tensor([tokenizer.encode(text) for text in batch_prompts])

    with torch.inference_mode():
        start = time.time()
        generated_sequences = model_neuron.sample(input_ids, sequence_length=seqlen)
        elapsed = time.time() - start

    generated_sequences = [tokenizer.decode(seq) for seq in generated_sequences]
    print(f'generated sequences {generated_sequences} in {elapsed} seconds')

if __name__ == "__main__":
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model_neuron = compile_llama2()
    batch_prompts = [
        "Hello, I'm a language model,",
        "Welcome to Amazon Elastic Compute Cloud,",
    ]
    sample(model_neuron, tokenizer, batch_prompts, seqlen=512)

