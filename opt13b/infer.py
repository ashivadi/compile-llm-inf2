from transformers import AutoTokenizer
import time
import torch 
from compile import compile_opt13b

def sample(model_neuron, tokenizer, batch_prompts, seqlen):

    input_ids = torch.as_tensor([tokenizer.encode(text) for text in batch_prompts])

    with torch.inference_mode():
        start = time.time()
        generated_sequences = model_neuron.sample(input_ids, sequence_length=2048)
        elapsed = time.time() - start

    generated_sequences = [tokenizer.decode(seq) for seq in generated_sequences]
    print(f'generated sequences {generated_sequences} in {elapsed} seconds')

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-13b')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_neuron = compile_opt13b()
    batch_prompts = [
        "Hello, I'm a language model,",
        "Welcome to Amazon Elastic Compute Cloud,",
    ]
    sample(model_neuron, tokenizer, batch_prompts, seqlen=512)

