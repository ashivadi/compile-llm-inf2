from transformers import AutoTokenizer
import time
from compile import compile_gpt2

def sample(model_neuron, tokenizer, text, seqlen):

    encoded_input = tokenizer(text, return_tensors='pt')

    input_ids = encoded_input.input_ids
    input_length = input_ids.shape[1]
    new_tokens = seqlen - input_length

    print(f'input prompt length: {input_length}')
    print(f'generated tokens: {new_tokens}')

    start = time.time()
    model_neuron.reset()
    sample_output = model_neuron.sample(
        input_ids,
        sequence_length=seqlen,
        top_k=50,
    )
    end = time.time()

    print('generated outputs:')
    print([tokenizer.decode(tok) for tok in sample_output])
    
    throughput = (seqlen) / (end - start)
    print(f'\nthroughput: {throughput} tok/sec')

    text = "Hello, I'm a language model,"

    r = sample(model_neuron, tokenizer, text, seqlen=512)
    print(r[0])

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_neuron = compile_gpt2()
    text = "Hello, I'm a language model,"
    sample(model_neuron, tokenizer, text, seqlen=512)

