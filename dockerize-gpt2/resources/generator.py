
import logging
from threading import Lock
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers_neuronx.module import save_pretrained_split
import os
from transformers_neuronx.gpt2.model import GPT2ForSampling
from flask import Flask, Response, request, jsonify

os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference'

logging.basicConfig(format='%(levelname)s:%(process)d:%(message)s', level=logging.INFO)
logger = logging.getLogger("gpt2")

class Gpt2Generator:
    lock = Lock()
    generator = None

    @classmethod
    def get_generator(cls):
        """load trained model"""
        with cls.lock:
            # check if model is already loaded
            if cls.generator:
                return cls.generator

            try:
                model_dir = os.environ["SM_MODEL_DIR"]
            except KeyError:
                model_dir = "/opt/ml/model"

            try:
                max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])
            except KeyError:
                max_new_tokens = 512

            cls.generator = Gpt2Generator(path=model_dir, max_new_tokens=max_new_tokens)
            return cls.generator
        
    def amp_callback(self, model, dtype):
        # cast attention and mlp to low precisions only; layernorms stay as f32
        for block in model.transformer.h:
            block.attn.to(dtype)
            block.mlp.to(dtype)
        model.lm_head.to(dtype)

    def __init__(self, path:str, max_new_tokens: int) -> None:
        logger.info("Create tokenizer and model")
        # create directory to store split weights
        os.makedirs('gpt2-split', exist_ok=True)
        # Create the tokenizer and model
        model_cpu = AutoModelForCausalLM.from_pretrained('gpt2')
        self.amp_callback(model_cpu, torch.bfloat16)
        save_pretrained_split(model_cpu, './gpt2-split')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model_neuron = GPT2ForSampling.from_pretrained(
            'gpt2-split', 
            batch_size=1,    # Must match batch size that's used during inference
            tp_degree=2,
            n_positions=512, # Maximum sequence length (input prompt len + generated tokens)
            amp='bf16', 
        )
        model_neuron.to_neuron()
        
        self.__model = model_neuron
        self.__tokenizer = tokenizer
        self.__max_new_tokens = max_new_tokens
        logger.info("Init complete.")
    
    def sample(self, model_neuron, tokenizer, text, seqlen):

        encoded_input = tokenizer(text, return_tensors='pt')

        input_ids = encoded_input.input_ids
        input_length = input_ids.shape[1]
        new_tokens = seqlen - input_length

        print(f'input prompt length: {input_length}')
        print(f'generated tokens: {new_tokens}')

        model_neuron.reset()
        sample_output = model_neuron.sample(
            input_ids,
            sequence_length=seqlen,
            top_k=50,
        )

        print('generated outputs:')
        r = [tokenizer.decode(tok) for tok in sample_output]
        print(r[0])
        return r[0]

    def __call__(self, user_input: str) -> str:
    
        logger.info(f"Model input: {user_input}")
        return self.sample(self.__model, self.__tokenizer, user_input, seqlen=self.__max_new_tokens)
    


app = Flask(__name__)
gpt2_generator = Gpt2Generator.get_generator()
logger.info(f"Got generator: {gpt2_generator}")

@app.route("/ping", methods=["GET"])
def health_check():
    status = 200
    return Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def inference():
    if not request.is_json:
        result = {"error": "Content type is not application/json"}
        logger.error(result)
        response = jsonify(result)
        response.staus = 415
        return response
    
    try: 
        content = request.get_json()
        logger.info(f"Content: {content}")
        
        user_input = content['user_input']
        output = gpt2_generator(user_input)
                   
        result = { "output": output }
        response=jsonify(result)
        response.status = 200
        
        logger.info(f"Invocation Response: {response}")
        return response
    except Exception as e:
        logger.error(str(e))
        raise e