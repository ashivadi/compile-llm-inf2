import re
import logging
import random
from threading import Lock
import os
from transformers import pipeline, set_seed
import torch
import torch_neuronx
from transformers import DistilBertTokenizer, DistilBertModel

logging.basicConfig(format='%(levelname)s:%(process)d:%(message)s', level=logging.INFO)
logger = logging.getLogger("generator")

class DistilBertGenerator:
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
                max_new_tokens = 128

            cls.generator = DistilBertGenerator(path=model_dir, max_new_tokens=max_new_tokens)
            return cls.generator

    def __init__(self, path:str, max_new_tokens: int) -> None:
        logger.info("Create tokenizer and model")
        # Create the tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        #tokenizer.pad_token = tokenizer.eos_token
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # Get an example input
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(
                        text,
                        max_length=128,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
        
        example = (
            encoded_input['input_ids'],
            encoded_input['attention_mask'],
        )
        # Compile the model for Neuron
        model= torch_neuronx.trace(model, example)
        self.__model = model
        self.__tokenizer = tokenizer
        self.__max_new_tokens = max_new_tokens
        logger.info("Init complete.")


    def __call__(self, user_input: str) -> str:
    
        logger.info(f"Model input: {user_input}")
        encoded_input = self.__tokenizer(
                user_input,
                max_length=self.__max_new_tokens,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        example = (
            encoded_input['input_ids'],
            encoded_input['attention_mask'],
        )
        response = self.__model(*example)
        logger.info(f"Model output: {response}")
        
        output = str(response)
        logger.info(f"Generated output: {output}")
        return output
    
import base64
import json
import tempfile
import threading

from flask import Flask, Response, request, jsonify

app = Flask(__name__)
generator = DistilBertGenerator.get_generator()
logger.info(f"Get generator: {generator}")

set_seed(random.randint(0, 10000000))

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
        output = generator(user_input)
                   
        result = { "output": output }
        response=jsonify(result)
        response.status = 200
        
        logger.info(f"Invocation Response: {response}")
        return response
    except Exception as e:
        logger.error(str(e))
        raise e