
import logging
import torch
import os
from threading import Lock
from flask import Flask, Response, request, jsonify, send_file
from PIL import Image
from wrapper import NeuronTextEncoder, UNetWrap, NeuronUNet
import torch_neuronx
from diffusers.models.cross_attention import CrossAttention
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import base64


os.environ['NEURON_RT_NUM_CORES'] = '6' #inf2.24xlarge
model_id = "stabilityai/stable-diffusion-2-1-base"
logging.basicConfig(
            format='%(levelname)s:%(process)d:%(message)s',
            level=logging.INFO)

logging.basicConfig(format='%(levelname)s:%(process)d:%(message)s', 
                    level=logging.INFO)
logger = logging.getLogger(model_id)

class Sd2Generator:
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

            cls.generator = Sd2Generator()
            return cls.generator
        

    def __init__(self) -> None:
        logger.info("Initializing Model")
        model_dir='sd21-model-wts'
        dtype = torch.bfloat16
        # create directory to store split weights
        # logger.info("Loading model parts...")
        print("Loading model parts...")
        text_encoder_filename = os.path.join(model_dir, 'text_encoder/model.pt')
        decoder_filename = os.path.join(model_dir, 'vae_decoder/model.pt')
        unet_filename = os.path.join(model_dir, 'unet/model.pt')
        post_quant_conv_filename = os.path.join(model_dir, 'vae_post_quant_conv/model.pt')

        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print('done setting pipe...')
        # Load the compiled UNet onto two neuron cores.
        print('Loading compiled  Unet onto two neuron cores')
        pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
        pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), None, set_dynamic_batching=False)

        # Load other compiled models onto a single neuron core.
        print('Loading compiled text encoder to a neuran core')
        pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
        pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
        logger.info('Loading compiled decoder encoder to a neuran core')
        pipe.vae.decoder = torch.jit.load(decoder_filename)
        print('Loading compiled Quant Conv  to a neuran core')
        pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
        print('Loading of models complete. Ready for inference..')
        self.__pipe = pipe

    def __call__(self, input_req) -> str:
        #input_req={
        #    "prompt": "football match; van gogh style",
        #    # more info about these 2 params here: https://huggingface.co/blog/stable_diffusion
        #    "num_inference_steps": 25,
        #    "guidance_scale": 7.5
        #}
        image = self.__pipe(input_req['prompt']).images[0]
        return image


app = Flask(__name__)
sd2_generator = Sd2Generator.get_generator()
print(f"Got generator: {sd2_generator}")

@app.route("/ping", methods=["GET"])
def health_check():
    status = 200
    return Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def inference():
    if not request.is_json:
        result = {"error": "Content type is not application/json"}
        print('error :', result)
        response = jsonify(result)
        response.staus = 415
        return response
    
    try: 
        content = request.get_json()
        print(f"Content: {content}")
        prompt = content['prompt']
        output = sd2_generator(content)
        output.save('/tmp/test.jpg')
        print('Saved output in test.jpg')
        return send_file('/tmp/test.jpg')
    except Exception as e:
        logger.error(str(e))
        raise e