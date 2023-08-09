import os
import torch
import torch_neuronx
import numpy as np
import time

from PIL import Image
from wrapper import NeuronTextEncoder, UNetWrap, NeuronUNet
from diffusers.models.cross_attention import CrossAttention
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

COMPILED_MODEL_WTS_DIR = 'sd21-model-wts'
os.environ['NEURON_RT_NUM_CORES'] = '6'
model_id = "stabilityai/stable-diffusion-2-1-base"
dtype = torch.bfloat16


def load_model_wts(model_dir):
    print("Loading model parts...")
    t=time.time()
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
    print('Loading compiled decoder encoder to a neuran core')
    pipe.vae.decoder = torch.jit.load(decoder_filename)
    print('Loading compiled Quant Conv  to a neuran core')
    pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
    print(f"Done. Elapsed time: {(time.time()-t)}seconds")
    return pipe

def prediction_fn(pipe):
    input_req={
        "prompt": "football match; van gogh style",
        # more info about these 2 params here: https://huggingface.co/blog/stable_diffusion
        "num_inference_steps": 25,
        "guidance_scale": 7.5
    }
    image = pipe(input_req['prompt']).images[0]
    image.save('results.jpg')


if __name__ == "__main__":
    sd_pipe = load_model_wts(COMPILED_MODEL_WTS_DIR)
    prediction_fn(sd_pipe)
