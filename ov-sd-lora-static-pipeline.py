from diffusers.pipeline_utils import DiffusionPipeline
from transformers import CLIPTokenizer
from typing import Union, List, Optional, Tuple
import cv2
from openvino.runtime import Model, Core

import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
from diffusers.utils import load_image
from ov_static.ov_sd_static_pipeline import OVStableDiffusionPipeline

hf_model_dir = "./models/HF_models"
ir_model_dir = "./models/IR_models/FP32_static"
lora_model_dir = "./models/Lora"
scheduler_dir = "scheduler"
dataset_dir = "./datasets"
output_dir = "./results"
device = "CPU"

core = Core()
# core.set_property({'CACHE_DIR': './cache'})
tokenizer = CLIPTokenizer.from_pretrained(f'{hf_model_dir}/clip-vit-large-patch14')

TEXT_ENCODER_OV_PATH=f"{ir_model_dir}/text_encoder.xml"
UNET_OV_PATH=f"{ir_model_dir}/unet.xml"
VAE_DECODER_OV_PATH=f"{ir_model_dir}/vae_decoder.xml"

LORA_DICT = {f"{lora_model_dir}/hipoly3DModelLora_v20.safetensors":0.4, 
             f"{lora_model_dir}/soulcard.safetensors":0.5}

LORA_DICT = None
img_path = f"{dataset_dir}/bird.png"
image = load_image(img_path)
image = np.array(image)
low_threshold = 150
high_threshold = 200
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
print("=====image.shape======",image.shape)
pi_image = Image.fromarray(image)
pi_image.save(f"{output_dir}/org_bird_canny.png")

scheduler = UniPCMultistepScheduler.from_config(scheduler_dir)
# print("==scheduler==",scheduler)

ov_pipe = OVStableDiffusionPipeline(tokenizer, scheduler, core, pi_image,
                                    TEXT_ENCODER_OV_PATH, 
                                    UNET_OV_PATH, VAE_DECODER_OV_PATH,
                                    LORA_DICT, device=device)
import time
t0 = time.time()
# generator = torch.manual_seed(42)
generator = None

prompt = ["bird"]
negative_prompt = ["EasyNegative, [ :(negative_hand-neg: 1.2): 15 ], (worst quality, low quality: 1.4), nsfw"]

image_result = ov_pipe(pi_image, prompt, negative_prompt,
                num_images_per_prompt=1,
                generator=generator,
                num_inference_steps=20)

t1 = time.time()
pipeline_infer_time = (t1-t0)
print(f'============pipeline_infer_time============ enhance time: {pipeline_infer_time} s')
for i in range(len(image_result)):
    filename = f'{output_dir}/ov_bird_out{i}.jpg'
    im1 = image_result[i].save(filename)
    print(filename)
print("==Save success==")