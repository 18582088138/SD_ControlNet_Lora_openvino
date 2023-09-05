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
from tools.lora_toolkit import LoRAToolkit
from tools.utils import scale_fit_to_window, randn_tensor_np, randn_tensor_torch
from ov_static.ov_sd_static_pipeline import OVStableDiffusionContrlNetPipeline

hf_model_dir = "./models/HF_models"
ir_model_dir = "./models/IR_models/FP32_dynamic"
ir_int8_model_dir = "./models/IR_models/FP16_INT8_dynamic"
lora_model_dir = "./models/Lora"
scheduler_dir = "./scheduler"
dataset_dir = "./datasets"
output_dir = "./results"
device = "CPU"

core = Core()
# core.set_property({'CACHE_DIR': './cache'})
tokenizer = CLIPTokenizer.from_pretrained(f'{hf_model_dir}/clip-vit-large-patch14')

CONTROLNET_OV_PATH=f"{ir_model_dir}/controlnet-canny.xml"
TEXT_ENCODER_OV_PATH=f"{ir_model_dir}/text_encoder.xml"
UNET_CONTROL_OV_PATH=f"{ir_model_dir}/unet-controlnet.xml"
UNET_OV_PATH=f"{ir_model_dir}/unet.xml"
VAE_DECODER_OV_PATH=f"{ir_model_dir}/vae_decoder.xml"

# ir_int8_model_dir = "./models/IR_models/FP16_INT8_static"
# CONTROLNET_OV_PATH=f"{ir_int8_model_dir}/controlnet-canny-int8.xml"
# UNET_CONTROL_OV_PATH=f"{ir_int8_model_dir}/unet_controlnet_int8.xml"

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

ov_pipe = OVStableDiffusionContrlNetPipeline(tokenizer, scheduler, core, pi_image,
                                            CONTROLNET_OV_PATH, TEXT_ENCODER_OV_PATH, 
                                            UNET_CONTROL_OV_PATH, VAE_DECODER_OV_PATH,
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
    filename = f'{output_dir}/ov_bird_canny_out{i}.jpg'
    im1 = image_result[i].save(filename)
    print(filename)
print("==Save success==")