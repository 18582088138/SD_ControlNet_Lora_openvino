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
from lora_toolkit import LoRAToolkit
from utils import scale_fit_to_window, randn_tensor_np, randn_tensor_torch
from ov_sd_pipeline import OVStableDiffusionContrlNetPipeline, OVStableDiffusionPipeline

core = Core()
core.set_property({'CACHE_DIR': './cache'})
tokenizer = CLIPTokenizer.from_pretrained('clip-vit-large-patch14')

model_save_dir = "models_dynamic_shape/"
CONTROLNET_OV_PATH=f"{model_save_dir}/controlnet-canny.xml"
TEXT_ENCODER_OV_PATH=f"{model_save_dir}/text_encoder.xml"
UNET_CONTROL_OV_PATH=f"{model_save_dir}/unet-controlnet.xml"
UNET_OV_PATH=f"{model_save_dir}/unet.xml"
VAE_DECODER_OV_PATH=f"{model_save_dir}/vae_decoder.xml"

# LORA_DICT = {"./Lora/soulcard.safetensors":0.6}
LORA_DICT = {"./Lora/hipoly3DModelLora_v20.safetensors":0.4, "./Lora/soulcard.safetensors":0.6}
# LORA_DICT = {"./Lora/hipoly3DModelLora_v20.safetensors":0.3}
# LORA_DICT = None
# img_path = "vermeer_512x512.png"
img_path = "sd-controlnet-canny/images/bird.png"
# image = load_image("pose.png")
image = load_image(img_path)
image = np.array(image)
low_threshold = 150
high_threshold = 200
# image = cv2.resize(image,(401,791))
# image = cv2.resize(image,(512,512))
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
print("=====image.shape======",image.shape)
pi_image = Image.fromarray(image)
pi_image.save("org_bird_canny.png")

controlnet = ControlNetModel.from_pretrained("sd-controlnet-canny", torch_dtype=torch.float32)
# scheduler = UniPCMultistepScheduler.from_config("scheduler")
scheduler = UniPCMultistepScheduler.from_config("stable-diffusion-v1-5/scheduler")
# print("==scheduler==",scheduler)

ov_pipe = OVStableDiffusionContrlNetPipeline(tokenizer, scheduler, core, pi_image,
                                            CONTROLNET_OV_PATH, TEXT_ENCODER_OV_PATH, 
                                            UNET_CONTROL_OV_PATH, VAE_DECODER_OV_PATH,
                                            LORA_DICT, device="AUTO")

# ov_pipe = OVStableDiffusionPipeline(tokenizer, scheduler, core, pi_image,
#                                     TEXT_ENCODER_OV_PATH, 
#                                     UNET_OV_PATH, VAE_DECODER_OV_PATH,
#                                     LORA_DICT, device="AUTO")

import time
t0 = time.time()
generator = None
# generator = torch.manual_seed(42)
prompt = ["bird"]
# prompt = ["1girl, handsome face, absurdres, highres"]
# prompt = ["absurdres, highres, soul card, line, 1girl, handsome face"]
negative_prompt = ["EasyNegative, [ :(negative_hand-neg: 1.2): 15 ], (worst quality, low quality: 1.4), nsfw"]
# prompt = ["bird"]
image_result = ov_pipe(pi_image, prompt, negative_prompt,
                num_images_per_prompt=2,
                generator=generator,
                num_inference_steps=20)

t1 = time.time()
pipeline_infer_time = (t1-t0)
print(f'============pipeline_infer_time============ enhance time: {pipeline_infer_time} s')

print(len(image_result))
for i in range(len(image_result)):
    # filename = f'./ov_vermeer_canny_out{i}.jpg'
    filename = f'./ov_bird_canny_out{i}.jpg'
    im1 = image_result[i].save(filename)
    print(filename)
print("==Save success==")