import torch
from torch.utils.data import Dataset
import torchvision

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import CLIPTokenizer

from openvino.runtime import serialize
import openvino.runtime as ov
import nncf

import argparse
import os
import sys
sys.path.append("../")
import json
import ast
from tqdm import trange
import numpy as np
from PIL import Image, ImageFile
from typing import Union, List, Optional, Tuple

from tools.utils import scale_fit_to_window, randn_tensor_np, randn_tensor_torch


core = ov.Core()


class SelfControlNetDataset(Dataset):
    def __init__(self, dataset_path, dataset_json, text_encoder, scheduler, tokenizer, subset_size=100):
        # self.text_encoder = core.compile_model(text_encoder, "CPU")
        self.text_encoder = text_encoder
        self.text_encoder_out = self.text_encoder.output(0)

        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.vae_scale_factor = 8
        self.do_classifier_free_guidance = True

        self.data_list = []
        self.dataset_path = dataset_path
        index = 0
        with open(dataset_json, 'r') as dataset_file:
            for dataset_line in dataset_file.readlines():
                dataset_data = ast.literal_eval(dataset_line.strip())
                latent_model_input, t, text_embeddings, image = self.prepare_dataset(dataset_data,index)
                index +=1
                self.data_list.append((latent_model_input, t, text_embeddings, image))
                if index >= subset_size:
                    break
        print(len(self.data_list))

    def prepare_dataset(self, dataset_data, index):
        batch_size = 1 
        num_images_per_prompt = 1
        num_channels_latents = 4
        t = index

        src_img_path = os.path.join(self.dataset_path, dataset_data['source'])
        target_img_path = os.path.join(self.dataset_path, dataset_data['target'])
        src_prompt = dataset_data['prompt']

        image = Image.open(src_img_path)
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = self._preprocess_img(image)
        height, width = image.shape[-2:]
        if self.do_classifier_free_guidance:
            image = np.concatenate(([image] * 2))
        text_embeddings = self._encode_prompt(src_prompt)

        
        latents = self._prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
        )

        latent_model_input = np.concatenate(
            [latents] * 2) if self.do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        return latent_model_input, t, text_embeddings, image

    def __len__(self):
        return len(self.data_list)

    def _preprocess_img(self, image: Image.Image):
        width, height = image.size
        width, height = (x - x % self.vae_scale_factor for x in (width, height))  # resize to integer multiple of vae_scale_factor
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
        image = np.array(image)[None, :]
        image = image.astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        return image
    
    def _encode_prompt(self, prompt:Union[str, List[str]], 
                        num_images_per_prompt:int = 1, 
                        do_classifier_free_guidance:bool = True, 
                        negative_prompt:Union[str, List[str]] = None,
                        prompt_embeds: Optional[torch.FloatTensor] = None,
                        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                        lora_scale: Optional[float] = None,
                        ):
        # batch_size = len(prompt) if isinstance(prompt, list) else 1
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        # tokenize input prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids

        text_embeddings = self.text_encoder(text_input_ids)[self.text_encoder_out]

        # get unconditional embeddings for classifier free guidance
        if self.do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self.text_encoder_out]
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings

    def _prepare_latents(self, batch_size:int, num_channels_latents:int, height:int, width:int, 
                        dtype:np.dtype = np.float32, generator=None, latents:np.ndarray = None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            if generator is None:
                latents = randn_tensor_np(shape, dtype=dtype)
            else:
                dtype = torch.float32
                latents = randn_tensor_torch(shape, generator=generator, dtype=dtype)
        else:
            latents = latents
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def __getitem__(self, index):
        data_item = self.data_list[index]
        inputs = {
            "sample": data_item[0],
            "timestep": data_item[1],
            "encoder_hidden_states": data_item[2],
            "controlnet_cond": data_item[3],
            "conditioning_scale": np.arange(1),
        }
        # print("==sample==",data_item[0].shape)

        # inputs = {
        #     "sample": torch.randn((1, 4, 105, 70)),
        #     "timestep": torch.tensor(1),
        #     "encoder_hidden_states": torch.randn((1,77,768)),
        #     "controlnet_cond": torch.randn((1,3,840,560)),
        #     "conditioning_scale":torch.tensor(1),
        # }

        return data_item

def transform_fn(data_item):
    inputs = {
            "sample": data_item[0].squeeze(0),
            "timestep": data_item[1].squeeze(0),
            "encoder_hidden_states": data_item[2].squeeze(0),
            "controlnet_cond": data_item[3].squeeze(0),
            "conditioning_scale": data_item[1].squeeze(0),
        }

    return inputs

scheduler_dir = "../scheduler"
hf_model_dir = "../models/HF_models"
onnx_model_dir = "../models/onnx_models/static"
ir_model_dir = "../models/IR_models/FP32_static"
ir_int8_model_dir = "../models/IR_models/FP16_INT8_static"

controlnet_fp32_ir_path = f"{ir_model_dir}/controlnet-canny.xml"
controlnet_int8_ir_path = f"{ir_int8_model_dir}/controlnet-canny-int8.xml"

text_encoder = core.compile_model(f"{ir_model_dir}/text_encoder.xml", "CPU")
dataset_path = "../datasets/fill50k/"
dataset_json = os.path.join(dataset_path, 'prompt.json')

scheduler = UniPCMultistepScheduler.from_config(scheduler_dir)
tokenizer = CLIPTokenizer.from_pretrained(f'{hf_model_dir}/clip-vit-large-patch14')

controlnet_dataset = SelfControlNetDataset(dataset_path, dataset_json, text_encoder, scheduler, tokenizer)

controlnet_dataloader = torch.utils.data.DataLoader(controlnet_dataset, batch_size=1,shuffle=False)
# controlnet_dataloader = torch.utils.data.DataLoader(controlnet_dataset, batch_size=1, shuffle=False)

calibration_dataset = nncf.Dataset(controlnet_dataloader, transform_fn)

controlnet_ov_model = core.read_model(controlnet_fp32_ir_path)
ov_quantized_model = nncf.quantize(controlnet_ov_model, calibration_dataset,
                     subset_size=100,
                     preset=nncf.QuantizationPreset.MIXED,)
serialize(ov_quantized_model, xml_path=controlnet_int8_ir_path,
                           bin_path=controlnet_int8_ir_path.replace("xml","bin"))
print("== Compression Model Success ==")