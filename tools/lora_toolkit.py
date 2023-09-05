import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import time
import numpy as np
from openvino.runtime import Core, Model, Type
from openvino.runtime.passes import MakeStateful, Manager, GraphRewrite, MatcherPass, WrapType, Matcher
from openvino.runtime import opset10 as ops
from safetensors.torch import load_file
from diffusers.utils import load_image


class LoRAToolkit():
    def __init__(self):
        ##===Initial lora weights===
        self.visited = []
        self.lora_dict = {}
        self.lora_dict_list = []
        self.LORA_PREFIX_UNET = "lora_unet"
        self.LORA_PREFIX_TEXT_ENCODER = "lora_te"

        self.core = Core()
        self.manager = Manager()
        # self.lora_parser()
        # self.manager.register_pass(InsertLoRA(self.lora_dict_list))
    
    def lora_parser(self,lora_model_path, lora_alpha):
        self.state_dict = load_file(lora_model_path)
        self.lora_alpha = lora_alpha
        for key in self.state_dict:
            if ".alpha" in key or key in self.visited:
                continue
            if "text" in key:
                layer_infos = key.split(self.LORA_PREFIX_TEXT_ENCODER + "_")[-1].split(".")[0]
                self.lora_dict = dict(name=layer_infos)
                self.lora_dict.update(type="text_encoder")
            else:
                layer_infos = key.split(self.LORA_PREFIX_UNET + "_")[1].split('.')[0]
                self.lora_dict = dict(name=layer_infos)
                self.lora_dict.update(type="unet")
            pair_keys = []
            if "lora_down" in key:
                pair_keys.append(key.replace("lora_down", "lora_up"))
                pair_keys.append(key)
            else:
                pair_keys.append(key)
                pair_keys.append(key.replace("lora_up", "lora_down"))

                # update weight
            if len(self.state_dict[pair_keys[0]].shape) == 4:
                weight_up = self.state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
                weight_down = self.state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
                # lora_weights = self.lora_alpha * 0
                lora_weights = self.lora_alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                self.lora_dict.update(value=lora_weights)
            else:
                weight_up = self.state_dict[pair_keys[0]].to(torch.float32)
                weight_down = self.state_dict[pair_keys[1]].to(torch.float32)
                # lora_weights = self.lora_alpha * 0
                lora_weights = self.lora_alpha * torch.mm(weight_up, weight_down)
                self.lora_dict.update(value=lora_weights)
            self.lora_dict_list.append(self.lora_dict)
            # update visited list
            for item in pair_keys:
                self.visited.append(item)
        # print("==self.lora_dict_list==",self.lora_dict_list)
        return self.lora_dict_list

    def lora_unet_insert(self,ov_unet):
        unet_insert_time_start = time.time()
        self.manager.register_pass(InsertLoRA(self.lora_dict_list))
        unet_insert_time_end = time.time()
        print("==unet_insert_time==",unet_insert_time_end-unet_insert_time_start)

        unet_run_passes_time_start = time.time()
        self.manager.run_passes(ov_unet)
        unet_run_passes_time_end = time.time()
        print("==unet_run_passes_time==",unet_run_passes_time_end-unet_run_passes_time_start)
        return ov_unet

    def lora_text_encoder_insert(self,ov_text_encoder,device="AUTO"):
        text_encoder_insert_time_start = time.time()
        self.manager.register_pass(InsertLoRA(self.lora_dict_list))
        text_encoder_insert_time_end = time.time()
        print("==text_encoder_insert_time==",text_encoder_insert_time_end-text_encoder_insert_time_start)

        text_encoder_run_passes_time_start = time.time()
        self.manager.run_passes(ov_text_encoder)
        text_encoder_run_passes_time_end = time.time()
        print("==text_encoder_run_passes_time==",text_encoder_run_passes_time_end-text_encoder_run_passes_time_start)
        return ov_text_encoder


class InsertLoRA(MatcherPass):
    def __init__(self,lora_dict_list):
        MatcherPass.__init__(self)
        self.model_changed = False
        param = WrapType("opset10.Convert")
        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            root_output = matcher.get_match_value()
            for y in lora_dict_list:
                if root.get_friendly_name().replace('.','_').replace('_weight','') == y["name"]:
                    consumers = root_output.get_target_inputs()
                    lora_weights = ops.constant(y["value"],Type.f32,name=y["name"])
                    add_lora = ops.add(root,lora_weights,auto_broadcast='numpy')
                    for consumer in consumers:
                        consumer.replace_source_output(add_lora.output(0))
                    # For testing purpose
                    self.model_changed = True
                    # Use new operation for additional matching
                    self.register_new_node(add_lora)
            # Root node wasn't replaced or changed
            return False
        self.register_matcher(Matcher(param,"InsertLoRA"), callback)