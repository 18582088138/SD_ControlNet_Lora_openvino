from pathlib import Path
import torch
import os
import argparse
from torch.onnx import _export as torch_onnx_export
from openvino.tools.mo import convert_model
from openvino.runtime import serialize
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline

hf_model_dir = "../models/HF_models"
onnx_model_dir = "../models/onnx_models/dynamic"
ir_model_dir = "../models/IR_models/FP32_dynamic"
if not os.path.exists(onnx_model_dir):
    os.makedirs(onnx_model_dir)

if not os.path.exists(ir_model_dir):
    os.makedirs(ir_model_dir)

def parse_args() -> argparse.Namespace:
    pass

# ==== covnert controlnet to IR ==== 
controlnet = ControlNetModel.from_pretrained(f"{hf_model_dir}/sd-controlnet-canny", torch_dtype=torch.float32).cpu()

inputs = {
    "sample": torch.randn((2, 4, 105, 70)),
    "timestep": torch.tensor(1),
    "encoder_hidden_states": torch.randn((2,77,768)),
    "controlnet_cond": torch.randn((2,3,840,560)),
    "conditioning_scale":torch.tensor(1),
}

CONTROLNET_ONNX_PATH = Path(f'{onnx_model_dir}/controlnet-canny.onnx')
CONTROLNET_OV_PATH = Path(f'{ir_model_dir}/controlnet-canny.xml')
controlnet.eval()
with torch.no_grad():
    down_block_res_samples, mid_block_res_sample = controlnet(**inputs, return_dict=False)
controlnet_output_names = [f"down_block_res_sample_{i}" for i in range(len(down_block_res_samples))]
controlnet_output_names.append("mid_block_res_sample")
controlnet_dynamic_names = {
    "sample": {0: "batch", 2: "height", 3: "width"},
    "encoder_hidden_states": {0: "batch"},
    "controlnet_cond": {0: "batch", 2: "height", 3: "width"},
    "mid_block_res_sample":{0: "batch", 2: "height", 3: "width"},
}

for i in range(len(down_block_res_samples)):
    controlnet_dynamic_names[f"down_block_res_sample_{i}"] = {0: "batch", 2: "height", 3: "width"}

if not CONTROLNET_OV_PATH.exists():
    if not CONTROLNET_ONNX_PATH.exists():
        with torch.no_grad():
            torch_onnx_export(controlnet, 
            inputs, 
            CONTROLNET_ONNX_PATH, 
            input_names=list(inputs), 
            output_names=controlnet_output_names,
            onnx_shape_inference=False,
            dynamic_axes=controlnet_dynamic_names,
            do_constant_folding=False,
            # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            )
    ov_ctrlnet = convert_model(CONTROLNET_ONNX_PATH,compress_to_fp16=False)
    serialize(ov_ctrlnet,CONTROLNET_OV_PATH)
    del ov_ctrlnet
    print('ControlNet successfully converted to IR')
else:
    print(f"ControlNet will be loaded from {CONTROLNET_OV_PATH}")


#  ==== convert SD-Unet with controlnet model to IR ====
pipe = StableDiffusionControlNetPipeline.from_pretrained(f"{hf_model_dir}/stable-diffusion-v1-5", controlnet=controlnet)
UNET_CONTROL_ONNX_PATH = Path(f'{onnx_model_dir}/unet-controlnet/unet-controlnet.onnx')
UNET_CONTROL_OV_PATH = Path(f'{ir_model_dir}/unet-controlnet.xml')

if not UNET_CONTROL_OV_PATH.exists():
    if not UNET_CONTROL_ONNX_PATH.exists():
        UNET_CONTROL_ONNX_PATH.parent.mkdir(exist_ok=True)
        unet_input = inputs.copy()
        unet_input.pop("controlnet_cond", None)
        unet_input.pop("conditioning_scale", None)
        unet_input["down_block_additional_residuals"] = down_block_res_samples
        unet_input["mid_block_additional_residual"] = mid_block_res_sample
        unet = pipe.unet
        unet.eval()

        input_names = ["latent_model_input", "timestep", "encoder_hidden_states", *controlnet_output_names]
        dynamic_names = {
            "latent_model_input": {0: "batch", 2: "height", 3: "width"},
            "encoder_hidden_states": {0: "batch"},
            "mid_block_res_sample":{0: "batch", 2: "height", 3: "width"},
        }
        for i in range(len(down_block_res_samples)):
            dynamic_names[f"down_block_res_sample_{i}"] = {0: "batch", 2: "height", 3: "width"}

        with torch.no_grad():
            torch_onnx_export(unet, unet_input, 
                        str(UNET_CONTROL_ONNX_PATH), 
                        input_names=input_names, 
                        dynamic_axes=dynamic_names,
                        output_names=["sample_out"],
                        do_constant_folding=False, 
                        onnx_shape_inference=False)
        del unet
    del pipe.unet
    ov_unet = convert_model(UNET_CONTROL_ONNX_PATH, compress_to_fp16=False)
    serialize(ov_unet,UNET_CONTROL_OV_PATH)
    del ov_unet
    print('Unet with controlnet successfully converted to IR')
else:
    del pipe.unet
    print(f"Unet with controlnet will be loaded from {UNET_CONTROL_OV_PATH}")


#  ==== convert SD-Unet without controlnet model to IR ====
pipe = StableDiffusionPipeline.from_pretrained(f"{hf_model_dir}/stable-diffusion-v1-5")
UNET_ONNX_PATH = Path(f'{onnx_model_dir}/unet/unet.onnx')
UNET_OV_PATH = Path(f'{ir_model_dir}/unet.xml')

if not UNET_OV_PATH.exists():
    if not UNET_ONNX_PATH.exists():
        UNET_ONNX_PATH.parent.mkdir(exist_ok=True)
        unet = pipe.unet
        unet.eval()

        input_names = ["latent_model_input", "timestep", "encoder_hidden_states"]
        dynamic_names = {
            "latent_model_input": {0: "batch", 2: "height", 3: "width"},
            "encoder_hidden_states": {0: "batch"},
        }

        with torch.no_grad():
            torch_onnx_export(unet, inputs, 
                        str(UNET_ONNX_PATH), 
                        input_names=input_names, 
                        dynamic_axes=dynamic_names,
                        output_names=["sample_out"],
                        do_constant_folding=False, 
                        onnx_shape_inference=False)
        del unet
    del pipe.unet
    ov_unet = convert_model(UNET_ONNX_PATH, compress_to_fp16=True)
    serialize(ov_unet,UNET_OV_PATH)
    del ov_unet
    print('Unet successfully converted to IR')
else:
    del pipe.unet
    print(f"Unet will be loaded from {UNET_OV_PATH}")



# ==== convert SD-text_encoder model to IR ====
TEXT_ENCODER_ONNX_PATH = Path(f'{onnx_model_dir}/text_encoder.onnx')
TEXT_ENCODER_OV_PATH = Path(f'{ir_model_dir}/text_encoder.xml')

def convert_text_encoder_onnx(text_encoder:torch.nn.Module, onnx_path:Path):
    if not onnx_path.exists():
        input_ids = torch.ones((1, 77), dtype=torch.long)
        dynamic_names = {
            "tokens": {0: "batch", 1: "channls"},
        }
        # switch model to inference mode
        text_encoder.eval()
        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            # infer model, just to make sure that it works
            text_encoder(input_ids)
            # export model to ONNX format
            torch_onnx_export(
                text_encoder,  # model instance
                input_ids,  # inputs for model tracing
                onnx_path,  # output file for saving result
                dynamic_axes=dynamic_names,
                input_names=['tokens'],  # model input name for onnx representation
                output_names=['last_hidden_state', 'pooler_out'],  # model output names for onnx representation
                opset_version=14,  # onnx opset version for export
                onnx_shape_inference=False,
                do_constant_folding=False,
            )
        print('Text Encoder successfully converted to ONNX')

if not TEXT_ENCODER_OV_PATH.exists():
    convert_text_encoder_onnx(pipe.text_encoder, TEXT_ENCODER_ONNX_PATH)
    ov_txten = convert_model(TEXT_ENCODER_ONNX_PATH, compress_to_fp16=True)
    serialize(ov_txten,TEXT_ENCODER_OV_PATH)
    print('Text Encoder successfully converted to IR')
else:
    print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")

###convert VAE model to IR
VAE_DECODER_ONNX_PATH = Path(f'{onnx_model_dir}/vae_decoder.onnx')
VAE_DECODER_OV_PATH = Path(f'{ir_model_dir}/vae_decoder.xml')

def convert_vae_decoder_onnx(vae: torch.nn.Module, onnx_path: Path):
    """
    Convert VAE model to ONNX, then IR format. 
    Function accepts pipeline, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        vae (torch.nn.Module): VAE model
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    if not onnx_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latents = torch.zeros((1, 4, 64, 64))
        dynamic_names = {"latents": {0: "batch", 2: "height", 3: "width"}}
        vae_decoder.eval()
        with torch.no_grad():
            torch.onnx.export(vae_decoder, latents, onnx_path, 
                input_names=['latents'], 
                dynamic_axes=dynamic_names,
                output_names=['sample'],
                do_constant_folding=False,
                )
        print('VAE decoder successfully converted to ONNX')


if not VAE_DECODER_OV_PATH.exists():
    convert_vae_decoder_onnx(pipe.vae, VAE_DECODER_ONNX_PATH)
    ov_vae = convert_model(VAE_DECODER_ONNX_PATH, compress_to_fp16=True)
    serialize(ov_vae,VAE_DECODER_OV_PATH)
    print('VAE decoder successfully converted to IR')
else:
    print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")
