# SD_ControlNet_Lora_openvino
OpenVINO enable stable diffusion with controlnet and Lora


## Pre-requisites
1. [Mini Conda](https://docs.conda.io/en/latest/miniconda.html)
2. Ubuntu 20.04, Ubuntu 22.04 or RedHat 8.7 host

### Environment setup 

Create Conda environment

```
conda create -n py39-ov23-SD python==3.9
conda activate py39-ov23-SD
pip install -r requirements.txt
```

### Download models from HuggingFace
Since the download of the model on HuggingFace is relatively slow, it is best to download it in advance and store it in the `models/HF_models` directory

Model requirt list
 - [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
 - [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)
 - [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

Of course you can use the HuggingFace downloader toolkit to automatic download, But sometimes the download is interrupted due to network reasons, and it takes a few more attempts to download all the models successfully.

```
# Such as download the runwayml/stable-diffusion-v1-5 model

python tools/HF_model_downloader.py --model_name runwayml/stable-diffusion-v1-5
```

### OpenVINO model convert

We provide the script to convert the model from ` pytorch (HF) -> onnx -> IR (OpenVINO)` 
And you can select to generates the `dynamic input shape model` or the `static input shape model`

`Dynamic model` can generate pictures of any input size, for example, 512x512, 768x768, 840x560, and support multi-batch pictures generate at a time and the quality of the generated pictures is better, but it needs a lot of memory to run

`Static model` only can generate 512x512 pictures. If you need pictures of other sizes, you need to perform corresponding pre-processing and post-processing operations on the pictures, And the static model generate workload is faster and more friendly to memory overhead
```
# Generate dynamic shape model
python tools/get_models_dynamic_shape.py

# Generate static shape model
python tools/get_models_static_shape.py
```
Please try to convert the model according to your project needs

### OpenVINO enable Stable diffusion pipeline 
In our repo we provide 4 kinds of pipelines to use
 - ov-sd-lora-dynamic-pipeline.py
 - ov-sd-controlnet-canny-lora-dynamic-pipeline.py
 - ov-sd-lora-static-pipeline.py
 - ov-sd-controlnet-canny-lora-static-pipeline.py
```
# Static shape OpenVINO Stable Diffusion pipeline with LoRA 
python ov-sd-lora-static-pipeline.py

# Static shape OpenVINO Stable Diffusion with controlnet-canny pipeline with LoRA 
python ov-sd-controlnet-canny-lora-static-pipeline.py
```
The generation result will save in the `./results` directory

### OpenVINO model compress
There are 2 method to compress OpenVINO IR models, 

One is `FP32->FP16` model compression which is efficient using on Intel GPU, the compression ratio is 1.5~2x times.

One is `FP32/FP16->INT8` model compression which need NNCF tools to quantize the model, both Intel CPU and GPU can be used, the compression ratio is higher, can reach 3~4x times, and the model inference latency is lower.

```
# FP32->FP16 model compress
python tools/convert_fp32tofp16_ir.py -m models/IR_models/FP32_static/unet.xml 

# P32/FP16->INT8 model compress 
./ov_unet_model_compress.sh 
```
In follow compress script is for dynamic shape unet-w-control model compress, If you want to compress other model, pls reference the source code to edit corresponding script.