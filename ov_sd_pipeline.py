from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
import time

from diffusers.pipeline_utils import DiffusionPipeline
from transformers import CLIPTokenizer
from typing import Union, List, Optional, Tuple
import cv2
from PIL import Image
from openvino.runtime import Model, Core
from lora_toolkit import LoRAToolkit
from utils import scale_fit_to_window, randn_tensor_np, randn_tensor_torch

class OVStableDiffusionContrlNetPipeline(DiffusionPipeline):
    """
    OpenVINO inference pipeline for Stable Diffusion with ControlNet guidence
    """
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        scheduler,
        core: Core,
        image: Image,
        controlnet: Model,
        text_encoder: Model,
        unet: Model,
        vae_decoder: Model,
        lora_dict: dict=None,
        device:str = "AUTO"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vae_scale_factor = 8
        self.scheduler = scheduler
        self.lora_toolkit = LoRAToolkit()
        self.load_models(core, device,image, controlnet, text_encoder, unet, vae_decoder, lora_dict)
        self.set_progress_bar_config(disable=True)
    
    def load_models(self, core: Core, device: str,image: Image, controlnet:Model, 
                    text_encoder: Model, unet: Model, vae_decoder: Model, lora_dict: dict=None):
        """
        Function for loading models on device using OpenVINO
        
        Parameters:
          core (Core): OpenVINO runtime Core class instance
          device (str): inference device
          controlnet (Model): OpenVINO Model object represents ControlNet
          text_encoder (Model): OpenVINO Model object represents text encoder
          unet (Model): OpenVINO Model object represents UNet
          vae_decoder (Model): OpenVINO Model object represents vae decoder
        Returns
          None
        """
        text_encoder = core.read_model(text_encoder)
        unet = core.read_model(unet)
        total_time_start = time.time()
        lora_time_start = time.time()
        if lora_dict is not None:
            lora_len = len(lora_dict)
            for lora, loar_alpha in lora_dict.items():
                lora_parser_time_start = time.time()
                lora_dict_list = self.lora_toolkit.lora_parser(lora, loar_alpha/lora_len)
                lora_parser_time_end = time.time()
                print("==lora_parser_time==",lora_parser_time_end-lora_parser_time_start)
                print("==lora_dict_list==",len(lora_dict_list[0]))
                if (True in [('type','text_encoder') in l.items() for l in lora_dict_list]):
                    text_encoder = self.lora_toolkit.lora_text_encoder_insert(text_encoder)
                    
                    # print("==lora_text_encoder==",text_encoder)
                if (True in [('type','unet') in l.items() for l in lora_dict_list]):
                    unet = self.lora_toolkit.lora_unet_insert(unet)
                    
                # if (True in [('type','text_encoder') in l.items() for l in lora_dict_list]):
                #     text_encoder = self.lora_toolkit.lora_text_encoder_insert(text_encoder)
                #     # print("==lora_text_encoder==",text_encoder)
                # if (True in [('type','unet') in l.items() for l in lora_dict_list]):
                #     unet = self.lora_toolkit.lora_unet_insert(unet)
                # print("==lora_unet==",unet)
        lora_time_end = time.time()
        text_encoder_time_start = time.time()
        self.text_encoder = core.compile_model(text_encoder, device)
        text_encoder_time_end = time.time()
        unet_time_start = time.time()
        self.unet = core.compile_model(unet, device)
        unet_time_end = time.time()

        vae_time_start = time.time()
        self.vae_decoder = core.compile_model(vae_decoder)
        vae_time_end = time.time()

        controlnet_time_start = time.time()
        self.controlnet = core.compile_model(controlnet, device)
        controlnet_time_end = time.time()

        total_time_end = time.time()
        print("==lora_time==",lora_time_end-lora_time_start)
        print("==text_encoder_time==",text_encoder_time_end-text_encoder_time_start)
        print("==unet_time==",unet_time_end-unet_time_start)
        print("==vae_time==",vae_time_end-vae_time_start)
        print("==controlnet_time==",controlnet_time_end-controlnet_time_start)
        print("==total_time==",total_time_end-total_time_start)
        
        self.text_encoder_out = self.text_encoder.output(0)
        self.unet_out = self.unet.output(0)
        self.vae_decoder_out = self.vae_decoder.output(0)

    def preprocess(self, image: Image.Image, batch_size, num_images_per_prompt):
        """
        Parameters:
        image (Image.Image): input image
        Returns:
        image (np.ndarray): preprocessed image tensor
        """
        width, height = image.size
        width, height = (x - x % self.vae_scale_factor for x in (width, height))  # resize to integer multiple of vae_scale_factor
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)

        image = np.array(image)[None, :]
        image = image.astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        
        image_batch_size = image.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt
        image = np.repeat(image, repeats=repeat_by, axis=0)
        print("==== image.shape ==== : ", repeat_by, image.shape)
        return image

    def _encode_prompt(self, prompt:Union[str, List[str]], 
                        num_images_per_prompt:int = 1, 
                        do_classifier_free_guidance:bool = True, 
                        negative_prompt:Union[str, List[str]] = None,
                        prompt_embeds: Optional[torch.FloatTensor] = None,
                        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                        lora_scale: Optional[float] = None,
                        ):
        """
        Encodes the prompt into text encoder hidden states.

        Parameters:
            prompt (str or list(str)): prompt to be encoded
            num_images_per_prompt (int): number of images that should be generated per prompt
            do_classifier_free_guidance (bool): whether to use classifier free guidance or not
            negative_prompt (str or list(str)): negative prompt to be encoded
        Returns:
            text_embeddings (np.ndarray): text encoder hidden states
        """
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

        # duplicate text embeddings for each generation per prompt
        if num_images_per_prompt != 1:
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = np.tile(text_embeddings, (1, num_images_per_prompt, 1))
            text_embeddings = np.reshape(text_embeddings, (bs_embed * num_images_per_prompt, seq_len, -1))

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
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

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings
    
    def prepare_latents(self, batch_size:int, num_channels_latents:int, height:int, width:int, 
                        dtype:np.dtype = np.float32, generator=None, latents:np.ndarray = None):
        """
        Preparing noise to image generation. If initial latents are not provided, they will be generated randomly, 
        then prepared latents scaled by the standard deviation required by the scheduler
        
        Parameters:
           batch_size (int): input batch size
           num_channels_latents (int): number of channels for noise generation
           height (int): image height
           width (int): image width
           dtype (np.dtype, *optional*, np.float32): dtype for latents generation
           latents (np.ndarray, *optional*, None): initial latent noise tensor, if not provided will be generated
        Returns:
           latents (np.ndarray): scaled initial noise for diffusion
        """
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

    def decode_latents(self, latents:np.array):
        """
        Decode predicted image from latent space using VAE Decoder
        Parameters:
           latents (np.ndarray): image encoded in diffusion latent space
        Returns:
           image: decoded by VAE decoder image
        """
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder(latents)[self.vae_decoder_out]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        return image

    def __call__(
        self,
        image: Image.Image,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 10,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: Union[float, List[float]] = [0.0], #single controlnet
        control_guidance_end: Union[float, List[float]] = [1.0], #single controlnet
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[np.array] = None,
        output_type: Optional[str] = "pil",
    ):
        """
        Function invoked when calling the pipeline for generation.

        Parameters:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`Image.Image`):
                `Image`, or tensor representing an image batch which will be repainted according to `prompt`.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            negative_prompt (`str` or `List[str]`):
                negative prompt or prompts for generation
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality. This pipeline requires a value of at least `1`.
            latents (`np.ndarray`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `Image.Image` or `np.array`.
        Returns:
            image ([List[Union[np.ndarray, Image.Image]]): generaited images
            
        """

        # 1. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        #print("batch:",batch_size)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # 2. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, num_images_per_prompt=num_images_per_prompt, negative_prompt=negative_prompt)

        # 3. Preprocess image
        orig_width, orig_height = image.size
        image = self.preprocess(image, batch_size * num_images_per_prompt, num_images_per_prompt)
        height, width = image.shape[-2:]
        if do_classifier_free_guidance:
            image = np.concatenate(([image] * 2))

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            generator,
            latents,
        )

         # 6.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0]) #keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.controlnet_pip
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = np.concatenate(
                    [latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                #text_embeddings = np.split(text_embeddings, 2)[1] if do_classifier_free_guidance else text_embeddings
                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    cond_scale = controlnet_conditioning_scale * controlnet_keep[i]
                
                result = self.controlnet([latent_model_input, t, text_embeddings, image, cond_scale])
                down_and_mid_blok_samples = [sample * cond_scale for _, sample in result.items()]

                # predict the noise residual
                # noise_pred = self.unet([latent_model_input, t, text_embeddings])[self.unet_out]
                noise_pred = self.unet([latent_model_input, t, text_embeddings, *down_and_mid_blok_samples])[self.unet_out]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred,2) #noise_pred[0], noise_pred[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents)).prev_sample.numpy()

                # update progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            image = [img.resize((orig_width, orig_height), Image.Resampling.LANCZOS) for img in image]
        else:
            image = [cv2.resize(img, (orig_width, orig_width))
                     for img in image]
        return image


class OVStableDiffusionPipeline(DiffusionPipeline):
    """
    OpenVINO inference pipeline for Stable Diffusion with ControlNet guidence
    """
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        scheduler,
        core: Core,
        image: Image,
        text_encoder: Model,
        unet: Model,
        vae_decoder: Model,
        lora_dict: dict=None,
        lora: Model=None,
        loar_alpha: float=0.5,
        device:str = "AUTO"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vae_scale_factor = 8
        self.scheduler = scheduler
        self.lora_toolkit = LoRAToolkit()
        self.load_models(core, device,image, text_encoder, unet, vae_decoder, lora_dict)
        self.set_progress_bar_config(disable=True)
    
    def load_models(self, core: Core, device: str,image: Image, 
                    text_encoder: Model, unet: Model, vae_decoder: Model, lora_dict: dict=None):
        """
        Function for loading models on device using OpenVINO
        
        Parameters:
          core (Core): OpenVINO runtime Core class instance
          device (str): inference device
          text_encoder (Model): OpenVINO Model object represents text encoder
          unet (Model): OpenVINO Model object represents UNet
          vae_decoder (Model): OpenVINO Model object represents vae decoder
        Returns
          None
        """
        text_encoder = core.read_model(text_encoder)
        unet = core.read_model(unet)

        total_time_start = time.time()
        lora_time_start = time.time()
        time.sleep(1)
        if lora_dict is not None:
            lora_len = len(lora_dict)
            for lora, loar_alpha in lora_dict.items():
                lora_dict_list = self.lora_toolkit.lora_parser(lora, loar_alpha/lora_len)
                if (True in [('type','text_encoder') in l.items() for l in lora_dict_list]):
                    text_encoder_insert_time_start = time.time()
                    text_encoder = self.lora_toolkit.lora_text_encoder_insert(text_encoder)
                    text_encoder_insert_time_end = time.time()
                    print("==text_encoder_insert_time==",text_encoder_insert_time_end-text_encoder_insert_time_start)
                    # print("==lora_text_encoder==",text_encoder)
                if (True in [('type','unet') in l.items() for l in lora_dict_list]):
                    unet_insert_time_start = time.time()
                    unet = self.lora_toolkit.lora_unet_insert(unet)
                    unet_insert_time_end = time.time()
                    print("==unet_insert_time==",unet_insert_time_end-unet_insert_time_start)
                # print("==lora_unet==",unet)
        lora_time_end = time.time()
        text_encoder_time_start = time.time()
        self.text_encoder = core.compile_model(text_encoder, device)
        text_encoder_time_end = time.time()
        unet_time_start = time.time()
        self.unet = core.compile_model(unet, device)
        unet_time_end = time.time()

        vae_time_start = time.time()
        self.vae_decoder = core.compile_model(vae_decoder)
        vae_time_end = time.time()

        total_time_end = time.time()
        print("==lora_time==",lora_time_end-lora_time_start)
        print("==text_encoder_time==",text_encoder_time_end-text_encoder_time_start)
        print("==unet_time==",unet_time_end-unet_time_start)
        print("==vae_time==",vae_time_end-vae_time_start)
        print("==total_time==",total_time_end-total_time_start)

        

        self.text_encoder_out = self.text_encoder.output(0)
        self.unet_out = self.unet.output(0)
        self.vae_decoder_out = self.vae_decoder.output(0)

    def preprocess(self, image: Image.Image, batch_size, num_images_per_prompt):
        """
        Parameters:
        image (Image.Image): input image
        Returns:
        image (np.ndarray): preprocessed image tensor
        """
        width, height = image.size
        width, height = (x - x % self.vae_scale_factor for x in (width, height))  # resize to integer multiple of vae_scale_factor
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)

        image = np.array(image)[None, :]
        image = image.astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        
        image_batch_size = image.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt
        image = np.repeat(image, repeats=repeat_by, axis=0)
        print("==== image.shape ==== : ", repeat_by, image.shape)
        return image

    def _encode_prompt(self, prompt:Union[str, List[str]], 
                        num_images_per_prompt:int = 1, 
                        do_classifier_free_guidance:bool = True, 
                        negative_prompt:Union[str, List[str]] = None,
                        prompt_embeds: Optional[torch.FloatTensor] = None,
                        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                        lora_scale: Optional[float] = None,
                        ):
        """
        Encodes the prompt into text encoder hidden states.

        Parameters:
            prompt (str or list(str)): prompt to be encoded
            num_images_per_prompt (int): number of images that should be generated per prompt
            do_classifier_free_guidance (bool): whether to use classifier free guidance or not
            negative_prompt (str or list(str)): negative prompt to be encoded
        Returns:
            text_embeddings (np.ndarray): text encoder hidden states
        """
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

        # duplicate text embeddings for each generation per prompt
        if num_images_per_prompt != 1:
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = np.tile(text_embeddings, (1, num_images_per_prompt, 1))
            text_embeddings = np.reshape(text_embeddings, (bs_embed * num_images_per_prompt, seq_len, -1))

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
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

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings
    
    def prepare_latents(self, batch_size:int, num_channels_latents:int, height:int, width:int, 
                        dtype:np.dtype = np.float32, generator=None, latents:np.ndarray = None):
        """
        Preparing noise to image generation. If initial latents are not provided, they will be generated randomly, 
        then prepared latents scaled by the standard deviation required by the scheduler
        
        Parameters:
           batch_size (int): input batch size
           num_channels_latents (int): number of channels for noise generation
           height (int): image height
           width (int): image width
           dtype (np.dtype, *optional*, np.float32): dtype for latents generation
           latents (np.ndarray, *optional*, None): initial latent noise tensor, if not provided will be generated
        Returns:
           latents (np.ndarray): scaled initial noise for diffusion
        """
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

    def decode_latents(self, latents:np.array):
        """
        Decode predicted image from latent space using VAE Decoder
        Parameters:
           latents (np.ndarray): image encoded in diffusion latent space
        Returns:
           image: decoded by VAE decoder image
        """
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder(latents)[self.vae_decoder_out]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        return image

    def __call__(
        self,
        image: Image.Image,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 10,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[np.array] = None,
        output_type: Optional[str] = "pil",
    ):
        """
        Function invoked when calling the pipeline for generation.

        Parameters:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`Image.Image`):
                `Image`, or tensor representing an image batch which will be repainted according to `prompt`.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            negative_prompt (`str` or `List[str]`):
                negative prompt or prompts for generation
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality. This pipeline requires a value of at least `1`.
            latents (`np.ndarray`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `Image.Image` or `np.array`.
        Returns:
            image ([List[Union[np.ndarray, Image.Image]]): generaited images
            
        """

        # 1. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        #print("batch:",batch_size)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # 2. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, num_images_per_prompt=num_images_per_prompt, negative_prompt=negative_prompt)

        # 3. Preprocess image
        orig_width, orig_height = image.size
        image = self.preprocess(image, batch_size * num_images_per_prompt, num_images_per_prompt)
        height, width = image.shape[-2:]
        if do_classifier_free_guidance:
            image = np.concatenate(([image] * 2))

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            generator,
            latents,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.controlnet_pip
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = np.concatenate(
                    [latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                # noise_pred = self.unet([latent_model_input, t, text_embeddings])[self.unet_out]
                noise_pred = self.unet([latent_model_input, t, text_embeddings])[self.unet_out]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred,2) #noise_pred[0], noise_pred[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents)).prev_sample.numpy()

                # update progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            image = [img.resize((orig_width, orig_height), Image.Resampling.LANCZOS) for img in image]
        else:
            image = [cv2.resize(img, (orig_width, orig_width))
                     for img in image]
        return image

