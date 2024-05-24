"""
inference code:

    input: rgba image, see #628 val_list = ["apple.png" ...]
        located at example/example_padded_rgba_pngs by default

    output: animated rgba videos, webp format

    usage: 
        reference to example/layerdiffuse_stage2_384.yaml, 
        modify transparent_unet_pretrained_model_path, transparent_VAE_pretrained_model_path -> your downloaded model paths

        please download pretrained transparent vae: https://cloudbook-public-daily.oss-cn-hangzhou.aliyuncs.com/animation/transparent_VAE.tar
            , extract this tar, and then place it at transparent_VAE_pretrained_model_path

        please download pretrained transparent unet: https://cloudbook-public-daily.oss-cn-hangzhou.aliyuncs.com/animation/transparent_unet.tar
            , extract this tar, and then place it at transparent_unet_pretrained_model_path

        run "python train_transparent_i2v_stage2.py --config example/layerdiffuse_stage2_384.yaml --eval" 
        results will be saved at "output/stage_2_eval" by default

    Note: This is code in the early stages and may be subject to significant changes. The training code and dataset are not yet open-sourced.
"""

import argparse
import datetime
import logging
import inspect
import math
import os

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers
import numpy as np

from tqdm.auto import tqdm
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers.models import AutoencoderKL
from diffusers import DDPMScheduler, TextToVideoSDPipeline, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock

from transformers import CLIPTextModel, CLIPTokenizer

from einops import rearrange, repeat
import imageio


from models.layerdiffuse_VAE import LatentTransparencyOffsetEncoder, UNet384
from models.unet_3d_condition_mask import UNet3DConditionModel
from models.pipeline_stage2 import MaskedLatentToVideoPipeline, ConcatLatentToVideoPipeline
from utils.common import calculate_motion_precision, calculate_latent_motion_score, \
    DDPM_forward_timesteps

already_printed_trainables = False

logger = get_logger(__name__, log_level="INFO")

def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
    out_dir = os.path.join(output_dir, f"train_{now}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir

def load_primary_models(pretrained_model_path, in_channels=-1, motion_strength=False, xl=False, alpha_chechpoint=None):
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    # load transparent models
    vae_alpha_encoder = LatentTransparencyOffsetEncoder()
    vae_alpha_decoder = UNet384()

    if(alpha_chechpoint):
        vae_alpha_encoder.load_state_dict(
            torch.load(os.path.join(alpha_chechpoint, 'vae_alpha_encoder.pth'))
        )

        vae_alpha_decoder.load_state_dict(
            torch.load(os.path.join(alpha_chechpoint, 'vae_alpha_decoder.pth'))
        )

        print(f'vae_alpha from ckpt {alpha_chechpoint} loaded..')

    return noise_scheduler, tokenizer, text_encoder, vae, unet, vae_alpha_encoder, vae_alpha_decoder

def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    if unet_enable:
        unet.enable_gradient_checkpointing()
    else:
        unet.disable_gradient_checkpointing()

    if text_enable:
        text_encoder.gradient_checkpointing_enable()
    else:
        text_encoder.gradient_checkpointing_disable()
        
def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False) 
            
def is_attn(name):
   return ('attn1' or 'attn2' == name.split('.')[-1])

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0()) 

def set_torch_2_attn(unet):
    optim_count = 0
    
    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0: 
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet): 
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn
        
        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        if enable_torch_2:
            set_torch_2_attn(unet)
            
    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model, 
        "condition": condition, 
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }
    

def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name, 
        "params": params, 
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v
    
    return params

def negate_params(name, negation):
    # We have to do this if we are co-training with LoRA.
    # This ensures that parameter groups aren't duplicated.
    if negation is None: return False
    for n in negation:
        if n in name and 'temp' not in name:
            return True
    return False


def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition and isinstance(model, list): 
            params = create_optim_params(
                params=itertools.chain(*model), 
                extra_params=extra_params
            )
            optimizer_params.append(params)
            continue
            
        if is_lora and  condition and not isinstance(model, list):
            for n, p in model.named_parameters():
                if 'lora' in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                should_negate = 'lora' in n and not is_lora
                if should_negate: continue

                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)
    
    return optimizer_params

def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW

def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype

def cast_to_gpu_and_type(model_list, device, weight_dtype):
    for model in model_list:
        if model is not None: model.to(device, dtype=weight_dtype)


def handle_trainable_modules(model, trainable_modules, not_trainable_modules=[], is_enabled=True, negation=None):
    global already_printed_trainables

    # This can most definitely be refactored :-)
    unfrozen_params = 0
    print(f"not trainable {not_trainable_modules}")
    for name, module in model.named_modules():
        check = False
        for tm in tuple(trainable_modules):
            if tm == 'all' or (tm in name and 'lora' not in name):
                check = True
                break
        for tm in not_trainable_modules:
            if tm in name:
                check = False
                break
        if check:
            for m in module.parameters():
                m.requires_grad_(is_enabled)
                if is_enabled: unfrozen_params +=1

    if unfrozen_params > 0 and not already_printed_trainables:
        already_printed_trainables = True 
        print(f"{unfrozen_params} params have been unfrozen for training.")

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents

def tensor_to_transparent_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae(t)
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)

    return latents

def sample_noise(latents, noise_strength, use_offset_noise=False):
    b ,c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step == 2)  \
    and validation_data.sample_preview


def replace_prompt(prompt, token, wlist):
    for w in wlist:
        if w in prompt: return prompt.replace(w, token)
    return prompt 


def prompt_image(image, processor, encoder):
    if type(image) == str:
        image = Image.open(image)
    image = processor(images=image, return_tensors="pt")['pixel_values']
    
    image = image.to(encoder.device).to(encoder.dtype)
    inputs = encoder(image).pooler_output.to(encoder.dtype).unsqueeze(1)
    #inputs = encoder(image).last_hidden_state.to(encoder.dtype)
    return inputs
    

def eval(pipeline, vae_processor, vae_alpha_encoder, vae_alpha_decoder, validation_data, out_file, index, forward_t=25, preview=True, in_channels=-1):
    vae = pipeline.vae
    device = vae.device
    dtype = vae.dtype

    diffusion_scheduler = pipeline.scheduler
    diffusion_scheduler.set_timesteps(validation_data.num_inference_steps, device=device)

    num_frames = validation_data.num_frames
    prompt = validation_data.prompt

    eval_video = validation_data.prompt_image[-3:] in ['mp4', 'gif']
    if eval_video:
        # default False
        #frames = read_video(validation_data.prompt_image)
        frames = imageio.mimread(validation_data.prompt_image)
        frames = [Image.fromarray(f) for f in frames]
        f = len(frames)
        pimg = frames[0]
    else:
        f = validation_data.num_frames
        pimg = Image.open(validation_data.prompt_image)
        assert pimg.mode == "RGBA"
        r, g, b, a = pimg.split()
        pimg = Image.merge("RGB", (r, g, b))

        pimg_alpha = Image.fromarray( np.array(a.convert('L'), dtype=np.uint8)) # [h, w]
        

    width, height = pimg.size
    scale = math.sqrt(width*height / (validation_data.height*validation_data.width))
    block_size=64
    validation_data.height = round(height/scale/block_size)*block_size
    validation_data.width = round(width/scale/block_size)*block_size

    if eval_video:
        latents = []
        for frame in frames:
            input_image = vae_processor.preprocess(frame, validation_data.height, validation_data.width)
            input_image = input_image.unsqueeze(0).to(dtype).to(device)
            input_image_latents = tensor_to_vae_latent(input_image, vae)
            latents.append(input_image_latents)
        latents = torch.cat(latents, dim=2)
    else:
        input_image = vae_processor.preprocess(pimg, validation_data.height, validation_data.width)
        input_image_alpha = vae_processor.preprocess(pimg_alpha, validation_data.height, validation_data.width) # [b, 1, h, w]
        
        input_image_alpha = (input_image_alpha+1.0) / 2.0 # normalize from between -1&1 to 0&1
        
        input_image_RGBA = torch.cat([input_image, input_image_alpha], dim=1)
        input_image_premul = input_image * input_image_alpha

        input_image = input_image.to(dtype).to(device)
        input_image_RGBA = input_image_RGBA.to(dtype).to(device)
        input_image_premul = input_image_premul.to(dtype).to(device)

        # input_image_latent = vae.encode(input_image).latent_dist.mode() * vae.config.scaling_factor

        input_image_latent = vae.encode(input_image_premul).latent_dist.mode() * vae.config.scaling_factor
        alpha_latent = vae_alpha_encoder(input_image_RGBA) # [b, 4, h, w]

        # latent viz
        # im = (alpha_latent[0][:3] + 1) * 127.5
        # im = im.permute(1, 2, 0).clip(0, 255).detach().cpu().numpy().astype(np.uint8)
        # Image.fromarray(im, 'RGB').save('viz_latent_alpha.jpg')

        latents = repeat(input_image_latent, 'b c h w->b c f h w', f=f)
        latents_alpha = repeat(alpha_latent, 'b c h w->b c f h w', f=f)

        latents = latents + latents_alpha
        clean_latents = latents.detach().clone()


    mask_path = validation_data.prompt_image.split('.')[0] + '_label.jpg'
    if not os.path.exists(mask_path):
        mask_path = validation_data.prompt_image.split('.')[0] + '_label.png'
    if os.path.exists(mask_path):
        mask = Image.open(mask_path)
        mask = mask.resize((validation_data.width, validation_data.height))
        np_mask = np.array(mask)
        if len(np_mask.shape) == 3:
            np_mask = np_mask[:,:,0]
        np_mask[np_mask!=0]=255
    else:
        np_mask = np.ones([validation_data.height, validation_data.width], dtype=np.uint8)*255
    out_mask_path = os.path.splitext(out_file)[0] + "_mask.jpg"
    Image.fromarray(np_mask).save(out_mask_path)

    mask = T.ToTensor()(np_mask).to(device, dtype=dtype)
    b, c, f, h, w = latents.shape
    mask = T.Resize([h, w], antialias=False)(mask)
    mask_1_frame = repeat(mask, 'b h w -> b 1 1 h w').detach().clone()
    mask = repeat(mask, 'b h w -> b 1 f h w',f=f).detach().clone()
    mask[:,:,0] = 0

    initial_latents, timesteps = DDPM_forward_timesteps(latents, forward_t, num_frames, diffusion_scheduler) 
    #freeze_latents, timesteps = DDPM_forward_timesteps(latents[:,:,:1], forward_t, num_frames, diffusion_scheduler)
    #initial_latents = torch.randn_like(initial_latents)
    #initial_latents = initial_latents * (1-mask) + freeze_latents * mask


    freeze = repeat(latents[:,:,0], 'b c h w -> b c f h w', f=f)
    condition_latent = latents * (1-mask) + freeze * mask
    #mask = torch.ones([b, 1, 1, h, w], dtype=dtype, device=device)
    motion_strength = (index*2+3)
    with torch.no_grad():
        if in_channels == 9:
            video_frames, video_latents = ConcatLatentToVideoPipeline.__call__(
                pipeline,
                prompt=prompt,
                latents=initial_latents,
                width=validation_data.width,
                height=validation_data.height,
                num_frames=num_frames,
                num_inference_steps=validation_data.num_inference_steps,
                guidance_scale=validation_data.guidance_scale,
                mask=mask,
                motion=[motion_strength],
                return_dict=False,
                condition_latent = condition_latent,
            )
            for i in range(0):
                context = num_frames//3
                condition_latent = repeat(video_latents[:,:,-1:], 'b c 1 h w -> b c f h w', f=f)
                condition_latent = condition_latent.detach().clone()
                condition_latent[:,:,:context] = video_latents[:,:,-context:]
                initial_latents, timesteps = DDPM_forward_timesteps(input_image_latents, forward_t, num_frames, diffusion_scheduler) 
                predict_frames, video_latents = ConcatLatentToVideoPipeline.__call__(
                    pipeline,
                    prompt=prompt,
                    latents=initial_latents,
                    width=validation_data.width,
                    height=validation_data.height,
                    num_frames=num_frames,
                    num_inference_steps=validation_data.num_inference_steps,
                    guidance_scale=validation_data.guidance_scale,
                    mask=mask,
                    motion=[motion_strength],
                    return_dict=False,
                    condition_latent = condition_latent
                )
                video_frames.extend(predict_frames[context:])

        elif in_channels == 5:
            video_frames, video_latents, pngs, alpha_png, pngs_rgb = MaskedLatentToVideoPipeline.__call__(
                pipeline,
                clean_latents=clean_latents,
                # vae_alpha_encoder=vae_alpha_encoder,
                vae_alpha_decoder=vae_alpha_decoder,
                prompt=prompt,
                latents=initial_latents,
                width=validation_data.width,
                height=validation_data.height,
                num_frames=num_frames,
                num_inference_steps=validation_data.num_inference_steps,
                guidance_scale=validation_data.guidance_scale,
                motion=[motion_strength],
                return_dict=False,
                condition_latent=latents[:, :, :1].detach().clone(),
                mask=mask_1_frame,
            ) 
            
            
        else:
            # TODO: support alpha channel
            raise NotImplementedError
            video_frames, video_latents = MaskedLatentToVideoPipeline.__call__(
                pipeline,
                prompt=prompt,
                latents=initial_latents,
                width=validation_data.width,
                height=validation_data.height,
                num_frames=num_frames,
                num_inference_steps=validation_data.num_inference_steps,
                guidance_scale=validation_data.guidance_scale,
                motion=[motion_strength],
                return_dict=False,
            )

    if preview:
        fps = validation_data.get('fps', 6)
        # decoded premultiplied video by sd decoder 
        imageio.mimwrite(out_file, video_frames, duration=int(1000/fps), loop=0)

        # decoded rgba video
        imageio.mimwrite(out_file.replace('.gif', '_decoded_rgba.webp'), pngs, duration=int(1000/fps), loop=0)
        # imageio.mimwrite(out_file.replace('.gif', '_decoded_rgb.webp'), pngs_rgb, duration=int(1000/fps), loop=0)
        imageio.mimwrite(out_file.replace('.gif', '_decoded_alpha.webp'), alpha_png, duration=int(1000/fps), loop=0)
        
    real_motion_strength = calculate_latent_motion_score(video_latents).cpu().numpy()[0]
    precision = calculate_motion_precision(video_frames, np_mask)
    print(f"save file {out_file}, motion strength {motion_strength} -> {real_motion_strength}, motion precision {precision}")

    del pipeline
    torch.cuda.empty_cache()
    err = (real_motion_strength - motion_strength)
    return err*err, precision


def batch_eval(unet, text_encoder, vae, vae_alpha_encoder, vae_alpha_decoder, vae_processor, lora_manager, pretrained_model_path, 
    validation_data, output_dir, preview, num_examples=-1, global_step=0, process_index=0, 
    total_process=100, eval_file=None, iters=3, in_channels=-1):
    device = vae.device
    dtype = vae.dtype
    unet.eval()
    text_encoder.eval()
    if lora_manager is not None:
        lora_manager.deactivate_lora_train([unet, text_encoder], True)    
    #unet = torch.compile(unet)
    pipeline = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet
    )
    eval_file = None
    if eval_file is not None: 
        if os.path.isdir(eval_file):
            val_list = []
            for f in os.listdir(eval_file):
                if f.endswith('.mp4'):
                    val_list.append([os.path.join(eval_file, f), "walking, talking, moves head and hands"])
    else:
        val_list = [
            ['apple.png', 'an apple.'],
            ['ziyan0.png', 'a girl smiling.']
        ]
        for val in val_list:
            val[0] = os.path.join("example/example_padded_rgba_pngs", val[0])
    
    num_examples = 100
    if num_examples > 0:
        val_list = val_list[:num_examples]
    motion_errors = []
    motion_precisions = []
    
    os.makedirs(output_dir, exist_ok=True)

    for example in val_list:
        motion_error = 0
        motion_precision = 0
        for t in range(iters):
            name, prompt = example
            #prompt += ", high quality, 4k, photo realistic, bright"
            out_file_dir = f"{output_dir}/{os.path.basename(name).split('.')[0]}"
            os.makedirs(out_file_dir, exist_ok=True)
            out_file = f"{out_file_dir}/{global_step+t}.gif"
            validation_data.prompt_image = name
            validation_data.prompt = prompt
            error, precision = eval(pipeline, vae_processor, vae_alpha_encoder, vae_alpha_decoder, 
                validation_data, out_file, t, forward_t=validation_data.num_inference_steps, preview=preview, in_channels=in_channels)
            motion_error += error
            motion_precision += precision
        motion_error = motion_error/iters
        motion_precision = motion_precision/iters
        # print(example[0], "average motion strength error", motion_error, "precision", motion_precision)
        motion_errors.append(motion_error)
        motion_precisions.append(motion_precision)
    # print(motion_errors, motion_precisions)
    # print("average motion strength error", sum(motion_errors)/len(val_list), 
    #     "precision", sum(motion_precisions)/len(val_list))
    del pipeline

def main_eval(
    validation_data: Dict,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = False,
    seed: Optional[int] = None,
    motion_mask = False, 
    motion_strength = False,
    eval_file=None,
    num_examples=1,
    iters=10,
    output_dir="output/stage_2_eval",
    in_channels=-1,
    transparent_unet_pretrained_model_path = './output/latent/transparent_unet',
    transparent_VAE_pretrained_model_path = './output/latent/transparent_VAE',
    **kwargs
):  
    iters=3

    noise_scheduler, tokenizer, text_encoder, vae, unet, vae_alpha_encoder, vae_alpha_decoder = load_primary_models(transparent_unet_pretrained_model_path, motion_strength=motion_strength, 
        alpha_chechpoint=transparent_VAE_pretrained_model_path
    )
    vae_processor = VaeImageProcessor()
    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet, vae_alpha_encoder, vae_alpha_decoder])
    
    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    lora_manager = None
    
    # Enable VAE slicing to save memory.
    vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.half

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder, unet, vae, vae_alpha_encoder, vae_alpha_decoder]
    cast_to_gpu_and_type(models_to_cast, torch.device("cuda"), weight_dtype)
    batch_eval(unet, text_encoder, vae, vae_alpha_encoder, vae_alpha_decoder, vae_processor, lora_manager, transparent_unet_pretrained_model_path, 
        validation_data, output_dir, True, num_examples=num_examples, eval_file=eval_file, iters=iters, in_channels=in_channels)

if __name__ == "__main__":
    # python train_transparent_i2v_stage2.py --config example/layerdiffuse_stage2_384.yaml --eval
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./example/layerdiffuse_stage2_384.yaml")
    parser.add_argument("--eval", action="store_true", default=True)
    args, unknownargs = parser.parse_known_args()
    args_dict = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args_dict = OmegaConf.merge(args_dict, cli_conf)
    if args.eval:
        main_eval(**args_dict)
    # else:
    #     main(**args_dict)
