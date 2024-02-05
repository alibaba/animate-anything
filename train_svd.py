import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy
import json

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers

from tqdm.auto import tqdm
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers.models import AutoencoderKL, UNetSpatioTemporalConditionModel
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid
from diffusers import StableVideoDiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor, CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIPEncoder
from utils.dataset import get_train_dataset, extend_datasets
from einops import rearrange, repeat
import imageio


from models.unet_3d_condition_mask import UNet3DConditionModel
from models.pipeline import MaskStableVideoDiffusionPipeline
from utils.common import read_mask, generate_random_mask, slerp, calculate_motion_score, \
    read_video, calculate_motion_precision, calculate_latent_motion_score, \
    DDPM_forward, DDPM_forward_timesteps, motion_mask_loss

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

def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir

def load_primary_models(pretrained_model_path, eval=False):
    if eval:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16, variant='fp16')
    else:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
    return pipeline, None, pipeline.feature_extractor, pipeline.scheduler, pipeline.image_processor, \
        pipeline.image_encoder, pipeline.vae, pipeline.unet

def convert_svd(pretrained_model_path, out_path):
    pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_path, subfolder="unet_mask", low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    unet.conv_in.bias.data = copy.deepcopy(pipeline.unet.conv_in.bias)
    torch.nn.init.zeros_(unet.conv_in.weight)
    unet.conv_in.weight.data[:,1:]= copy.deepcopy(pipeline.unet.conv_in.weight)
    new_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_path, unet=unet)
    new_pipeline.save_pretrained(out_path)

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
    

def negate_params(name, negation):
    # We have to do this if we are co-training with LoRA.
    # This ensures that parameter groups aren't duplicated.
    if negation is None: return False
    for n in negation:
        if n in name and 'temp' not in name:
            return True
    return False


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

def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        for n, p in model.named_parameters():
            if p.requires_grad:
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

def handle_trainable_modules(model, trainable_modules=None, is_enabled=True, negation=None):
    global already_printed_trainables

    # This can most definitely be refactored :-)
    unfrozen_params = 0
    if trainable_modules is not None:
        for name, module in model.named_modules():
            for tm in tuple(trainable_modules):
                if tm == 'all':
                    model.requires_grad_(is_enabled)
                    unfrozen_params =len(list(model.parameters()))
                    break
                    
                if tm in name and 'lora' not in name:
                    for m in module.parameters():
                        m.requires_grad_(is_enabled)
                        if is_enabled: unfrozen_params +=1

    if unfrozen_params > 0 and not already_printed_trainables:
        already_printed_trainables = True 
        print(f"{unfrozen_params} params have been unfrozen for training.")

def sample_noise(latents, noise_strength, use_offset_noise=False):
    b ,c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def enforce_zero_terminal_snr(betas):
    """
    Corrects noise in diffusion schedulers.
    From: Common Diffusion Noise Schedules and Sample Steps are Flawed
    https://arxiv.org/pdf/2305.08891.pdf
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
        alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas

def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step == 5)  \
    and validation_data.sample_preview

def save_pipe(
        path, 
        global_step,
        accelerator, 
        unet, 
        text_encoder, 
        vae, 
        output_dir,
        is_checkpoint=False,
        save_pretrained_model=True
    ):

    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    unet_out = copy.deepcopy(unet)
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        path, unet=unet_out).to(torch_dtype=torch.float32)

    if save_pretrained_model:
        pipeline.save_pretrained(save_path)

    logger.info(f"Saved model at {save_path} on step {global_step}")
    
    del pipeline
    del unet_out
    torch.cuda.empty_cache()
    gc.collect()


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
    
def finetune_unet(accelerator, pipeline, batch, use_offset_noise,
    rescale_schedule, offset_noise_strength, unet, motion_mask, 
    P_mean=0.7, P_std=1.6):
    pipeline.vae.eval()
    pipeline.image_encoder.eval()
    device = unet.device
    dtype = pipeline.vae.dtype
    vae = pipeline.vae
    # Convert videos to latent space
    pixel_values = batch['pixel_values']
    bsz, num_frames = pixel_values.shape[:2]

    frames = rearrange(pixel_values, 'b f c h w-> (b f) c h w').to(dtype)
    latents = vae.encode(frames).latent_dist.mode() * vae.config.scaling_factor
    latents = rearrange(latents, '(b f) c h w-> b f c h w', b=bsz)

    # enocde image latent
    image = pixel_values[:,0].to(dtype)
    noise_aug_strength = math.exp(random.normalvariate(mu=-3, sigma=0.5))
    image = image + noise_aug_strength * torch.randn_like(image)
    image_latent = vae.encode(image).latent_dist.mode() * vae.config.scaling_factor

    if motion_mask:
        mask = batch['mask']
        mask = mask.div(255)
        h, w = latents.shape[-2:]
        mask = T.Resize((h, w), antialias=False)(mask)
        mask[mask<0.5] = 0
        mask[mask>=0.5] = 1
        mask = repeat(mask, 'b h w -> b f 1 h w', f=num_frames).detach().clone()
        mask[:,0] = 0
        freeze = repeat(image_latent, 'b c h w -> b f c h w', f=num_frames)
        condition_latent = latents * (1-mask) + freeze * mask
    else:
        condition_latent = repeat(image_latent, 'b c h w->b f c h w',f=num_frames)


    pipeline.image_encoder.to(device, dtype=dtype)
    images = _resize_with_antialiasing(pixel_values[:,0], (224, 224)).to(dtype)
    images = (images + 1.0) / 2.0 # [-1, 1] -> [0, 1]
    images = pipeline.feature_extractor(
        images=images,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values 
    image_embeddings = pipeline._encode_image(images, device, 1, False)

    encoder_hidden_states = image_embeddings
    uncond_hidden_states = torch.zeros_like(image_embeddings)
    
    if random.random() < 0.15: 
        encoder_hidden_states = uncond_hidden_states
    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process) #[bsz, f, c, h , w]
    rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    c_skip = 1 / (sigma**2 + 1)
    c_out =  -sigma / (sigma**2 + 1) ** 0.5
    c_in = 1 / (sigma**2 + 1) ** 0.5
    c_noise = (sigma.log() / 4).reshape([bsz])
    loss_weight = (sigma ** 2 + 1) / sigma ** 2

    noisy_latents = latents + torch.randn_like(latents) * sigma
    input_latents = torch.cat([c_in * noisy_latents, 
        condition_latent/vae.config.scaling_factor], dim=2)
    if motion_mask:
        input_latents = torch.cat([mask, input_latents], dim=2)

    motion_bucket_id = 127
    fps = 7
    added_time_ids = pipeline._get_add_time_ids(fps, motion_bucket_id, 
        noise_aug_strength, image_embeddings.dtype, bsz, 1, False)
    added_time_ids = added_time_ids.to(device)

    loss = 0

    accelerator.wait_for_everyone()
    model_pred = unet(input_latents, c_noise, encoder_hidden_states=encoder_hidden_states, added_time_ids=added_time_ids).sample
    predict_x0 = c_out * model_pred + c_skip * noisy_latents 
    loss += ((predict_x0 - latents)**2 * loss_weight).mean()
    if motion_mask:
        loss += F.mse_loss(predict_x0*(1-mask), condition_latent*(1-mask))
    return loss


def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    extra_train_data: list = [],
    dataset_types: Tuple[str] = ('json'),
    shuffle: bool = True,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = None, # Eg: ("attn1", "attn2")
    extra_unet_params = None,
    extra_text_encoder_params = None,
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 5e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    text_encoder_gradient_checkpointing: bool = False,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    resume_step: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = False,
    seed: Optional[int] = None,
    use_offset_noise: bool = False,
    rescale_schedule: bool = False,
    offset_noise_strength: float = 0.1,
    extend_dataset: bool = False,
    cache_latents: bool = False,
    cached_latent_dir = None,
    save_pretrained_model: bool = True,
    logger_type: str = 'tensorboard',
    motion_mask=False,
    **kwargs
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logger_type,
        project_dir=output_dir
    )

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
       output_dir = create_output_folders(output_dir, config)

    # Load scheduler, tokenizer and models. The text encoder is actually image encoder for SVD
    pipeline, tokenizer, feature_extractor, train_scheduler, vae_processor, text_encoder, vae, unet = load_primary_models(pretrained_model_path)
    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])
    
    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    # Create parameters to optimize over with a condition (if "condition" is true, optimize it)
    extra_unet_params = extra_unet_params if extra_unet_params is not None else {}
    extra_text_encoder_params = extra_unet_params if extra_unet_params is not None else {}

    trainable_modules_available = trainable_modules is not None

    # Unfreeze UNET Layers
    if trainable_modules_available:
        unet.train()
        handle_trainable_modules(
            unet, 
            trainable_modules, 
            is_enabled=True,
        )

    optim_params = [
        param_optim(unet, trainable_modules_available, extra_params=extra_unet_params)
    ]

    params = create_optimizer_params(optim_params, learning_rate)
    
    # Create Optimizer
    optimizer = optimizer_cls(
        params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Get the training dataset based on types (json, single_video, image)
    train_datasets = get_train_dataset(dataset_types, train_data, tokenizer)

    # If you have extra train data, you can add a list of however many you would like.
    # Eg: extra_train_data: [{: {dataset_types, train_data: {etc...}}}] 
    try:
        if extra_train_data is not None and len(extra_train_data) > 0:
            for dataset in extra_train_data:
                d_t, t_d = dataset['dataset_types'], dataset['train_data']
                train_datasets += get_train_dataset(d_t, t_d, tokenizer)

    except Exception as e:
        print(f"Could not process extra train datasets due to an error : {e}")

    # Extend datasets that are less than the greatest one. This allows for more balanced training.
    attrs = ['train_data', 'frames', 'image_dir', 'video_files']
    extend_datasets(train_datasets, attrs, extend=extend_dataset)

    # Process one dataset
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    
    # Process many datasets
    else:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets) 

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=shuffle
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, 
        optimizer, 
        train_dataloader, 
        lr_scheduler, 
    )

    # Use Gradient Checkpointing if enabled.
    unet_and_text_g_c(
        unet, 
        text_encoder, 
        gradient_checkpointing, 
        text_encoder_gradient_checkpointing
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder, vae]
    cast_to_gpu_and_type(models_to_cast, accelerator.device, weight_dtype)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    num_train_epochs = math.ceil(max_train_steps * gradient_accumulation_steps / len(train_dataloader) / accelerator.num_processes)


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
     
    # *Potentially* Fixes gradient checkpointing training.
    # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
    if kwargs.get('eval_train', False):
        unet.eval()
        text_encoder.eval()
 
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(unet) ,accelerator.accumulate(text_encoder):
                with accelerator.autocast():
                    loss = finetune_unet(accelerator, pipeline, batch, use_offset_noise, 
                        rescale_schedule, offset_noise_strength, unet, motion_mask)
                device = loss.device 
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                params_to_clip = unet.parameters()
                accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
            
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                    
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
            
                if global_step % checkpointing_steps == 0 and accelerator.is_main_process:
                    save_pipe(
                        pretrained_model_path, 
                        global_step, 
                        accelerator, 
                        accelerator.unwrap_model(unet),
                        accelerator.unwrap_model(text_encoder), 
                        vae, 
                        output_dir, 
                        is_checkpoint=True,
                        save_pretrained_model=save_pretrained_model
                    )

                if should_sample(global_step, validation_steps, validation_data):
                    if global_step == 1: print("Performing validation prompt.")
                    if accelerator.is_main_process:
                        
                        with accelerator.autocast():
                            curr_dataset_name = batch['dataset'][0]
                            save_filename = f"{global_step}_dataset-{curr_dataset_name}"
                            out_file = f"{output_dir}/samples/{save_filename}.gif"
                            eval(pipeline, vae_processor, validation_data, out_file, global_step)
                            logger.info(f"Saved a new sample to {out_file}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log({"training_loss": loss.detach().item()}, step=step)
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_pipe(
                pretrained_model_path, 
                global_step, 
                accelerator, 
                accelerator.unwrap_model(unet),
                accelerator.unwrap_model(text_encoder), 
                vae, 
                output_dir, 
                is_checkpoint=False,
                save_pretrained_model=save_pretrained_model
        )     
    accelerator.end_training()

def eval(pipeline, vae_processor, validation_data, out_file, index, forward_t=25, preview=True):
    vae = pipeline.vae
    device = vae.device
    dtype = vae.dtype
    
    diffusion_scheduler = pipeline.scheduler
    diffusion_scheduler.set_timesteps(validation_data.num_inference_steps, device=device)

    prompt = validation_data.prompt
    pimg = Image.open(validation_data.prompt_image)
    if pimg.mode == "RGBA":
        pimg = pimg.convert("RGB")
    width, height = pimg.size
    scale = math.sqrt(width*height / (validation_data.height*validation_data.width))
    block_size=64
    validation_data.height = round(height/scale/block_size)*block_size
    validation_data.width = round(width/scale/block_size)*block_size

    mask_path = validation_data.prompt_image.split('.')[0] + '_label.jpg'
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
    motion_mask = pipeline.unet.config.in_channels == 9

    # prepare inital latents
    initial_latents = None
    with torch.no_grad():
        if motion_mask:
            h, w = validation_data.height//pipeline.vae_scale_factor, validation_data.width//pipeline.vae_scale_factor
            initial_latents = torch.randn([1, validation_data.num_frames, 4, h, w], dtype=dtype, device=device)
            mask = T.ToTensor()(np_mask).to(dtype).to(device)
            mask = T.Resize([h, w], antialias=False)(mask)
            video_frames = MaskStableVideoDiffusionPipeline.__call__(
                pipeline,
                image=pimg,
                width=validation_data.width,
                height=validation_data.height,
                num_frames=validation_data.num_frames,
                num_inference_steps=validation_data.num_inference_steps,
                decode_chunk_size=validation_data.decode_chunk_size,
                fps=validation_data.fps,
                motion_bucket_id=validation_data.motion_bucket_id,
                mask=mask
            ).frames[0]
        else:
            video_frames = pipeline(
                image=pimg,
                width=validation_data.width,
                height=validation_data.height,
                num_frames=validation_data.num_frames,
                num_inference_steps=validation_data.num_inference_steps,
                fps=validation_data.fps,
                decode_chunk_size=validation_data.decode_chunk_size,
                motion_bucket_id=validation_data.motion_bucket_id,
            ).frames[0]
    
    if preview:
        fps = validation_data.get('fps', 8)
        imageio.mimwrite(out_file, video_frames, duration=int(1000/fps), loop=0)
        imageio.mimwrite(out_file.replace('.gif', '.mp4'), video_frames, fps=fps)
    return 0

def main_eval(
    pretrained_model_path: str,
    validation_data: Dict,
    seed: Optional[int] = None,
    eval_file = None,
    **kwargs
):
    if seed is not None:
        set_seed(seed)
    # Load scheduler, tokenizer and models.
    pipeline, tokenizer, feature_extractor, train_scheduler, vae_processor, text_encoder, vae, unet = load_primary_models(pretrained_model_path, eval=True)
    device = torch.device("cuda")
    pipeline.to(device)

    if eval_file is not None:
        eval_list = json.load(open(eval_file))
    else:
        eval_list = [[validation_data.prompt_image, validation_data.prompt]]

    output_dir = "output/svd_out"
    iters = 5
    for example in eval_list:
        for t in range(iters):
            name, prompt = example
            out_file_dir = f"{output_dir}/{name.split('.')[0]}"
            os.makedirs(out_file_dir, exist_ok=True)
            out_file = f"{out_file_dir}/{t}.gif"
            validation_data.prompt_image = name
            validation_data.prompt = prompt
            eval(pipeline, vae_processor, validation_data, out_file, t)
            print("save file", out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/my_config.yaml")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args_dict = OmegaConf.load(args.config)
    cli_dict = OmegaConf.from_dotlist(args.rest)
    args_dict = OmegaConf.merge(args_dict, cli_dict)
    if args.eval:
        main_eval(**args_dict)
    else:
        main(**args_dict)
