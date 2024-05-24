import torch
from einops import rearrange, repeat
import numpy as np

from diffusers import TextToVideoSDPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid, TextToVideoSDPipelineOutput
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
import torchvision.transforms as T


class ImageToVideoPipeline(TextToVideoSDPipeline):
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        image_embeds=None,
        negative_prompt=None,
        prompt_embeds: torch.FloatTensor = None,
        negative_prompt_embeds: torch.FloatTensor = None,
        lora_scale: float = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                input_ids=text_input_ids.to(device), 
                ctx_embeddings=image_embeds,
                ctx_begin_pos=[2] * text_input_ids.shape[0],
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype)#.to(device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = text_input_ids.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input.input_ids.to(device), 
                ctx_embeddings=image_embeds,
                ctx_begin_pos=[2] * uncond_input.input_ids.shape[0],
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

class MaskedLatentToVideoPipeline(TextToVideoSDPipeline):
    @torch.no_grad()
    def __call__(
        self,
        clean_latents=None, # [b, c, f, h, w] 
        vae_alpha_decoder=None,
        prompt = None,
        height= None,
        width= None,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale= 9.0,
        negative_prompt= None,
        eta: float = 0.0,
        generator= None,
        latents= None,
        condition_latent=None,
        prompt_embeds= None,
        negative_prompt_embeds= None,
        output_type= "np",
        return_dict: bool = True,
        callback= None,
        callback_steps: int = 1,
        cross_attention_kwargs= None,
        timesteps=None,
        mask=None,
        motion=None,
        image_embeds=None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_images_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        #device = self._execution_device
        device = latents.device
        dtype = latents.dtype
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self.encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, 
            negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, lora_scale=text_encoder_lora_scale,)
        prompt_embeds = torch.cat([prompt_embeds[1],prompt_embeds[0]]) if do_classifier_free_guidance else prompt_embeds[0]
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if timesteps is None:
            timesteps = self.scheduler.timesteps
        else:
            num_inference_steps = len(timesteps)
        # 5. Prepare latent variables. do nothing

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        init_latents = latents
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        uncondition_latent = condition_latent
        condition_latent = torch.cat([uncondition_latent, condition_latent]) if do_classifier_free_guidance else condition_latent 
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if motion is not None:
                    motion = torch.tensor(motion, device=device)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    mask=mask,
                    motion=motion,
                    condition_latent=condition_latent,
                    image_embeds = image_embeds,
                ).sample
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                bsz, channel, frames, height, width = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, height, width)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, height, width)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(bsz, frames, channel, height, width).permute(0, 2, 1, 3, 4)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # latents = clean_latents
        video_tensor = self.decode_latents(latents) # [1, c, f, h, w] float32
        # b, c, f, h, w = video_tensor.shape
        # video_tensor = self.decode_latents(latents[:, :4]) # [b, c , f, h, w]
        # alpha_video_tensor = self.decode_latents(latents[:, 4:]) # [b, c , f, h, w]
        b, c, f, h, w = video_tensor.shape


        x = video_tensor.permute(0, 2, 1, 3, 4).reshape(b*f, c, h, w).to(dtype)
        latent = latents.permute(0, 2, 1, 3, 4).reshape(b*f, 4, height, width)

        decoded_rgba = vae_alpha_decoder(x, latent)

        decoded_rgba = decoded_rgba.reshape(b, f, 4, h, w).permute(0, 2, 1, 3, 4)

        alpha = decoded_rgba[:, 3:]
        fg = decoded_rgba[:, :3] 

        alpha = alpha * 255.0

        alpha[alpha>127] = 255
        alpha[alpha<=127] = 0

        fg = (fg+1.0) * 127.5

        
        pngs = torch.cat((fg, alpha), dim=1)[0].permute(1, 0, 2, 3).permute(0, 2, 3, 1) # [f, h, w, c]
        pngs = pngs.detach().cpu().float().numpy().clip(0, 255).astype(np.uint8)
        pngs_rgb = pngs[:, :, :, :3]
        alpha_jpg = pngs[:, :, :, 3]


        assert b == 1

        if output_type == "pt":
            video = video_tensor
        else:
            video = tensor2vid(video_tensor) # [f, h, w, c] # video_tensor required -1&1
            # alpha_video = tensor2vid(alpha_video_tensor)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        # pngs_denoised = pngs_denoised[:, :, :, -1]
        if not return_dict:
            return (video, latents, pngs, alpha_jpg, pngs_rgb)
            # return (video, latents, rgba_video)

        return TextToVideoSDPipelineOutput(frames=video)

class ConcatLatentToVideoPipeline(TextToVideoSDPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt = None,
        height= None,
        width= None,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale= 9.0,
        negative_prompt= None,
        eta: float = 0.0,
        generator= None,
        latents= None,
        prompt_embeds= None,
        negative_prompt_embeds= None,
        output_type= "np",
        return_dict: bool = True,
        callback= None,
        callback_steps: int = 1,
        cross_attention_kwargs= None,
        timesteps=None,
        mask=None,
        motion=None,
        image_embeds=None,
        condition_latent = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_images_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        #device = self._execution_device
        device = latents.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self.encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, 
            negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, lora_scale=text_encoder_lora_scale,)
        prompt_embeds = torch.cat([prompt_embeds[1],prompt_embeds[0]]) if do_classifier_free_guidance else prompt_embeds[0]
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if timesteps is None:
            timesteps = self.scheduler.timesteps
        else:
            num_inference_steps = len(timesteps)
        # 5. Prepare latent variables. do nothing

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                latent_model_input = torch.cat([condition_latent, latents], dim=1)
                latent_model_input = torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input
                if motion is not None:
                    motion = torch.tensor(motion, device=device)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    mask=mask,
                    motion=motion,
                    image_embeds = image_embeds,
                ).sample
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                bsz, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        video_tensor = self.decode_latents(latents)

        if output_type == "pt":
            video = video_tensor
        else:
            video = tensor2vid(video_tensor)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (video, latents)

        return TextToVideoSDPipelineOutput(frames=video)
    @torch.no_grad()
    def __call__(
        self,
        vae_transparent_encoder,
        prompt = None,
        height= None,
        width= None,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale= 9.0,
        negative_prompt= None,
        eta: float = 0.0,
        generator= None,
        latents= None,
        prompt_embeds= None,
        negative_prompt_embeds= None,
        output_type= "np",
        return_dict: bool = True,
        callback= None,
        callback_steps: int = 1,
        cross_attention_kwargs= None,
        timesteps=None,
        mask=None,
        motion=None,
        image_embeds=None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_images_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        #device = self._execution_device
        device = latents.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self.encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, 
            negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, lora_scale=text_encoder_lora_scale,)
        prompt_embeds = torch.cat([prompt_embeds[1],prompt_embeds[0]]) if do_classifier_free_guidance else prompt_embeds[0]
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if timesteps is None:
            timesteps = self.scheduler.timesteps
        else:
            num_inference_steps = len(timesteps)
        # 5. Prepare latent variables. do nothing

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        init_latents = latents
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if motion is not None:
                    motion = torch.tensor(motion, device=device)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    mask=mask,
                    motion=motion,
                    image_embeds = image_embeds,
                ).sample
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                bsz, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        latents # latents transparency
        video_tensor = self.decode_latents(latents) # [1, c, f, h, w] decoded premul imgs

        if output_type == "pt":
            video = video_tensor
        else:
            video = tensor2vid(video_tensor) # [f, h, w, c]

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (video, latents)

        return TextToVideoSDPipelineOutput(frames=video)