import json
import os
import os.path as osp
import random
from argparse import ArgumentParser
from datetime import datetime
import math

import gradio as gr
import numpy as np
import torch
from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from diffusers.image_processor import VaeImageProcessor
from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms as T
from einops import rearrange, repeat
import imageio

from models.pipeline import LatentToVideoPipeline
from utils.common import tensor_to_vae_latent, DDPM_forward_timesteps

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""


class AnimateController:
    def __init__(self, pretrained_model_path: str, validation_data,
        output_dir, motion_mask = False, motion_strength = False):
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        device=torch.device("cuda")
        self.validation_data = validation_data
        self.output_dir = output_dir
        self.pipeline = LatentToVideoPipeline.from_pretrained(pretrained_model_path, 
            torch_dtype=torch.float16, variant="fp16").to(device)
        self.sample_idx = 0

    def animate(
        self,
        init_img,
        motion_scale,
        prompt_textbox,
        negative_prompt_textbox,
        sample_step_slider,
        cfg_scale_slider,
        seed_textbox,
        style,
        progress=gr.Progress(),
    ):

        if seed_textbox != -1 and seed_textbox != "":
            torch.manual_seed(int(seed_textbox))
        else:
            torch.seed()
        seed = torch.initial_seed()

        vae = self.pipeline.vae
        diffusion_scheduler = self.pipeline.scheduler
        validation_data = self.validation_data
        vae_processor =  VaeImageProcessor()

        device = vae.device
        dtype = vae.dtype

        pimg = Image.fromarray(init_img["background"]).convert('RGB')
        width, height = pimg.size
        scale = math.sqrt(width*height / (validation_data.height*validation_data.width))
        block_size=8
        height = round(height/scale/block_size)*block_size
        width = round(width/scale/block_size)*block_size
        input_image = vae_processor.preprocess(pimg, height, width)
        input_image = input_image.unsqueeze(0).to(dtype).to(device)
        input_image_latents = tensor_to_vae_latent(input_image, vae)
        np_mask = init_img["layers"][0][:,:,3]
        np_mask[np_mask!=0] = 255
        if np_mask.sum() == 0:
            np_mask[:] = 255
        #np_mask = np.ones([height, width], dtype=np.uint8)*255


        b, c, _, h, w = input_image_latents.shape
        initial_latents, timesteps = DDPM_forward_timesteps(input_image_latents, 
            sample_step_slider, validation_data.num_frames, diffusion_scheduler) 
        mask = T.ToTensor()(np_mask).to(dtype).to(device)
        b, c, f, h, w = initial_latents.shape
        mask = T.Resize([h, w], antialias=False)(mask)
        mask = rearrange(mask, 'b h w -> b 1 1 h w')

        motion_strength = motion_scale
        with torch.no_grad():
            video_frames, video_latents = self.pipeline(
                prompt=prompt_textbox,
                latents=initial_latents,
                width=width,
                height=height,
                num_frames=validation_data.num_frames,
                num_inference_steps=sample_step_slider,
                guidance_scale=cfg_scale_slider,
                condition_latent=input_image_latents,
                mask=mask,
                motion=[motion_strength],
                return_dict=False,
                timesteps=timesteps,
            )

        save_sample_path = os.path.join(
            self.output_dir, f"{self.sample_idx}.mp4")
        imageio.mimwrite(save_sample_path, video_frames, fps=8)
        self.sample_idx += 1
        return save_sample_path


def ui(controller):
    with gr.Blocks(css=css) as demo:

        gr.HTML(
            "<div align='center'><font size='7'> <img src=\"file/example/barbie2.jpg\" style=\"height: 72px;\"/ >Animate Anything</font></div>"
        )
        with gr.Row():
            gr.Markdown(
                "<div align='center'><font size='5'><a href='https://animationai.github.io/AnimateAnything'>Project Page</a> &ensp;"  # noqa
                "<a href='https://arxiv.org/abs/2311.12886'>Paper</a> &ensp;"
                "<a href='https://github.com/alibaba/animate-anything'>Code</a> &ensp;"  # noqa
            )

        with gr.Row(equal_height=False):
            with gr.Column():
                with gr.Row():
                    init_img = gr.ImageMask(label='Input Image', brush=gr.Brush(default_size=100))
                style_dropdown = gr.Dropdown(label='Style', choices=['384', '512'], value='512')
                with gr.Row():
                    prompt_textbox = gr.Textbox(label="Prompt", lines=1)
                    gift_button = gr.Button(
                        value='🎁', elem_classes='toolbutton'
                    )

                def append_gift(prompt):
                    rand = random.randint(0, 2)
                    if rand == 1:
                        prompt = prompt + 'wearing santa hats'
                    elif rand == 2:
                        prompt = prompt + 'lift a Christmas gift'
                    else:
                        prompt = prompt + 'in Christmas suit, lift a Christmas gift'
                    gr.Info('Merry Christmas! Add magic to your prompt!')
                    return prompt

                gift_button.click(
                    fn=append_gift,
                    inputs=[prompt_textbox],
                    outputs=[prompt_textbox],
                )

                motion_scale_silder = gr.Slider(
                    label='Motion Strength (Larger value means larger motion but less identity consistency)',
                    value=5, step=1, minimum=1, maximum=20)

                with gr.Accordion('Advance Options', open=False):
                    negative_prompt_textbox = gr.Textbox(
                        value="", label="Negative prompt", lines=2)

                    sample_step_slider = gr.Slider(
                            label="Sampling steps", value=25, minimum=10, maximum=100, step=1)

                    cfg_scale_slider = gr.Slider(
                        label="CFG Scale", value=7.5, minimum=0, maximum=20)

                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=-1)
                        seed_button = gr.Button(
                            value="\U0001F3B2", elem_classes="toolbutton")
                    seed_button.click(
                        fn=lambda x: random.randint(1, 1e8),
                        outputs=[seed_textbox],
                        queue=False
                    )

                generate_button = gr.Button(
                    value="Generate", variant='primary')

            result_video = gr.Video(
                label="Generated Animation", interactive=False)

        generate_button.click(
            fn=controller.animate,
            inputs=[
                init_img,
                motion_scale_silder,
                prompt_textbox,
                negative_prompt_textbox,
                sample_step_slider,
                cfg_scale_slider,
                seed_textbox,
                style_dropdown,
            ],
            outputs=[result_video]
        )

        def create_example(input_list):
            return gr.Examples(
                examples=input_list,
                inputs=[
                    init_img,
                    result_video,
                    prompt_textbox,
                    negative_prompt_textbox,
                    style_dropdown,
                    motion_scale_silder,
                ],
            )

        gr.Markdown(
            '### Merry Christmas!'
        )
        create_example(
            [
                [
                    'example/girl5.jpg',
                    'docs/girl5.mp4',
                    'a girl is smiling',
                    '3d_cartoon',
                    2,
                ],
            ],

        )

    return demo


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='example/config/base.yaml')
    parser.add_argument('--server-name', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--local-debug', action='store_true')
    parser.add_argument('--save-path', default='samples')

    args, unknownargs = parser.parse_known_args()
    LOCAL_DEBUG = args.local_debug
    args_dict = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args_dict = OmegaConf.merge(args_dict, cli_conf)
    controller = AnimateController(args_dict.pretrained_model_path, args_dict.validation_data, 
        args_dict.output_dir, args_dict.motion_mask, args_dict.motion_strength)
    demo = ui(controller)
    demo.queue(max_size=10)
    demo.launch(server_name=args.server_name,
                server_port=args.port, max_threads=40,
                allowed_paths=['example/barbie2.jpg'])
