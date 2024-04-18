import os
import random
from argparse import ArgumentParser
import math

import gradio as gr
import torch
from diffusers.image_processor import VaeImageProcessor
from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms as T
import imageio

from diffusers import StableVideoDiffusionPipeline
from models.pipeline import TextStableVideoDiffusionPipeline
from einops import rearrange, repeat
from utils.common import read_video

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
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16, variant="fp16").to(device)
        #self.pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path).to(device)
        self.sample_idx = 0

    def animate(
        self,
        init_img,
        input_video,
        sample_step_slider,
        seed_textbox,
        fps_textbox,
        num_frames_textbox,
        motion_bucket_id_slider,
        progress=gr.Progress(),
    ):

        if seed_textbox != "-1" and seed_textbox != "":
            torch.manual_seed(int(seed_textbox))
        else:
            torch.seed()
        seed = torch.initial_seed()

        with torch.no_grad():
            vae = self.pipeline.vae
            validation_data = self.validation_data
            validation_data.fps = int(fps_textbox)
            validation_data.num_frames = int(num_frames_textbox)
            validation_data.motion_bucket_id = int(motion_bucket_id_slider)
            vae_processor = VaeImageProcessor()

            device = vae.device
            dtype = vae.dtype

            f = validation_data.num_frames
            pimg = Image.fromarray(init_img["background"]).convert('RGB')
            np_mask = init_img["layers"][0][:,:,3]
            np_mask[np_mask!=0] = 255
            if np_mask.sum() == 0:
                np_mask[:] = 255
            if input_video is not None:
                frames = read_video(input_video)
                frames = [Image.fromarray(f) for f in frames]
                pimg = frames[0]
                width, height = pimg.size
                scale = math.sqrt(width*height / (validation_data.height*validation_data.width))
                block_size=64
                height = round(height/scale/block_size)*block_size
                width = round(width/scale/block_size)*block_size
                f = len(frames)
                
                latents = []
                for frame in frames:
                    input_image = vae_processor.preprocess(frame, height, width)
                    input_image = input_image.to(dtype).to(device)
                    input_image_latent = vae.encode(input_image).latent_dist.mode() * vae.config.scaling_factor
                    latents.append(input_image_latent.unsqueeze(1))
                latents = torch.cat(latents, dim=1)
            else:
                width, height = pimg.size
                scale = math.sqrt(width*height / (validation_data.height*validation_data.width))
                block_size=64
                height = round(height/scale/block_size)*block_size
                width = round(width/scale/block_size)*block_size
                input_image = vae_processor.preprocess(pimg, height, width)
                input_image = input_image.to(dtype).to(device)
                input_image_latent = vae.encode(input_image).latent_dist.mode() * vae.config.scaling_factor
                latents = repeat(input_image_latent, 'b c h w->b f c h w', f=f)
        
            b, f, c, h, w = latents.shape

            mask = T.ToTensor()(np_mask).to(dtype).to(device)
            mask = T.Resize([h, w], antialias=False)(mask)
            mask = repeat(mask, 'b h w -> b f 1 h w', f=f).detach().clone()
            mask[:,0] = 0
            freeze = repeat(latents[:,0], 'b c h w -> b f c h w', f=f)
            condition_latents = latents * (1-mask) + freeze * mask
            condition_latents = condition_latents/vae.config.scaling_factor

            motion_mask = self.pipeline.unet.config.in_channels == 9
            decode_chunk_size=validation_data.get("decode_chunk_size", 7)
            fps=validation_data.get("fps", 7)
            motion_bucket_id=validation_data.get("motion_bucket_id", 127)
            if motion_mask:
                video_frames = TextStableVideoDiffusionPipeline.__call__(
                    self.pipeline,
                    image=pimg,
                    width=width,
                    height=height,
                    num_frames=validation_data.num_frames,
                    num_inference_steps=validation_data.num_inference_steps,
                    decode_chunk_size=decode_chunk_size,
                    fps=fps,
                    motion_bucket_id=motion_bucket_id,
                    mask=mask,
                    condition_type="image",
                    condition_latent=condition_latents
                ).frames[0]
            else:
                video_frames = self.pipeline(
                    image=pimg,
                    width=width,
                    height=height,
                    num_frames=validation_data.num_frames,
                    num_inference_steps=validation_data.num_inference_steps,
                    fps=validation_data.fps,
                    decode_chunk_size=validation_data.decode_chunk_size,
                    motion_bucket_id=validation_data.motion_bucket_id,
                ).frames[0]

        save_sample_path = os.path.join(
            self.output_dir, f"{self.sample_idx}.mp4")
        Image.fromarray(np_mask).save(os.path.join(
            self.output_dir, f"{self.sample_idx}_label.jpg"))
        imageio.mimwrite(save_sample_path, video_frames, fps=7)
        self.sample_idx += 1
        return save_sample_path

import cv2

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return length

def update_num_frames(input_video, num_frames_textbox):
    frame_count = get_video_info(input_video)
    return frame_count or 14

def ui(controller):
    with gr.Blocks(css=css) as demo:

        gr.HTML(
            "<div align='center'><font size='7'> <img src=\"file/example/barbie2.jpg\" style=\"height: 72px;\"/ >Animate Anything For SVD</font></div>"
        )
        with gr.Row():
            gr.Markdown(
                "<div align='center'><font size='5'><a href='https://animationai.github.io/AnimateAnything'>Project Page</a> &ensp;"  # noqa
                "<a href='https://arxiv.org/abs/2311.12886'>Paper</a> &ensp;"
                "<a href='https://github.com/alibaba/animate-anything'>Code</a> &ensp;"  # noqa
            )

        with gr.Row(equal_height=True):
            with gr.Column():
                init_img = gr.ImageMask(label='Input Image', brush=gr.Brush(default_size=100))
                generate_button = gr.Button(
                    value="Generate", variant='primary')
            input_video = gr.Video(label="Input video", interactive=True)
            
            result_video = gr.Video(
                label="Generated Animation", interactive=False)

        with gr.Accordion('Advance Options', open=False):
            with gr.Row():
                fps_textbox = gr.Number(label="Fps", value=7, minimum=1)
                num_frames_textbox = gr.Number(label="Num frames", value=14, minimum=1, maximum=78)

            input_video.upload(
                fn=update_num_frames,
                inputs=[input_video],
                outputs=[num_frames_textbox]
            )
            
            motion_bucket_id_slider = gr.Slider(
                label='motion_bucket_id',
                value=127, step=1, minimum=0, maximum=511)

            sample_step_slider = gr.Slider(
                label="Sampling steps", value=25, minimum=10, maximum=100, step=1)

            with gr.Row():
                seed_textbox = gr.Textbox(label="Seed", value=-1)
                seed_button = gr.Button(
                    value="\U0001F3B2", elem_classes="toolbutton")
            seed_button.click(
                fn=lambda x: random.randint(1, 1e8),
                outputs=[seed_textbox],
                queue=False
            )



        generate_button.click(
            fn=controller.animate,
            inputs=[
                init_img,
                input_video,
                sample_step_slider,
                seed_textbox,
                fps_textbox,
                num_frames_textbox,
                motion_bucket_id_slider
            ],
            outputs=[result_video]
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
                )
