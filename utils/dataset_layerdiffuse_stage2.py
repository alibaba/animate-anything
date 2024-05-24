import os
import decord
import numpy as np
import random
import json
import torchvision
import torchvision.transforms as T
import torch
import cv2

from glob import glob
from PIL import Image
from itertools import islice
from pathlib import Path
from .bucketing import sensible_buckets
from .common import get_moved_area_mask, calculate_motion_score, get_full_white_area_mask

decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange, repeat

# Inspired by the VideoMAE repository.
def normalize_input(
    item, 
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225],
    use_simple_norm=True,
    is_alpha=False,
):
    if(is_alpha is True):
        # Normalize between 0 & 1
        return  item / 255.0
    
    if item.dtype == torch.uint8 and not use_simple_norm:
        item = rearrange(item, 'f c h w -> f h w c')
        
        item = item.float() / 255.0
        mean = torch.tensor(mean)
        std = torch.tensor(std)

        out = rearrange((item - mean) / std, 'f h w c -> f c h w')
        
        return out
    else:
        # Normalize between -1 & 1
        item = rearrange(item, 'f c h w -> f h w c')
        return  rearrange(item / 127.5 - 1.0, 'f h w c -> f c h w')
            
def get_prompt_ids(prompt, tokenizer):
    prompt_input = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
    )
    prompt_ids = prompt_input.input_ids[0]
    prompt_attention_mask = prompt_input.attention_mask[0]
    return prompt_ids, prompt_attention_mask

def read_caption_file(caption_file):
        with open(caption_file, 'r', encoding="utf8") as t:
            return t.read()

def get_text_prompt(
        text_prompt: str = '', 
        fallback_prompt: str= '',
        file_path:str = '', 
        ext_types=['.mp4'],
        use_caption=False
    ):
    try:
        if use_caption:
            if len(text_prompt) > 1: return text_prompt
            caption_file = ''
            # Use caption on per-video basis (One caption PER video)
            for ext in ext_types:
                maybe_file = file_path.replace(ext, '.txt')
                if maybe_file.endswith(ext_types): continue
                if os.path.exists(maybe_file): 
                    caption_file = maybe_file
                    break

            if os.path.exists(caption_file):
                return read_caption_file(caption_file)
            
            # Return fallback prompt if no conditions are met.
            return fallback_prompt

        return text_prompt
    except:  # noqa: E722
        print(f"Couldn't read prompt caption for {file_path}. Using fallback.")
        return fallback_prompt

    
def get_frame_batch(max_frames, sample_fps, vr, transform):
    native_fps = vr.get_avg_fps()
    max_range = len(vr)
    frame_step = max(1, round(native_fps / sample_fps))
    frame_range = range(0, max_range, frame_step)
    if len(frame_range) < max_frames:
        frame_range = np.linspace(0, max_range-1, max_frames).astype(int)
    start = random.randint(0, len(frame_range) - max_frames)
    #start = len(frame_range) - max_frames
    frame_range_indices = list(frame_range)[start:start+max_frames]
    frames = vr.get_batch(frame_range_indices)
    video = rearrange(frames, "f h w c -> f c h w")
    video = transform(video)
    return video


def process_video(vid_path, use_bucketing, w, h, get_frame_buckets, get_frame_batch):
    if use_bucketing:
        vr = decord.VideoReader(vid_path)
        resize = get_frame_buckets(vr)
        video = get_frame_batch(vr, resize=resize)

    else:
        vr = decord.VideoReader(vid_path)
        video = get_frame_batch(vr)

    return video, vr

# https://github.com/ExponentialML/Video-BLIP2-Preprocessor
class VideoBLIPDataset(Dataset):
    def __init__(
            self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 4,
            sample_start_idx: int = 0,
            fps: int = 1,
            json_path: str ="",
            json_data = None,
            vid_data_key: str = "video_path",
            preprocessed: bool = False,
            use_bucketing: bool = False,
            cache_latents = False,
            motion_threshold = 50,
            **kwargs
    ):
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.use_bucketing = use_bucketing
        self.tokenizer = tokenizer
        self.preprocessed = preprocessed
        
        self.vid_data_key = vid_data_key
        self.vid_data_key_alpha = 'alpha_video_path'
        self.vid_data_key_rgb = 'rgb_video_path'

        self.train_data = self.load_from_json(json_path, json_data)
        self.cache_latents = cache_latents
        self.motion_threshold = motion_threshold
        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.fps = fps
        self.transform = T.Compose([
            #T.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(width/height, width/height), antialias=True)
            T.Resize((height, width), antialias=True),
            T.CenterCrop([height, width])
        ])

    def build_json(self, json_data):
        extended_data = []
        for data in json_data['data']:
            for nested_data in data['data']:
                self.build_json_dict(
                    data, 
                    nested_data, 
                    extended_data
                )
        json_data = extended_data
        return json_data

    def build_json_dict(self, data, nested_data, extended_data):
        alpha_clip_path = nested_data['alpha_clip_path'] if 'alpha_clip_path' in nested_data else None
        rgb_clip_path = nested_data['rgb_clip_path'] if 'rgb_clip_path' in nested_data else None
        
        # print('data[self.vid_data_key_alpha]: ', data[self.vid_data_key_alpha])
        # print('data[self.vid_data_key_rgb]: ', data[self.vid_data_key_rgb])
        # input()

        extended_data.append({
            self.vid_data_key_alpha: data[self.vid_data_key_alpha],
            self.vid_data_key_rgb: data[self.vid_data_key_rgb],
            'frame_index': nested_data['frame_index'], ####### 
            'prompt': nested_data['prompt'],
            'alpha_clip_path': alpha_clip_path,
            'rgb_clip_path': rgb_clip_path
        })
        
    def load_from_json(self, path, json_data):
        try:
            with open(path) as jpath:
                json_data = json.load(jpath)
                 
                train_data = self.build_json(json_data)
                print(f"Loading VideoBLIP {len(train_data)} videos from {path}")
                return train_data

        except:
            print("Non-existant JSON path. Skipping.")
            return []
            
    def validate_json(self, base_path, path):
        return os.path.exists(f"{base_path}/{path}")

    def get_frame_buckets(self, vr):
        _, h, w = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize
    
    def fetch_video_frames_rgb(self, directory_path, n_sample_frames, sample_start_idx, transform, mode: str='jpg'):
        """
        directory_path: video dir composed of imgs, 00000.jpg/00005.jpg...
        mode: 'png' or 'jpg'
        """
        # print('fetch_video_frames_rgb.directory_path: ', directory_path)
        if(os.path.exists(directory_path) is False):
            raise FileNotFoundError(f'{directory_path} does not exist!')

        frame_paths = sorted(glob(f"{directory_path}/*.{mode}"), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        frame_paths = frame_paths[sample_start_idx:]
        max_range = len(frame_paths)
        # print("directory_path: ", directory_path)
        # print('sample_start_idx: ', sample_start_idx)

        assert max_range > 0

        # frame padding

        if max_range < n_sample_frames:
            frame_indices = np.linspace(0, max_range - 1, n_sample_frames).astype(int)
            sampled_paths = [frame_paths[idx] for idx in frame_indices]
        else:
            sampled_paths = frame_paths[ : n_sample_frames]


        frames = []

        for frame_path in sampled_paths:
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        frames = np.stack(frames)

        video = rearrange(frames, "f h w c -> f c h w")
        video = torch.from_numpy(video).float()
        video = transform(video)
        # print('video_rgb.shape: ', video.shape)
        return video

    def fetch_video_frames_alpha(self, directory_path, n_sample_frames, sample_start_idx, transform, mode: str='png'):
        """"
        similar to fetch_video_frames_rgb, except for num channels
        """
        if(os.path.exists(directory_path) is False):
            raise FileNotFoundError(f'{directory_path} does not exist!')

        frame_paths = sorted(glob(f"{directory_path}/*.{mode}"), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        frame_paths = frame_paths[sample_start_idx:]
        max_range = len(frame_paths)

        assert max_range > 0

        # frame padding
        if max_range < n_sample_frames:
            frame_indices = np.linspace(0, max_range - 1, n_sample_frames).astype(int)
            sampled_paths = [frame_paths[idx] for idx in frame_indices]
        else:
            sampled_paths = frame_paths[ : n_sample_frames]

        frames = []

        for frame_path in sampled_paths:
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
            frames.append(frame)

        frames = np.stack(frames) 
        # print('video_alpha.shape: ', frames.shape)
        # print('frames.shape: ', frames.shape)
        assert len(frames.shape)==3
        frames = np.expand_dims(frames, axis=3)  # [f, h, w, 1]
        # print('video_alpha.shape: ', frames.shape)
        video = rearrange(frames, "f h w c -> f c h w")
        video = torch.from_numpy(video).float()
        video = transform(video)


        # print('video_alpha.shape: ', video.shape)
        
        return video


    def train_data_batch(self, index):
        vid_data = self.train_data[index]

        prompt = vid_data['prompt']

        # print('vid_data: ', vid_data)

        alpha_clip_path = vid_data[self.vid_data_key_alpha]
        rgb_clip_path = vid_data[self.vid_data_key_rgb]

        # print('rgb_clip_path: ', rgb_clip_path)
        # print('alpha_clip_path:', alpha_clip_path)
        # print('self.sample_start_idx: ', self.sample_start_idx)
        self.sample_start_idx = vid_data['frame_index'] # 会导致caption都是第一帧
        # print('self.sample_start_idx: ', self.sample_start_idx)
        # input()
        # input()

        # print('train_data_batch.rgb_clip_path: ', rgb_clip_path)

        rgb_video = self.fetch_video_frames_rgb(rgb_clip_path, self.n_sample_frames, self.sample_start_idx, self.transform)
        alpha_video = self.fetch_video_frames_alpha(alpha_clip_path, self.n_sample_frames, self.sample_start_idx, self.transform)
        # print(rgb_video.shape[2], rgb_video.shape[3])
        # print(alpha_video.shape[2], alpha_video.shape[3])
        assert rgb_video.shape[2]==384 and rgb_video.shape[3]==384
        assert alpha_video.shape[2]==384 and alpha_video.shape[3]==384

        prompt_ids, prompt_mask = get_prompt_ids(prompt, self.tokenizer)

        # normalize RGB between -1&1, alpha 0&1, referenced from paper Layerdiffuse
        
        example = {
            "pixel_values_rgb": normalize_input(rgb_video),  # [f, c, h, w]
            "pixel_values_alpha": normalize_input(alpha_video, is_alpha=True), # [f, 1, h, w]
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "text_prompt": prompt,
            'dataset': self.__getname__(),
        }
        # TODO: gaussian filter
        # mask = get_moved_area_mask(rgb_video.permute([0,2,3,1]).numpy())
        mask = get_full_white_area_mask(rgb_video.permute([0, 2, 3, 1]).numpy())
        example['mask'] = mask
        # example['motion'] = calculate_motion_score(rgb_video.permute([0, 2, 3, 1]).numpy())

        return example

    @staticmethod
    def __getname__(): return 'video_blip'

    def __len__(self):
        if self.train_data is not None:
            return len(self.train_data)
        else: 
            return 0

    def __getitem__(self, index):
        example = self.train_data_batch(index)
        # if example['motion'] < self.motion_threshold or example['mask'] is None:
        #     return self.__getitem__(random.randint(0, len(self)-1))
        return example


class SingleVideoDataset(Dataset):
    def __init__(
        self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 4,
            frame_step: int = 1,
            single_video_path: str = "",
            single_video_prompt: str = "",
            use_caption: bool = False,
            use_bucketing: bool = False,
            **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing
        self.frames = []
        self.index = 1

        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.n_sample_frames = n_sample_frames
        self.frame_step = frame_step

        self.single_video_path = single_video_path
        self.single_video_prompt = single_video_prompt

        self.width = width
        self.height = height
    def create_video_chunks(self):
        # Create a list of frames separated by sample frames
        # [(1,2,3), (4,5,6), ...]
        vr = decord.VideoReader(self.single_video_path)
        vr_range = range(1, len(vr), self.frame_step)

        self.frames = list(self.chunk(vr_range, self.n_sample_frames))

        # Delete any list that contains an out of range index.
        for i, inner_frame_nums in enumerate(self.frames):
            for frame_num in inner_frame_nums:
                if frame_num > len(vr):
                    print(f"Removing out of range index list at position: {i}...")
                    del self.frames[i]

        return self.frames

    def chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def get_frame_batch(self, vr, resize=None):
        index = self.index
        frames = vr.get_batch(self.frames[self.index])
        video = rearrange(frames, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video

    def get_frame_buckets(self, vr):
        _, h, w = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize
    
    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch
            )
        
        return video, vr 

    @staticmethod
    def __getname__(): return 'single_video'

    def __len__(self):
        
        return len(self.create_video_chunks())

    def __getitem__(self, index):
        train_data = self.single_video_path
        self.index = index

        if train_data.endswith(self.vid_types):
            video, _ = self.process_video_wrapper(train_data)

            prompt = self.single_video_prompt
            prompt_ids, prompt_mask = get_prompt_ids(prompt, self.tokenizer)
        else:
            raise ValueError(f"Single video is not a video type. Types: {self.vid_types}")
 
        example = {
            "pixel_values": normalize_input(video),
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "text_prompt": prompt,
            'dataset': self.__getname__()
        }

        return example
    
class ImageDataset(Dataset):
    
    def __init__(
        self,
        image_json,
        tokenizer = None,
        width: int = 256,
        height: int = 256,
        base_width: int = 256,
        base_height: int = 256,
        use_caption:     bool = False,
        image_dir: str = '',
        single_img_prompt: str = '',
        use_bucketing: bool = False,
        fallback_prompt: str = '',
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.img_types = (".png", ".jpg", ".jpeg", '.bmp')
        self.use_bucketing = use_bucketing
        self.image_dir = image_dir
        if image_json is None:
            image_json = []
            for entry in os.scandir(image_dir):
                if entry.name.endswith('.txt'):
                    prompt = open(os.path.join(image_dir, entry.name)).readline()
                    image_json.append({'prompt': prompt, 'path': entry.name.replace('txt', 'jpg')})
        elif type(image_json) == str:
            self.image_json = json.load(open(image_json))
        else:
            self.image_json = image_json
        print(f'load {len(self.image_json)} from image {image_dir}')
        self.fallback_prompt = fallback_prompt

        self.use_caption = use_caption
        self.single_img_prompt = single_img_prompt

        self.width = width
        self.height = height
        self.transform = T.Compose([
            #T.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(width/height, width/height), antialias=True),
            T.Resize(min(height, width), antialias=True),
            T.CenterCrop([height, width])
        ])


    @staticmethod
    def __getname__(): return 'image'
    
    def __len__(self):
        # Image directory
        return len(self.image_json)

    def __getitem__(self, index):
        train_data = self.image_json[index]
        img, prompt = train_data['path'], train_data['prompt']
        img = os.path.join(self.image_dir, img)
        try:
            img = torchvision.io.read_image(img, mode=torchvision.io.ImageReadMode.RGB)
        except:
            img = T.transforms.PILToTensor()(Image.open(img).convert("RGB"))

        width = self.width
        height = self.height

        if self.use_bucketing:
            _, h, w = img.shape
            width, height = sensible_buckets(width, height, w, h)
              
        img = self.transform(img) 
        img = repeat(img, 'c h w -> f c h w', f=1)
        prompt_ids, prompt_mask = get_prompt_ids(prompt, self.tokenizer)
        example = {
            "pixel_values": normalize_input(img),
            "frames": img,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "text_prompt": prompt, 
            'dataset': self.__getname__()
        }

        return example

class VideoFolderDataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        width: int = 256,
        height: int = 256,
        n_sample_frames: int = 16,
        fps: int = 8,
        path: str = "./data",
        fallback_prompt: str = "",
        use_bucketing: bool = False,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing

        self.fallback_prompt = fallback_prompt

        self.video_files = glob(f"{path}/*.mp4")

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.fps = fps

    def get_frame_buckets(self, vr):
        _, h, w = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize

    def get_frame_batch(self, vr, resize=None):
        n_sample_frames = self.n_sample_frames
        native_fps = vr.get_avg_fps()
        
        every_nth_frame = max(1, round(native_fps / self.fps))
        every_nth_frame = min(len(vr), every_nth_frame)
        
        effective_length = len(vr) // every_nth_frame
        if effective_length < n_sample_frames:
            n_sample_frames = effective_length
            raise RuntimeError("not enough frames")

        effective_idx = random.randint(0, (effective_length - n_sample_frames))
        idxs = every_nth_frame * np.arange(effective_idx, effective_idx + n_sample_frames)

        video = vr.get_batch(idxs)
        video = rearrange(video, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video, vr
        
    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch
            )
        return video, vr
    
    @staticmethod
    def __getname__(): return 'folder'

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        try:
            video, _ = self.process_video_wrapper(self.video_files[index])
        except Exception as err:
            print("read video error", self.video_files[index])
            video, _ = self.process_video_wrapper(self.video_files[index+1])

        if os.path.exists(self.video_files[index].replace(".mp4", ".txt")):
            with open(self.video_files[index].replace(".mp4", ".txt"), "r") as f:
                lines = f.readlines()
                prompt = random.choice(lines)
        else:
            prompt = self.fallback_prompt

        prompt_ids, prompt_mask = get_prompt_ids(prompt, self.tokenizer)

        return {"pixel_values": normalize_input(video[0]), "frames": video[0],
                "prompt_ids": prompt_ids, "prompt_mask": prompt_mask, "text_prompt": prompt, 'dataset': self.__getname__()}

class VideoJsonDataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        width: int = 256,
        height: int = 256,
        n_sample_frames: int = 16,
        fps: int = 8,
        video_dir: str = "./data",
        video_json: str = "",
        fallback_prompt: str = "",
        use_bucketing: bool = False,
        cache_latents = False,
        motion_threshold = 50,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing

        self.fallback_prompt = fallback_prompt
        self.video_dir = video_dir
        self.video_files = json.load(open(video_json))
        print(f'load {len(self.video_files)} from video {video_dir}')
        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.fps = fps
        self.cache_latents = cache_latents
        self.motion_threshold = motion_threshold
        self.transform = T.Compose([
            #T.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(width/height, width/height), antialias=True),
            T.Resize(min(height, width), antialias=True),
            T.CenterCrop([height, width])
        ])


    def get_frame_buckets(self, vr):
        _, h, w = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize

       
    @staticmethod
    def __getname__(): return 'video_json'

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        mask = None
        try:
            item = self.video_files[index]
            video_path = os.path.join(self.video_dir, item['video'])
            cache_path = os.path.splitext(video_path)[0] + '.pt'
            if os.path.exists(cache_path):
                if self.cache_latents == 'clear':
                    os.remove(cache_path)
                elif self.cache_latents == 'load':
                    return torch.load(cache_path, map_location='cpu')

            prompt = item['caption']
            if self.fallback_prompt == "<no_text>":
                prompt = ""
            vr = decord.VideoReader(video_path)
            video = get_frame_batch(self.n_sample_frames, self.fps, vr, self.transform)
        except Exception as err:
            print("read video error", err, video_path)
            return self.__getitem__(index+1)
        prompt_ids, prompt_mask = get_prompt_ids(prompt, self.tokenizer)

        example = {
            "pixel_values": normalize_input(video), 
            "prompt_ids": prompt_ids, 
            "prompt_mask": prompt_mask,
            "text_prompt": prompt, 
            'cache_path': cache_path,
            'dataset': self.__getname__()
        }
        example['mask'] = get_moved_area_mask(video.permute([0,2,3,1]).numpy())
        example['motion'] = calculate_motion_score(video.permute([0,2,3,1]).numpy())
        if example['motion'] < self.motion_threshold or example['mask'] is None:
            return self.__getitem__(random.randint(0, len(self)-1))
        return example

class CachedDataset(Dataset):
    def __init__(self,cache_dir: str = ''):
        self.cache_dir = cache_dir
        self.cached_data_list = self.get_files_list()

    def get_files_list(self):
        tensors_list = [f"{self.cache_dir}/{x}" for x in os.listdir(self.cache_dir) if x.endswith('.pt')]
        return sorted(tensors_list)

    def __len__(self):
        return len(self.cached_data_list)

    def __getitem__(self, index):
        cached_latent = torch.load(self.cached_data_list[index], map_location='cuda:0')
        return cached_latent

def get_train_dataset(dataset_types, train_data, tokenizer, cache_latents=False):
    train_datasets = []
    dataset_cls = [VideoJsonDataset, SingleVideoDataset, ImageDataset, VideoFolderDataset, VideoBLIPDataset]
    dataset_map = {d.__getname__(): d for d in dataset_cls}

    # Loop through all available datasets, get the name, then add to list of data to process.
    for dataset in dataset_types:
        if dataset in dataset_map:
            train_datasets.append(dataset_map[dataset](**train_data, tokenizer=tokenizer, cache_latents=cache_latents))
        else:
            raise ValueError(f"Dataset type not found: {dataset} not in {dataset_map.keys()}")
    return train_datasets

def get_empty_image_dataset():
    return ImageDataset([])

def extend_datasets(datasets, dataset_items, extend=False):
    biggest_data_len = max(x.__len__() for x in datasets)
    extended = []
    for dataset in datasets:
        if dataset.__len__() == 0:
            del dataset
            continue
        if dataset.__len__() < biggest_data_len:
            for item in dataset_items:
                if extend and item not in extended and hasattr(dataset, item):
                    print(f"Extending {item}")

                    value = getattr(dataset, item)
                    value *= biggest_data_len
                    value = value[:biggest_data_len]

                    setattr(dataset, item, value)

                    print(f"New {item} dataset length: {dataset.__len__()}")
                    extended.append(item)
