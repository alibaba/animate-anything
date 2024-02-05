import cv2
import json
from PIL import Image
import torch
import random
import numpy as np
import torchvision.transforms as T
from einops import rearrange, repeat
import imageio
import sys

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.mode()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents

def DDPM_forward(x0, step, num_frames, scheduler):
    device = x0.device
    t = scheduler.timesteps[-1]
    xt = repeat(x0, 'b c 1 h w -> b c f h w', f = num_frames)

    eps = torch.randn_like(xt)
    alpha_vec = torch.prod(scheduler.alphas[t:])
    xt = torch.sqrt(alpha_vec) * xt + torch.sqrt(1-alpha_vec) * eps
    return xt, None

def DDPM_forward_timesteps(x0, step, num_frames, scheduler):
    '''larger step -> smaller t -> smaller alphas[t:] -> smaller xt -> smaller x0'''

    device = x0.device
    # timesteps are reversed
    timesteps = scheduler.timesteps[len(scheduler.timesteps)-step:]
    t = timesteps[0]

    if x0.shape[2] == 1:
        xt = repeat(x0, 'b c 1 h w -> b c f h w', f = num_frames)
    else:
        xt = x0
    noise = torch.randn(xt.shape, dtype=xt.dtype, device=device)
    # t to tensor of batch size 
    t = torch.tensor([t]*xt.shape[0], device=device)
    xt = scheduler.add_noise(xt, noise, t)
    return xt, timesteps

def DDPM_forward_mask(x0, step, num_frames, scheduler, mask):
    '''larger step -> smaller t -> smaller alphas[t:] -> smaller xt -> smaller x0'''
    device = x0.device
    dtype = x0.dtype
    b, c, f, h, w = x0.shape

    move_xt, timesteps = DDPM_forward_timesteps(x0, step, num_frames, scheduler)
    mask = T.ToTensor()(mask).to(dtype).to(device)
    mask = T.Resize([h, w], antialias=False)(mask)
    mask = rearrange(mask, 'b h w -> b 1 1 h w')
    freeze_xt = repeat(x0, 'b c 1 h w -> b c f h w', f = num_frames)
    initial = freeze_xt * (1-mask) + move_xt * mask
    return initial, timesteps

def read_video(video_path, frame_number=-1):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    if frame_number == -1:
        frame_number = count
    else:
        frame_number = min(frame_number, count)
    frames = []
    for i in range(frame_number):
        ret, ref_frame = cap.read()
        ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        if not ret:
            raise ValueError("Failed to read video file")
        frames.append(ref_frame)
    return frames


def get_moved_area_mask(frames, move_th=5, th=-1):
    ref_frame = frames[0] 
    # Convert the reference frame to gray
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = ref_gray
    # Initialize the total accumulated motion mask
    total_mask = np.zeros_like(ref_gray)

    # Iterate through the video frames
    for i in range(1, len(frames)):
        frame = frames[i]
        # Convert the frame to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between the reference frame and the current frame
        diff = cv2.absdiff(ref_gray, gray)
        #diff += cv2.absdiff(prev_gray, gray)

        # Apply a threshold to obtain a binary image
        ret, mask = cv2.threshold(diff, move_th, 255, cv2.THRESH_BINARY)

        # Accumulate the mask
        total_mask = cv2.bitwise_or(total_mask, mask)

        # Update the reference frame
        prev_gray = gray

    contours, _ = cv2.findContours(total_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    ref_mask = np.zeros_like(ref_gray)
    ref_mask = cv2.drawContours(ref_mask, contours, -1, (255, 255, 255), -1)
    for cnt in contours:
        cur_rec = cv2.boundingRect(cnt)
        rects.append(cur_rec) 

    #rects = merge_overlapping_rectangles(rects)
    mask = np.zeros_like(ref_gray)
    if th < 0:
        h, w = mask.shape
        th = int(h*w*0.005)
    for rect in rects:
        x, y, w, h = rect
        if w*h < th:
            continue
        #ref_frame = cv2.rectangle(ref_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        mask[y:y+h, x:x+w] = 255
    return mask

def calculate_motion_precision(frames, mask):
    moved_mask = get_moved_area_mask(frames, move_th=20, th=0)
    moved = moved_mask == 255
    gt = mask == 255
    precision = np.sum(moved & gt) / np.sum(moved)
    return precision

def check_overlap(rect1, rect2):
    # Calculate the coordinates of the edges of the rectangles
    rect1_left = rect1[0]
    rect1_right = rect1[0] + rect1[2]
    rect1_top = rect1[1]
    rect1_bottom = rect1[1] + rect1[3]

    rect2_left = rect2[0]
    rect2_right = rect2[0] + rect2[2]
    rect2_top = rect2[1]
    rect2_bottom = rect2[1] + rect2[3]

    # Check if the rectangles overlap
    if (rect2_left >= rect1_right or rect2_right <= rect1_left or
        rect2_top >= rect1_bottom or rect2_bottom <= rect1_top):
        return False
    else:
        return True

def merge_rects(rect1, rect2):
    left = min(rect1[0], rect2[0])
    top = min(rect1[1], rect2[1])
    right = max(rect1[0]+rect1[2], rect2[0]+rect2[2])
    bottom = max(rect1[1]+rect1[3], rect2[1]+rect2[3])
    width = right - left
    height = bottom - top
    return (left, top, width, height)

def merge_overlapping_rectangles(rectangles):
    # Sort the rectangles based on their left coordinate
    sorted_rectangles = sorted(rectangles, key=lambda x: x[0])

    # Initialize an empty list to store the merged rectangles
    merged_rectangles = []

    # Iterate through the sorted rectangles and merge them
    for rect in sorted_rectangles:
        if not merged_rectangles:
            # If the merged rectangles list is empty, add the first rectangle to it
            merged_rectangles.append(rect)
        else:
            # Get the last merged rectangle
            last_merged = merged_rectangles[-1]

            # Check if the current rectangle overlaps with the last merged rectangle
            if last_merged[0] + last_merged[2] >= rect[0]:
                # Merge the rectangles if they overlap
                merged_rectangles[-1] = (
                    min(last_merged[0], rect[0]),
                    min(last_merged[1], rect[1]),
                    max(last_merged[0] + last_merged[2], rect[0] + rect[2]) - min(last_merged[0], rect[0]),
                    max(last_merged[1] + last_merged[3], rect[1] + rect[3]) - min(last_merged[1], rect[1])
                )
            else:
                # Add the current rectangle to the merged rectangles list if they don't overlap
                merged_rectangles.append(rect)

    return merged_rectangles

def generate_random_mask(image):
    # Create a blank mask with the same size as the image
    b, c , h, w = image.shape
    mask = np.zeros([b, h, w], dtype=np.uint8)
    
    # Generate random coordinates for the mask
    num_points = np.random.randint(3, 10)  # Randomly choose the number of points to generate
    points = np.random.randint(0, min(h, w), size=(num_points, 2))  # Randomly generate the points
    # Draw a filled polygon on the mask using the random points
    for i in range(b):
        width = random.randint(w//4, w)
        height = random.randint(h//4, h)
        x = random.randint(0, w-width)
        y = random.randint(0, h-height)
        points=np.array([[x, y], [x+width, y], [x+width, y+height], [x, y+height]])
        mask[i] = cv2.fillPoly(mask[i], [points], 255)
    
    # Apply the mask to the image
    #masked_image = cv2.bitwise_and(image, image, mask=mask)
    return mask    

def generate_center_mask(image):
    # Create a blank mask with the same size as the image
    b, c , h, w = image.shape
    mask = np.zeros([b, h, w], dtype=np.uint8)
    
    # Generate random coordinates for the mask
    for i in range(b):
        width = int(w/10)
        height = int(h/10)
        mask[i][height:-height,width:-width] = 255
    # Apply the mask to the image
    #masked_image = cv2.bitwise_and(image, image, mask=mask)
    return mask    

def read_mask(json_path, label=["mask"]):
    j = json.load(open(json_path))    
    if type(label) != list:
        labels = [label]
    height = j['imageHeight']
    width = j['imageWidth']
    mask = np.zeros([height, width], dtype=np.uint8)
    for shape in j['shapes']:
        if shape['label'] in label:
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            mask[int(y1):int(y2), int(x1):int(x2)] = 255
    return mask


def slerp(z1, z2, alpha):
    theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
    return (
        torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
        + torch.sin(alpha * theta) / torch.sin(theta) * z2
    )

def _detect_edges(lum: np.ndarray, kernel_size=5) -> np.ndarray:
    """Detect edges using the luma channel of a frame.

    Arguments:
        lum: 2D 8-bit image representing the luma channel of a frame.

    Returns:
        2D 8-bit image of the same size as the input, where pixels with values of 255
        represent edges, and all other pixels are 0.
    """
    # Initialize kernel.
    #kernel_size = _estimated_kernel_size(lum.shape[1], lum.shape[0])
    _kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Estimate levels for thresholding.
    # TODO(0.6.3): Add config file entries for sigma, aperture/kernel size, etc.
    sigma: float = 1.0 / 3.0
    median = np.median(lum)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))

    # Calculate edges using Canny algorithm, and reduce noise by dilating the edges.
    # This increases edge overlap leading to improved robustness against noise and slow
    # camera movement. Note that very large kernel sizes can negatively affect accuracy.
    edges = cv2.Canny(lum, low, high)
    return cv2.dilate(edges, _kernel)


def _mean_pixel_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the mean average distance in pixel values between `left` and `right`.
    Both `left and `right` should be 2 dimensional 8-bit images of the same shape.
    """
    assert len(left.shape) == 2 and len(right.shape) == 2
    assert left.shape == right.shape
    num_pixels: float = float(left.shape[0] * left.shape[1])
    return (np.sum(np.abs(left.astype(np.int32) - right.astype(np.int32))) / num_pixels)

def calculate_latent_motion_score(latents):
    #latents b, c f, h, w
    diff=torch.abs(latents[:,:,1:]-latents[:,:,:-1])
    motion_score = torch.sum(torch.mean(diff, dim=[2,3,4]), dim=1) * 10
    return motion_score

def motion_mask_loss(latents, mask):
    diff = torch.abs(latents[:,:,1:] - latents[:,:,:-1])
    loss = torch.sum(torch.mean(diff * (1-mask), dim=[2,3,4]), dim=1)
    return loss

def calculate_motion_score(frame_imgs, calculate_edges=False, color="RGB") -> float:
    # Convert image into HSV colorspace.
    _last_frame = None

    _weights = [1.0, 1.0, 1.0, 0.0]
    score = 0
    for frame_img in frame_imgs:
        if color == "RGB":
            hue, sat, lum = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_RGB2HSV))
        else:
            hue, sat, lum = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))
        # Performance: Only calculate edges if we have to.
        edges = _detect_edges(lum) if calculate_edges else None
        if _last_frame == None:
            _last_frame = (hue, sat, lum, edges)
            continue

        score_components = [
            _mean_pixel_distance(hue, _last_frame[0]),
            _mean_pixel_distance(sat, _last_frame[1]),
            _mean_pixel_distance(lum, _last_frame[2]),
            0.0 if edges is None else _mean_pixel_distance(edges, _last_frame[3]),
        ]

        frame_score: float = (
            sum(component * weight for (component, weight) in zip(score_components, _weights))
            / sum(abs(weight) for weight in _weights))
        score += frame_score
        _last_frame = (hue, sat, lum, edges)

    return round(score/(len(frame_imgs)-1) * 10)

if __name__ == "__main__":
    
    # Example usage
    video_paths = [
        "/data/video/animate2/Bleach.Sennen.Kessen.Hen.S01E01.2022.1080p.WEB-DL.x264.AAC-DDHDTV-Scene-002.mp4",
        "/data/video/animate2/Evangelion.3.0.1.01.Thrice.Upon.A.Time.2021.BLURAY.720p.BluRay.x264.AAC-[YTS.MX]-Scene-0780.mp4",
        "/data/video/animate2/[GM-Team][国漫][永生 第2季][IMMORTALITY Ⅱ][2023][09][AVC][GB][1080P]-Scene-180.mp4",
        "/data/video/animate2/[orion origin] Legend of the Galactic Heroes Die Neue These [07] [WebRip 1080p] [H265 AAC] [GB]-Scene-048.mp4",
        "/data/video/MSRVTT/videos/all/video33.mp4",
        "/webvid/webvid/data/videos/000001_000050/1066692580.mp4",
        "/webvid/webvid/data/videos/000001_000050/1066685533.mp4",
        "/webvid/webvid/data/videos/000001_000050/1066685548.mp4",
        "/webvid/webvid/data/videos/000001_000050/1066676380.mp4",
        "/webvid/webvid/data/videos/000001_000050/1066676377.mp4",
    ]
    for i, video_path in enumerate(video_paths[:5]):
        frames = read_video(video_path, 200)[::3]
        if sys.argv[1] == 'test_mask':
            mask = get_moved_area_mask(frames)
            Image.fromarray(mask).save(f"output/mask/{i}.jpg")
            imageio.mimwrite(f"output/mask/{i}.gif", frames, duration=125, loop=0)
        elif sys.argv[1] == 'test_motion':
            for r in range(0, len(frames), 16):
                video_frames = frames[r:r+16]
                video_frames = [cv2.resize(f, (512, 512)) for f in video_frames]
                score = calculate_motion_score(video_frames, calculate_edges=False, color="BGR")
                imageio.mimwrite(f"output/example_video/{i}_{r}_{score}.mp4", video_frames, fps=8)
        elif sys.argv[1] == 'to_gif':
            imageio.mimwrite(f"output/example_video/{i}.gif", frames, duration=125, loop=0)
