"""
Based on the implementation from:
https://huggingface.co/spaces/fffiloni/lama-video-watermark-remover/tree/main

Modules were adapted by Hans Brouwer to only support the final configuration of the model uploaded here:
https://huggingface.co/akhaliq/lama

Apache License 2.0: https://github.com/advimman/lama/blob/main/LICENSE

@article{suvorov2021resolution,
  title={Resolution-robust Large Mask Inpainting with Fourier Convolutions},
  author={Suvorov, Roman and Logacheva, Elizaveta and Mashikhin, Anton and Remizova, Anastasia and Ashukha, Arsenii and Silvestrov, Aleksei and Kong, Naejin and Goka, Harshith and Park, Kiwoong and Lempitsky, Victor},
  journal={arXiv preprint arXiv:2109.07161},
  year={2021}
}
"""

import os
import sys
from urllib.request import urlretrieve

import torch
from einops import rearrange
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from train import export_to_video


LAMA_URL = "https://huggingface.co/akhaliq/lama/resolve/main/best.ckpt"
LAMA_PATH = "models/lama.ckpt"


def download_progress(t):
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def download(url, path):
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=path) as t:
        urlretrieve(url, filename=path, reporthook=download_progress(t), data=None)


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]

        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        # (batch,c, t, h, w/2+1, 2)
        ffted = ffted.view((batch, -1, 2) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm="ortho")

        return output


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(SpectralTransform, self).__init__()
        self.stride = stride
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        return output


class FFC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin,
        ratio_gout,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_type="reflect",
        gated=False,
    ):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(
            in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type
        )
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(
            in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type
        )
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(
            in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type
        )
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin=0,
        ratio_gout=0,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
    ):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias
        )
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, ratio_gin, ratio_gout):
        super().__init__()
        self.conv1 = FFC_BN_ACT(
            dim, dim, kernel_size=3, padding=1, dilation=1, ratio_gin=ratio_gin, ratio_gout=ratio_gout
        )
        self.conv2 = FFC_BN_ACT(
            dim, dim, kernel_size=3, padding=1, dilation=1, ratio_gin=ratio_gin, ratio_gout=ratio_gout
        )

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class LargeMaskInpainting(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=18, max_features=1024):
        super().__init__()

        model = [nn.ReflectionPad2d(3), FFC_BN_ACT(input_nc, ngf, kernel_size=7)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                FFC_BN_ACT(
                    min(max_features, ngf * mult),
                    min(max_features, ngf * mult * 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    ratio_gout=0.75 if i == n_downsampling - 1 else 0,
                )
            ]

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(min(max_features, ngf * 2**n_downsampling), ratio_gin=0.75, ratio_gout=0.75)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    min(max_features, ngf * mult),
                    min(max_features, int(ngf * mult / 2)),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(min(max_features, int(ngf * mult / 2))),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, img, mask):
        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)
        pred = self.model(masked_img)
        inpainted = mask * pred + (1 - mask) * img
        return inpainted


@torch.inference_mode()
def inpaint_watermark(imgs):
    if not os.path.exists(LAMA_PATH):
        download(LAMA_URL, LAMA_PATH)

    mask = to_tensor(Image.open("./utils/mask.png").convert("L")).unsqueeze(0).to(imgs.device)
    if mask.shape[-1] != imgs.shape[-1]:
        mask = F.interpolate(mask, size=(imgs.shape[2], imgs.shape[3]), mode="nearest")
    mask = mask.expand(imgs.shape[0], 1, mask.shape[2], mask.shape[3])

    model = LargeMaskInpainting().to(imgs.device)
    state_dict = torch.load(LAMA_PATH, map_location=imgs.device)["state_dict"]
    g_dict = {k.replace("generator.", ""): v for k, v in state_dict.items() if k.startswith("generator")}
    model.load_state_dict(g_dict)

    inpainted = model.forward(imgs, mask)

    return inpainted


if __name__ == "__main__":
    import decord

    decord.bridge.set_bridge("torch")

    if len(sys.argv) < 2:
        print("Usage: python -m utils.lama <path/to/video>")
        sys.exit(1)

    video_path = sys.argv[1]
    out_path = video_path.replace(".mp4", " inpainted.mp4")

    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    video = rearrange(vr[:], "f h w c -> f c h w").div(255)

    inpainted = inpaint_watermark(video)
    inpainted = rearrange(inpainted, "f c h w -> f h w c").clamp(0, 1).mul(255).byte().cpu().numpy()
    export_to_video(inpainted, out_path, fps)
