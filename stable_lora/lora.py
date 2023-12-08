import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import loralib as loralb
from loralib import LoRALayer
import math
import json

from torch.utils.data import ConcatDataset
from transformers import CLIPTokenizer

try:
    from safetensors.torch import save_file, load_file
except: 
    print("Safetensors is not installed. Saving while using use_safetensors will fail.")

UNET_REPLACE = ["Transformer2DModel", "ResnetBlock2D"]
TEXT_ENCODER_REPLACE = ["CLIPAttention", "CLIPTextEmbeddings"]

UNET_ATTENTION_REPLACE = ["CrossAttention"]
TEXT_ENCODER_ATTENTION_REPLACE = ["CLIPAttention", "CLIPTextEmbeddings"]

"""
Copied from: https://github.com/cloneofsimo/lora/blob/bdd51b04c49fa90a88919a19850ec3b4cf3c5ecd/lora_diffusion/lora.py#L189
"""
def find_modules(
        model,
        ancestor_class= None,
        search_class = [torch.nn.Linear],
        exclude_children_of = [loralb.Linear, loralb.Conv2d, loralb.Embedding],
    ):
        """
        Find all modules of a certain class (or union of classes) that are direct or
        indirect descendants of other modules of a certain class (or union of classes).

        Returns all matching modules, along with the parent of those moduless and the
        names they are referenced by.
        """

        # Get the targets we should replace all linears under
        if ancestor_class is not None:
            ancestors = (
                module
                for module in model.modules()
                if module.__class__.__name__ in ancestor_class
            )
        else:
            # this, incase you want to naively iterate over all modules.
            ancestors = [module for module in model.modules()]

        # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
        for ancestor in ancestors:
            for fullname, module in ancestor.named_modules():
                if any([isinstance(module, _class) for _class in search_class]):
                    # Find the direct parent if this is a descendant, not a child, of target
                    *path, name = fullname.split(".")
                    parent = ancestor
                    while path:
                        parent = parent.get_submodule(path.pop(0))
                    # Skip this linear if it's a child of a LoraInjectedLinear
                    if exclude_children_of and any(
                        [isinstance(parent, _class) for _class in exclude_children_of]
                    ):
                        continue
                    # Otherwise, yield it
                    yield parent, name, module

class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x, 
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return nn.Conv2d.forward(self, x)

class Conv3d(nn.Conv3d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Conv3d.__init__(self, in_channels, out_channels, (kernel_size, 1, 1), **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters

        # Get view transform shape
        i, o, k = self.weight.shape[:3]
        self.view_shape = (i, o, k, kernel_size, 1)
        self.force_disable_merge = True

        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv3d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv3d.train(self, mode)

        # HACK Merging the weights this way could potentially cause vanishing gradients if validation is enabled.
        # If you are to save this as a pretrained model, you will have to merge these weights afterwards, then save.
        if self.force_disable_merge:
            return
            
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.weight.data -= torch.mean((self.lora_B @ self.lora_A).view(self.view_shape), dim=-2,  keepdim=True) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                self.weight.data += torch.mean((self.lora_B @ self.lora_A).view(self.view_shape), dim=-2,  keepdim=True) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv3d(
                x, 
                self.weight + torch.mean((self.lora_B @ self.lora_A).view(self.view_shape), dim=-2,  keepdim=True) * \
                    self.scaling, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return nn.Conv3d.forward(self, x)

def create_lora_linear(child_module, r, dropout=0, bias=False, scale=0):
    return loralb.Linear(
        child_module.in_features, 
        child_module.out_features, 
        merge_weights=False,
        bias=bias,
        lora_dropout=dropout,
        lora_alpha=r,
        r=r
    )
    return lora_linear

def create_lora_conv(child_module, r, dropout=0, bias=False, rescale=False, scale=0):
    return Conv2d(
        child_module.in_channels, 
        child_module.out_channels,
        kernel_size=child_module.kernel_size[0],
        padding=child_module.padding,
        stride=child_module.stride,
        merge_weights=False,
        bias=bias,
        lora_dropout=dropout,
        lora_alpha=r,
        r=r,
    )
    return lora_conv    

def create_lora_conv3d(child_module, r, dropout=0, bias=False, rescale=False, scale=0):
    return Conv3d(
        child_module.in_channels, 
        child_module.out_channels,
        kernel_size=child_module.kernel_size[0],
        padding=child_module.padding,
        stride=child_module.stride,
        merge_weights=False,
        bias=bias,
        lora_dropout=dropout,
        lora_alpha=r,
        r=r,
    )
    return lora_conv  

def create_lora_emb(child_module, r):
    return loralb.Embedding(
        child_module.num_embeddings, 
        child_module.embedding_dim, 
        merge_weights=False,
        lora_alpha=r,
        r=r
    )

def activate_lora_train(model, bias):
    def unfreeze():
        print(model.__class__.__name__ + " LoRA set for training.")
        return loralb.mark_only_lora_as_trainable(model, bias=bias)

    return unfreeze

def add_lora_to(
    model, 
    target_module=UNET_REPLACE, 
    search_class=[torch.nn.Linear], 
    r=32, 
    dropout=0,
    lora_bias='none'
):
    for module, name, child_module in find_modules(
        model, 
        ancestor_class=target_module, 
        search_class=search_class
    ):
        bias = hasattr(child_module, "bias")
        
        # Check if child module of the model has bias.
        if bias:
            if child_module.bias is None:
                bias = False

        # Check if the child module of the model is type Linear or Conv2d.
        if isinstance(child_module, torch.nn.Linear):
            l = create_lora_linear(child_module, r, dropout, bias=bias)

        if isinstance(child_module, torch.nn.Conv2d):
            l = create_lora_conv(child_module, r, dropout, bias=bias)

        if isinstance(child_module, torch.nn.Conv3d):
            l = create_lora_conv3d(child_module, r, dropout, bias=bias)

        if isinstance(child_module, torch.nn.Embedding):
            l = create_lora_emb(child_module, r)
            
        # If the model has bias and we wish to add it, use the child_modules in place
        if bias:
            l.bias = child_module.bias
        
        # Assign the frozen weight of model's Linear or Conv2d to the LoRA model.
        l.weight =  child_module.weight

        # Replace the new LoRA model with the model's Linear or Conv2d module.
        module._modules[name] = l
        

    # Unfreeze only the newly added LoRA weights, but keep the model frozen.
    return activate_lora_train(model, lora_bias)

def save_lora(
        unet=None, 
        text_encoder=None, 
        save_text_weights=False,
        output_dir="output",
        lora_filename="lora.safetensors",
        lora_bias='none', 
        save_for_webui=True,
        only_webui=False,
        metadata=None,
        unet_dict_converter=None,
        text_dict_converter=None
    ):

        if not only_webui:
            # Create directory for the full LoRA weights.
            trainable_weights_dir = f"{output_dir}/full_weights"
            lora_out_file_full_weight = f"{trainable_weights_dir}/{lora_filename}"
            os.makedirs(trainable_weights_dir, exist_ok=True)

        ext = '.safetensors'
        # Create LoRA out filename.
        lora_out_file = f"{output_dir}/webui_{lora_filename}{ext}"

        if not only_webui:
            save_path_full_weights = lora_out_file_full_weight + ext

        save_path = lora_out_file

        if not only_webui:
            for i, model in enumerate([unet, text_encoder]):
                if save_text_weights and i == 1:
                    non_webui_weights = save_path_full_weights.replace(ext, f"_text_encoder{ext}")

                else:
                    non_webui_weights = save_path_full_weights.replace(ext, f"_unet{ext}")

                # Load only the LoRAs from the state dict.
                lora_dict = loralb.lora_state_dict(model, bias=lora_bias)
                
                # Save the models as fp32. This ensures we can finetune again without having to upcast.                      
                save_file(lora_dict, non_webui_weights)
        
        if save_for_webui:
            # Convert the keys to compvis model and webui
            unet_lora_dict = loralb.lora_state_dict(unet, bias=lora_bias) 
            lora_dict_fp16 = unet_dict_converter(unet_lora_dict, strict_mapping=True)
            
            if save_text_weights:
                text_encoder_dict = loralb.lora_state_dict(text_encoder, bias=lora_bias)
                lora_dict_text_fp16 = text_dict_converter(text_encoder_dict)
                
                # Update the Unet dict to include text keys.
                lora_dict_fp16.update(lora_dict_text_fp16)

            # Cast tensors to fp16. It's assumed we won't be finetuning these.
            for k, v in lora_dict_fp16.items():
                lora_dict_fp16[k] = v.to(dtype=torch.float16)

            save_file(
                lora_dict_fp16, 
                save_path, 
                metadata=metadata
            )

def load_lora(model, lora_path: str):
    try:
        if os.path.exists(lora_path):
            lora_dict = load_file(lora_path)
            model.load_state_dict(lora_dict, strict=False)

    except Exception as e:
        print(f"Could not load your lora file: {e}")

def set_mode(model, train=False):
    for n, m in model.named_modules():
        is_lora = hasattr(m, 'merged')
        if is_lora:
            m.train(train)

def set_mode_group(models, train):
   for model in models: 
        set_mode(model, train)
        model.train(train)
