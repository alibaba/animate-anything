import os
from logging import warnings
import torch
from typing import Union
from types import SimpleNamespace
from models.unet_3d_condition_mask import UNet3DConditionModel
from transformers import CLIPTextModel
from utils.convert_diffusers_to_original_ms_text_to_video import convert_unet_state_dict, convert_text_enc_state_dict_v20

from .lora import (
    extract_lora_ups_down,
    inject_trainable_lora_extended,
    save_lora_weight,
    train_patch_pipe,
    monkeypatch_or_replace_lora,
    monkeypatch_or_replace_lora_extended
)

from stable_lora.lora import (
    activate_lora_train,
    add_lora_to,
    save_lora,
    load_lora,
    set_mode_group
)

FILE_BASENAMES = ['unet', 'text_encoder']
LORA_FILE_TYPES = ['.pt', '.safetensors']
CLONE_OF_SIMO_KEYS = ['model', 'loras', 'target_replace_module', 'r']
STABLE_LORA_KEYS = ['model', 'target_module', 'search_class', 'r', 'dropout', 'lora_bias']

lora_versions = dict(
    stable_lora = "stable_lora",
    cloneofsimo = "cloneofsimo"
)

lora_func_types = dict(
    loader = "loader",
    injector = "injector"
)

lora_args = dict(
    model = None,
    loras = None,
    target_replace_module = [],
    target_module = [],
    r = 4,
    search_class = [torch.nn.Linear],
    dropout = 0,
    lora_bias = 'none'
)

LoraVersions = SimpleNamespace(**lora_versions)
LoraFuncTypes = SimpleNamespace(**lora_func_types)

LORA_VERSIONS = [LoraVersions.stable_lora, LoraVersions.cloneofsimo]
LORA_FUNC_TYPES = [LoraFuncTypes.loader, LoraFuncTypes.injector]

def filter_dict(_dict, keys=[]):
    if len(keys) == 0:
        assert "Keys cannot empty for filtering return dict."
    
    for k in keys:
        if k not in lora_args.keys():
            assert f"{k} does not exist in available LoRA arguments"
            
    return {k: v for k, v in _dict.items() if k in keys}

class LoraHandler(object):
    def __init__(
        self, 
        version: LORA_VERSIONS = LoraVersions.cloneofsimo, 
        use_unet_lora: bool = False,
        use_text_lora: bool = False,
        save_for_webui: bool = False,
        only_for_webui: bool = False,
        lora_bias: str = 'none',
        unet_replace_modules: list = ['UNet3DConditionModel'],
        text_encoder_replace_modules: list = ['CLIPEncoderLayer']
    ):
        self.version = version
        self.lora_loader = self.get_lora_func(func_type=LoraFuncTypes.loader)
        self.lora_injector = self.get_lora_func(func_type=LoraFuncTypes.injector)
        self.lora_bias = lora_bias
        self.use_unet_lora = use_unet_lora
        self.use_text_lora = use_text_lora
        self.save_for_webui = save_for_webui
        self.only_for_webui = only_for_webui
        self.unet_replace_modules = unet_replace_modules
        self.text_encoder_replace_modules = text_encoder_replace_modules
        self.use_lora = any([use_text_lora, use_unet_lora])

        if self.use_lora:
            print(f"Using LoRA Version: {self.version}")

    def is_cloneofsimo_lora(self):
        return self.version == LoraVersions.cloneofsimo

    def is_stable_lora(self):
        return self.version == LoraVersions.stable_lora

    def get_lora_func(self, func_type: LORA_FUNC_TYPES = LoraFuncTypes.loader):

        if self.is_cloneofsimo_lora():

            if func_type == LoraFuncTypes.loader:
                return monkeypatch_or_replace_lora_extended

            if func_type == LoraFuncTypes.injector:
                return inject_trainable_lora_extended

        if self.is_stable_lora():

            if func_type == LoraFuncTypes.loader:
                return load_lora

            if func_type == LoraFuncTypes.injector:
                return add_lora_to
                
        assert "LoRA Version does not exist."

    def check_lora_ext(self, lora_file: str):
        return lora_file.endswith(tuple(LORA_FILE_TYPES))

    def get_lora_file_path(
        self, 
        lora_path: str, 
        model: Union[UNet3DConditionModel, CLIPTextModel]
    ):
        if os.path.exists(lora_path):
            lora_filenames = [fns for fns in os.listdir(lora_path)]
            is_lora = self.check_lora_ext(lora_path)

            is_unet = isinstance(model, UNet3DConditionModel)
            is_text =  isinstance(model, CLIPTextModel)
            idx = 0 if is_unet else 1

            base_name = FILE_BASENAMES[idx]
            
            for lora_filename in lora_filenames:
                is_lora = self.check_lora_ext(lora_filename)
                if not is_lora:
                    continue
                
                if base_name in lora_filename:
                    return os.path.join(lora_path, lora_filename)

        return None

    def handle_lora_load(self, file_name:str, lora_loader_args: dict = None):
        self.lora_loader(**lora_loader_args)
        print(f"Successfully loaded LoRA from: {file_name}")
    
    def load_lora(self, model, lora_path: str = '', lora_loader_args: dict = None,):
        try:
            lora_file = self.get_lora_file_path(lora_path, model)

            if lora_file is not None:
                lora_loader_args.update({"lora_path": lora_file})
                self.handle_lora_load(lora_file, lora_loader_args)

            else:
                print(f"Could not load LoRAs for {model.__class__.__name__}. Injecting new ones instead...")

        except Exception as e:
            print(f"An error occured while loading a LoRA file: {e}")
                 
    def get_lora_func_args(self, lora_path, use_lora, model, replace_modules, r, dropout, lora_bias):
        return_dict = lora_args.copy()
    
        if self.is_cloneofsimo_lora():
            return_dict = filter_dict(return_dict, keys=CLONE_OF_SIMO_KEYS)
            return_dict.update({
                "model": model,
                "loras": self.get_lora_file_path(lora_path, model),
                "target_replace_module": replace_modules,
                "r": r
            })

        if self.is_stable_lora():
            KEYS = ['model', 'lora_path']
            return_dict = filter_dict(return_dict, KEYS)
            
            return_dict.update({'model': model, 'lora_path': lora_path})

        return return_dict

    def do_lora_injection(
        self, 
        model, 
        replace_modules, 
        bias='none',
        dropout=0,
        r=4,
        lora_loader_args=None,
    ):  
        REPLACE_MODULES = replace_modules

        params = None
        negation = None
        is_injection_hybrid = False
        
        if self.is_cloneofsimo_lora():
            is_injection_hybrid = True
            injector_args = lora_loader_args

            params, negation = self.lora_injector(**injector_args)   
            for _up, _down in extract_lora_ups_down(
                model, 
                target_replace_module=REPLACE_MODULES):

                if all(x is not None for x in [_up, _down]):
                    print(f"Lora successfully injected into {model.__class__.__name__}.")

                break

            return params, negation, is_injection_hybrid

        if self.is_stable_lora():
            injector_args = lora_args.copy()
            injector_args = filter_dict(injector_args, keys=STABLE_LORA_KEYS)

            SEARCH_CLASS = [torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Embedding]

            injector_args.update({
                "model": model,
                "target_module": REPLACE_MODULES,
                "search_class": SEARCH_CLASS,
                "r": r,
                "dropout": dropout,
                "lora_bias": self.lora_bias
            })

            activator = self.lora_injector(**injector_args)
            activator()

        return params, negation, is_injection_hybrid

    def add_lora_to_model(self, use_lora, model, replace_modules, dropout=0.0, lora_path='', r=16):

        params = None
        negation = None

        lora_loader_args = self.get_lora_func_args(
            lora_path,
            use_lora,
            model,
            replace_modules,
            r,
            dropout,
            self.lora_bias
        )
        if use_lora:
            params, negation, is_injection_hybrid = self.do_lora_injection(
                model, 
                replace_modules, 
                bias=self.lora_bias,
                lora_loader_args=lora_loader_args,
                dropout=dropout,
                r=r
            )

            if not is_injection_hybrid:
                self.load_lora(model, lora_path=lora_path, lora_loader_args=lora_loader_args)
        
        params = model if params is None else params
        return params, negation
    

    def deactivate_lora_train(self, models, deactivate=True):
        """
        Usage: Use before and after sampling previews.
        Currently only available for Stable LoRA.
        """
        if self.is_stable_lora():
            set_mode_group(models, not deactivate)

    def save_cloneofsimo_lora(self, model, save_path, step):
        
        def save_lora(model, name, condition, replace_modules, step, save_path): 
            if condition and replace_modules is not None:
                save_path = f"{save_path}/{step}_{name}.pt"
                save_lora_weight(model, save_path, replace_modules)

        save_lora(
            model.unet, 
            FILE_BASENAMES[0], 
            self.use_unet_lora, 
            self.unet_replace_modules, 
            step,
            save_path, 
        )
        save_lora(
            model.text_encoder, 
            FILE_BASENAMES[1], 
            self.use_text_lora, 
            self.text_encoder_replace_modules, 
            step, 
            save_path
        )

        train_patch_pipe(model, self.use_unet_lora, self.use_text_lora)

    def save_stable_lora(
        self, 
        model, 
        step, 
        name, 
        save_path = '', 
        save_for_webui=False,
        only_for_webui=False
    ):
        import uuid

        save_filename = f"{step}_{name}"
        lora_metadata =  metadata = {
        "stable_lora_text_to_video": "v1", 
        "lora_name": name + "_" + uuid.uuid4().hex.lower()[:5]
    }
        save_lora(
            unet=model.unet,
            text_encoder=model.text_encoder,
            save_text_weights=self.use_text_lora,
            output_dir=save_path,
            lora_filename=save_filename,
            lora_bias=self.lora_bias,
            save_for_webui=self.save_for_webui,
            only_webui=self.only_for_webui,
            metadata=lora_metadata,
            unet_dict_converter=convert_unet_state_dict,
            text_dict_converter=convert_text_enc_state_dict_v20
        )

    def save_lora_weights(self, model: None, save_path: str ='',step: str = ''):
        save_path = f"{save_path}/lora"
        os.makedirs(save_path, exist_ok=True)

        if self.is_cloneofsimo_lora():
            if any([self.save_for_webui, self.only_for_webui]):
                warnings.warn(
                    """
                    You have 'save_for_webui' enabled, but are using cloneofsimo's LoRA implemention.
                    Only 'stable_lora' is supported for saving to a compatible webui file.
                    """
                )
            self.save_cloneofsimo_lora(model, save_path, step)

        if self.is_stable_lora():
            name = 'lora_text_to_video'
            self.save_stable_lora(model, step, name, save_path)