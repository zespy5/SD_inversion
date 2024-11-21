from transformers import CLIPTextModel, CLIPTokenizer, logging, CLIPTextModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerDiscreteScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from pnp_utils import *
import torchvision.transforms as T


def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start

class PreprocessXL(nn.Module):
    
    def __init__(self, scheduler = None):
        super().__init__()
        
        self.device = 'cuda'
        self.sd_version = "stabilityai/stable-diffusion-xl-base-1.0"
        
        self.vae = AutoencoderKL.from_pretrained(self.sd_version, 
                                                 subfolder="vae",
                                                 revision="fp16", 
                                                 torch_dtype=torch.float16).to(self.device)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(self.sd_version,
                                                        subfloder = "tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.sd_version,
                                                         subfloder = "tokenizer_2")
        
        self.text_encoder = CLIPTextModel.from_pretrained(self.sd_version,
                                                          subfloder = "text_encoder",
                                                          revision="fp16",
                                                          torch_dtype=torch.float16).to(self.device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(self.sd_version,
                                                                          subfloder="text_encoder_2",
                                                                          revision="fp16",
                                                                          torch_dtype=torch.float16).to(self.device)
        
        self.unet = UNet2DConditionModel.from_pretrained(self.sd_version,
                                                         subfolder="unet",
                                                         revision="fp16",
                                                         torch_dtype=torch.float16).to(self.device)
        
        self.scheduler = EulerDiscreteScheduler.from_pretrained(self.sd_version, subfolder="scheduler")\
            if scheduler == None else scheduler
            
        