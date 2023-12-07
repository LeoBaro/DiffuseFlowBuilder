import torch
from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, seed, device, cuda_index):
        self.model = None
        self.model_ready = False
        self.seed, self.device, self.cuda_index = seed, device, cuda_index

    def set_device(self):
        if self.device == "cuda":
            self.model = self.model.to(f"{self.device}:{self.cuda_index}")
        else:
            self.model = self.model.to(self.device)
        self.model_ready = True

    @staticmethod
    def set_device_static(model, device, cuda_index):
        if device == "cuda":
            model = model.to(f"{device}:{cuda_index}")
        else:
            model = model.to(device)
        return model

    def unset_device(self):
        del self.model.unet
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        self.model_ready = False

    def get_generator(self, seed=None, device=None, cuda_index=None):
        if seed is None:
            seed = self.seed
        if device is None:
            device = self.device
        if cuda_index is None:
            cuda_index = self.cuda_index 
        if device == "cuda":
            generator = torch.Generator(f"{device}:{cuda_index}")
        else:
            generator = torch.Generator("cpu")
        if seed:
            generator.manual_seed(seed)
        return generator

    @abstractmethod
    def lazy_load(self, **kwargs):
        pass

    def __del__(self):
        if self.model_ready:
            self.unset_device()
        
            
class HuggingFaceModel(BaseModel):
    pass