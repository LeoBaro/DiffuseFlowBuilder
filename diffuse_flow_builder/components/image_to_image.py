import importlib

import PIL

from diffuse_flow_builder.components.component_base import ComponentBase

class ImageToImage(ComponentBase):
    """
    Stable Diffusion 2  __call__ params : ?
    Stable Diffusion XL __call__ params : https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline.__call__
    """
    available_models =[
        "StableDiffusion2",
        "StableDiffusionXL"
    ] 
    
    def __init__(self, model_class_name: str):
        super().__init__(model_class_name)
        if model_class_name not in ImageToImage.available_models:
            raise ValueError(f"Model {model_class_name} can not perform in image to image")

        ModelClass = getattr(importlib.import_module("diffuse_flow_builder.models"), model_class_name)

        self.model = ModelClass("in_painting")

    
    def __call__(self,
        **kwargs
    ) -> list[PIL.Image]:
        raise NotImplementedError("Image to image is not implemented yet")