from copy import deepcopy
import importlib

import PIL
from diffuse_flow_builder.components.component_base import ComponentBase
from diffuse_flow_builder.components.component_output import ComponentOutput

class SuperResolution(ComponentBase):
    """
    Stable Diffusion 2  __call__ params : https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale#diffusers.StableDiffusionUpscalePipeline.__call__
    """
    available_models = [
        "StableDiffusion2"
    ] 
    unsupported_kwargs = {
        "StableDiffusion2": [
            "name", "model", "prompt_prefix", "prompt_subject", "prompt_enanchment", "use_image_for_previous_step", "apply_refinement", "height", "width", "output_dir"
        ]
    }       
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs["model"] not in SuperResolution.available_models:
            raise ValueError(f"Model {kwargs['model']} can not perform super-resolution")

        ModelClass = getattr(importlib.import_module("diffuse_flow_builder.models"), kwargs["model"])

        self.model = ModelClass("super_resolution")
                
    def __call__(self, input_obj: ComponentOutput = None) -> list[PIL.Image]:
        
        # This code will be refactored into a PromptManager class
        prompt = f"{self.kwargs['prompt_prefix']} {self.kwargs['prompt_subject']}, {self.kwargs['prompt_enanchment']}"

        kwargs = self.remove_unsupported_keywords(
            deepcopy(self.kwargs),
            SuperResolution.unsupported_kwargs[self.model_class_name]
        )

        kwargs["image"] = PIL.Image.open(kwargs["image"]).convert("RGB")
        #kwargs["image"].thumbnail((500,500))


        return self.model.inference(prompt=prompt, **kwargs)
