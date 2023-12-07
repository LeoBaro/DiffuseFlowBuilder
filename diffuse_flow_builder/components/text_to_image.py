import importlib

import PIL
from diffuse_flow_builder.components.component_base import ComponentBase
from diffuse_flow_builder.components.component_output import ComponentOutput
from diffuse_flow_builder.prompts.prompt import Prompt

class TextToImage(ComponentBase):
    """
    Stable Diffusion 2  __call__ params : ?
    Stable Diffusion XL __call__ params : https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.__call__
    """
    available_models =[
        "StableDiffusion2",
        "StableDiffusionXL"
    ] 
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs["model"] not in TextToImage.available_models:
            raise ValueError(f"Model {kwargs['model']} can not perform text_to_image")

        ModelClass = getattr(importlib.import_module("diffuse_flow_builder.models"), kwargs["model"])

        self.model = ModelClass("text_to_image", kwargs["apply_refinement"])

    def __call__(self, input_obj: ComponentOutput = None) -> list[PIL.Image]:

        self.check_inputs()

        # if use prompt from previous step...

        # if use random prompt...

        # This code will be refactored into a PromptManager class
        prompt = Prompt.from_dict(self.kwargs["static_prompt"])

        return ComponentOutput(
            images=self.model.inference(prompt=prompt.get_str_prompt(), **self.kwargs),
            prompts=[prompt]
        )
    
    def check_inputs(self):
        super().check_required_inputs()

        if not self.kwargs["use_random_prompt"] and self.kwargs["static_prompt"] is None:
            raise ValueError("Prompt is empty")
        
