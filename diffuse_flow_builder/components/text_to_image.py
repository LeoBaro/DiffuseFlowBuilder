from copy import deepcopy
import importlib

import PIL
from diffuse_flow_builder.components.component_base import ComponentBase
from diffuse_flow_builder.components.component_output import ComponentOutput
from diffuse_flow_builder.prompts.prompt import Prompt
from diffuse_flow_builder.logger import logger

class TextToImage(ComponentBase):
    """
    Stable Diffusion 2  __call__ params : ?
    Stable Diffusion XL __call__ params : https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.__call__
    
    """
    available_models =[
        "StableDiffusion2",
        "StableDiffusionXL",
        "StableDiffusionXLTurbo",
    ] 
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs["model"] not in TextToImage.available_models:
            raise ValueError(f"Model {kwargs['model']} can not perform text_to_image")

        ModelClass = getattr(importlib.import_module("diffuse_flow_builder.models"), kwargs["model"])

        self.model = ModelClass("text_to_image", kwargs["apply_refinement"])
        logger.info("Initialized text_to_image component with seed %s", self.seed)

    def __call__(self, input_obj: ComponentOutput = None) -> list[PIL.Image]:

        kwargs = self.check_inputs()

        ## Prompts : move to superclass
        prompt = self.prompt_randomizer.from_dict(
            kwargs.pop("prompt")
        )
        
        if kwargs["use_prompt_from_previous_step"]:
            prompt = input_obj.prompts[-1]

        elif kwargs["combine_prompt_with_previous_step"]:
            prompt = prompt.combine_with(input_obj.prompts[-1])


        images, kwargs = self.model.inference(prompt=prompt.get_str_prompt(), **kwargs)

        return ComponentOutput(
            images=images,
            prompts=[prompt],
            comp_name=self.__class__.__name__,
            kwargs=kwargs
        )
    
    def check_inputs(self):
        super().check_required_inputs()

        if not self.kwargs["use_prompt_from_previous_step"] and self.kwargs["prompt"] is None:
            raise ValueError("Prompt is empty")
        return deepcopy(self.kwargs)
