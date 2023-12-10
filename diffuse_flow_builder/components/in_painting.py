from copy import deepcopy
import os
import random
import importlib
from pathlib import Path

import PIL
from diffusers.utils import load_image

from diffuse_flow_builder.prompts.prompt import PromptRandomizer
from diffuse_flow_builder.components.component_base import ComponentBase
from diffuse_flow_builder.components.component_output import ComponentOutput


class InPainting(ComponentBase):
    """
    Stable Diffusion 2  __call__ params : ?
    Stable Diffusion XL __call__ params : https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline
    """
    available_models =[
        "StableDiffusion2",
        "StableDiffusionXL",
        "StableDiffusionXLTurbo"
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs["model"] not in InPainting.available_models:
            raise ValueError(f"Model {kwargs['model']} can not perform text_to_image")

        ModelClass = getattr(importlib.import_module("diffuse_flow_builder.models"), kwargs["model"])

        self.model = ModelClass("in_painting", kwargs["apply_refinement"])


    def __call__(self, input_obj: ComponentOutput = None) -> list[PIL.Image]:

        kwargs = self.check_inputs()

        if kwargs["use_image_from_previous_step"]:
            image = input_obj.images[-1]
        else:
            image = load_image(str(kwargs["image"]))

        ## Prompts : move to superclass
        prompt = self.prompt_randomizer.from_dict(kwargs["prompt"])

        if kwargs["use_prompt_from_previous_step"]:
            prompt = input_obj.prompts[-1]

        elif kwargs["combine_prompt_with_previous_step"]:
            prompt = prompt.combine_with(input_obj.prompts[-1])


        ## Masks
        if kwargs["use_random_masks"]:
            if not Path(kwargs["use_random_masks"]).is_dir():
                raise ValueError("Masks directory does not exist")
            mask_image = load_image(
                os.path.join(
                    kwargs["use_random_masks"],
                    random.choice(os.listdir(kwargs["use_random_masks"]))
                )
            )
        elif kwargs["mask_image"] is not None:
            mask_image = load_image(str(kwargs["mask_image"]))

        mask_image = mask_image.resize(image.size)
        kwargs.pop("image")
        kwargs.pop("prompt")
        kwargs.pop("mask_image")

        return ComponentOutput(
            images=self.model.inference(image=image, prompt=prompt.get_str_prompt(), mask_image=mask_image, **kwargs),
            prompts=[prompt]
        )
        
    def check_inputs(self):
        super().check_required_inputs()

        if not self.kwargs["use_prompt_from_previous_step"] and self.kwargs["prompt"] is None:
            raise ValueError("No prompt is given")
        
        if self.kwargs["mask_image"] is None and not self.kwargs["use_random_masks"]:
            raise ValueError("No mask is given")
        
        if self.kwargs["image"] is None and not self.kwargs["use_image_from_previous_step"]:
            raise ValueError("No image is given")
        
        return deepcopy(self.kwargs)
