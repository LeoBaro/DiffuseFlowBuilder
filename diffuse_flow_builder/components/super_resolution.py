from copy import deepcopy
import importlib
from diffusers.utils import load_image

import PIL
from diffuse_flow_builder.components.component_base import ComponentBase
from diffuse_flow_builder.components.component_output import ComponentOutput
from diffuse_flow_builder.prompts.prompt import Prompt, PromptRandomizer

class SuperResolution(ComponentBase):
    """
    Stable Diffusion 2  __call__ params : https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale#diffusers.StableDiffusionUpscalePipeline.__call__
    """
    available_models = [
        "StableDiffusion2"
    ] 
    unsupported_kwargs = {
        "StableDiffusion2": [
            "name", "model", "use_prompt_from_previous_step", "combine_prompt_with_previous_step", "use_image_for_previous_step", "apply_refinement", "strength", "use_image_from_previous_step", "height", "width", "output_dir"
        ]
    }       
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs["model"] not in SuperResolution.available_models:
            raise ValueError(f"Model {kwargs['model']} can not perform super-resolution")

        ModelClass = getattr(importlib.import_module("diffuse_flow_builder.models"), kwargs["model"])

        self.model = ModelClass("super_resolution")
                
    def __call__(self, input_obj: ComponentOutput = None) -> list[PIL.Image]:


        self.check_inputs()

        kwargs = deepcopy(self.kwargs)

        if kwargs["use_image_from_previous_step"]:
            image = input_obj.images[-1]
        else:
            image = PIL.Image.open(kwargs["image"]).convert("RGB")

        # This code will be refactored into a PromptManager class
        prompt = self.prompt_randomizer.from_dict(kwargs["prompt"])

        ## Prompts
        if kwargs["use_prompt_from_previous_step"]:
            prompt = input_obj.prompts[-1].get_str_prompt()

        elif kwargs["combine_prompt_with_previous_step"]:
            prompt = prompt.combine_with(input_obj.prompts[-1]).get_str_prompt()

        self.remove_unsupported_keywords(
            kwargs,
            SuperResolution.unsupported_kwargs[self.model_class_name]
        )

        kwargs["image"] = PIL.Image.open(kwargs["image"]).convert("RGB")
        #kwargs["image"].thumbnail((500,500))

        kwargs.pop("image")
    
        return ComponentOutput(
            images=self.model.inference(image=image, prompt=prompt.get_str_prompt(), **kwargs),
            prompts=[prompt]
        )
        

    def check_inputs(self):
        super().check_required_inputs()

        if not self.kwargs["use_prompt_from_previous_step"] and self.kwargs["prompt"] is None:
            raise ValueError("No prompt is given")
        
        if self.kwargs["image"] is None and not self.kwargs["use_image_from_previous_step"]:
            raise ValueError("No image is given")
