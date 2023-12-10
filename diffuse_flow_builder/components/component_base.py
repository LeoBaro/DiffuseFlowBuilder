from abc import ABC, abstractmethod

from pathlib import Path

from diffuse_flow_builder.components.component_output import ComponentOutput
from diffuse_flow_builder.prompts.prompt import PromptRandomizer
from diffuse_flow_builder.utils.constant import ROOT

class ComponentBase(ABC):
    def __init__(self, **kwargs):
        self.model_class_name = kwargs["model"]
        self.seed = kwargs["seed"]
        self.kwargs = kwargs
        self.prompt_randomizer = PromptRandomizer(ROOT / "diffuse_flow_builder" / "default_configs" / "prompts.yaml")
        self.expand_root_dir()

    @abstractmethod
    def __call__(self, input_obj: ComponentOutput = None) -> ComponentOutput:
        pass

    def get_output_dir(self):
        return self.kwargs["output_dir"]
    
    def get_model_name(self):
        return self.model_class_name
    
    def expand_root_dir(self):
        for key, val in self.kwargs.items():
            if isinstance(val, str) and "$ROOT" in val:
                self.kwargs[key] = Path(self.kwargs[key].replace("$ROOT", str(ROOT)))



    def remove_unsupported_keywords(self, kwargs, unsupported_kwargs):
        keys = list(kwargs.keys())
        for uns_k in unsupported_kwargs:
            if uns_k in keys:
                kwargs.pop(uns_k)
    
    def use_input_from_previous_step(self):
        if "use_input_from_previous_step" not in self.kwargs:
            return False
        return self.kwargs["use_input_from_previous_step"]
    
    def check_required_inputs(self):
        required_params = [
            "name", 
            "model", 
            "prompt", 
            "use_prompt_from_previous_step",
            "combine_prompt_with_previous_step",
            "apply_refinement", 
            "strength",
            "guidance_scale",
            "num_inference_steps",
            "height",
            "width",
            "num_images_per_prompt",
            "output_dir",
            "override_with_recommended_parameters"
        ]
        for param in required_params:
            if param not in self.kwargs:
                raise ValueError(f"Parameter {param} is missing")
