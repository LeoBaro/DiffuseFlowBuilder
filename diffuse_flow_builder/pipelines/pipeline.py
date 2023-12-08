from copy import deepcopy
import time
from pathlib import Path
from diffuse_flow_builder.components.component_output import ComponentOutput

from diffuse_flow_builder.utils.constant import ROOT
from diffuse_flow_builder.components.component_base import ComponentBase
from diffuse_flow_builder.logger import logger

import PIL

class Pipeline:
    def __init__(self, pipeline_id: str = None, output_format: str = "jpg"):
        self.pipeline_id = pipeline_id
        self.output_format = output_format
        self.components : list[ComponentBase] = []
        self.n_components = 0

    def is_last_component(self):
        return len(self.components) == 0

    def is_first_component(self):
        return len(self.components) == self.n_components


    def run(self, n_runs: int, save_intermediate_results: bool):
        start_time = time.time()
    
        current_results  : list[ComponentOutput] = []
        previous_results : list[ComponentOutput] = []

        while len(self.components) > 0:
            
            if self.is_first_component():
                comp = self.components.pop(0) # components are garbage collected after each iteration
                for _ in range(n_runs):
                    current_results.append(comp())
            else:
                comp = self.components.pop(0) # components are garbage collected after each iteration
                if len(previous_results) == 0:
                    raise ValueError(f"Component {comp.get_model_name()} needs input from previous step, but no previous step was found")
                
                for comp_output in previous_results:
                    current_results.append(
                        comp(
                            input_obj=comp_output
                            )
                    )

            if (not self.is_last_component() and save_intermediate_results) or self.is_last_component():
                self.save_images(current_results, comp.get_output_dir(), comp.get_model_name())

            previous_results = deepcopy(current_results)
            current_results = []

            logger.info("Offloading component's model %s from the GPU..", comp.get_model_name())
            del comp # this should be done automatically, but just in case

        logger.info(f"Took {time.time() - start_time:0.2f} seconds to run pipeline %s", self.pipeline_id)

    
    def add_component(self, component: ComponentBase):
        self.components.append(component)
        self.n_components += 1
        return self
    
    def save_images(self, runs: list[ComponentOutput], output_dir, model_class_name):
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, comp_output in enumerate(runs):
            for image in comp_output.images:
                image.save((Path(output_dir) / f"{self.pipeline_id}_{model_class_name.lower()}_run_{i}_{time.strftime('%Y%m%d-%H%M%S')}").with_suffix(f".{self.output_format}"))