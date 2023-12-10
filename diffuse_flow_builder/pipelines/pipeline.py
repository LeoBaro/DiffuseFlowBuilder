import yaml
import time
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from diffuse_flow_builder.components.component_output import ComponentOutput

from diffuse_flow_builder.utils.constant import ROOT
from diffuse_flow_builder.components.component_base import ComponentBase
from diffuse_flow_builder.logger import logger

import PIL

class Pipeline:
    def __init__(self, pipeline_id: str = None, write_results_after_x_run: int = 5, output_format: str = "jpg"):
        self.pipeline_id = pipeline_id
        self.write_results_after_x_run = write_results_after_x_run
        self.output_format = output_format
        self.components : list[ComponentBase] = []
        self.n_components = 0

    def is_last_component(self, total_components):
        return (len(self.components) == 0) or (total_components == 1)

    def is_first_component(self, total_components):
        return (len(self.components) == self.n_components) or (total_components == 1)

    

    def run(self, n_runs: int, save_intermediate_results: bool):
        start_time = time.time()
    
        current_results  : list[ComponentOutput] = []
        previous_results : list[ComponentOutput] = []
        total_components = len(self.components)

        while len(self.components) > 0:
            
            # compute these before the pop()
            is_first_component = self.is_first_component(total_components)
            is_last_component = self.is_last_component(total_components)
            comp = self.components.pop(0) # components are garbage collected after each iteration

            for i in range(n_runs):

                comp_input = None
                if not is_first_component:
                    comp_input = previous_results[i]

                current_results.append(
                    comp(
                        input_obj = comp_input
                    )
                )

                can_save = (i % self.write_results_after_x_run) == 0

                if is_last_component and can_save:
                    logger.info("Writing intermidiate results to %s", comp.get_output_dir())
                    self.save_images(
                        current_results[i:i+self.write_results_after_x_run],
                        comp.get_output_dir(),
                        comp.get_model_name()
                    )

                if not is_last_component and save_intermediate_results and can_save:
                    logger.info("Writing intermidiate results to %s", comp.get_output_dir())
                    self.save_images(
                        current_results[i:i+self.write_results_after_x_run],
                        comp.get_output_dir(),
                        comp.get_model_name()
                    )
                

            #if not is_last_component and save_intermediate_results:
            #    self.save_images(current_results, comp.get_output_dir(), comp.get_model_name())

            previous_results = deepcopy(current_results)
            current_results = []
            self.dump_configuration(start_time, comp.get_output_dir(), previous_results[-1].kwargs)
            logger.info("Offloading component's model %s from the GPU..", comp.get_model_name())
            del comp # this should be done automatically, but just in case

        logger.info(f"Took {time.time() - start_time:0.2f} seconds to run pipeline %s", self.pipeline_id)

    
    def add_component(self, component: ComponentBase):
        self.components.append(component)
        self.n_components += 1
        return self

    def dump_configuration(self, start_time, output_dir, kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        if "image" in kwargs: kwargs.pop("image")
        kwargs["output_dir"] = str(output_dir)
        with open(output_dir / f"{self.pipeline_id}_config_{datetime.fromtimestamp(start_time).strftime('%Y%m%d-%H%M%S')}.yaml", "w") as c:
            yaml.dump(kwargs, c)

    def save_images(self, runs: list[ComponentOutput], output_dir, model_class_name):
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, comp_output in enumerate(runs):
            for image in comp_output.images:
                output_images_path = (Path(output_dir) / f"{self.pipeline_id}_{comp_output.comp_name}_{model_class_name.lower()}_run_{i}_{time.strftime('%Y%m%d-%H%M%S')}").with_suffix(f".{self.output_format}")
                image.save(output_images_path)


