import yaml
import importlib
from pathlib import Path

from diffuse_flow_builder.pipelines.pipeline import Pipeline
from diffuse_flow_builder.components.component_base import ComponentBase
from diffuse_flow_builder.logger import logger

class PipelineBuilder:

    @staticmethod
    def create_component(component_config: dict) -> ComponentBase:
        ComponentClass = getattr(importlib.import_module("diffuse_flow_builder.components"), component_config["name"])
        return ComponentClass(**component_config)

    @staticmethod
    def build_pipeline(config_path: Path, pipeline_id: str) -> Pipeline:

        with open(config_path, "r", encoding="utf-8") as cf:
            p_config = yaml.safe_load(cf)

        if pipeline_id not in p_config["pipelines"].keys():
            raise ValueError(f"Pipeline {pipeline_id} is not present in {config_path} ")
        
        pipeline_config = p_config["pipelines"][pipeline_id]
        pipe = Pipeline(pipeline_id, output_format=pipeline_config["output_format"])
        logger.info("Building pipeline %s with %s steps", pipeline_id, len(pipeline_config['steps']))

        for comp_config in pipeline_config["steps"]:
            pipe.add_component(
                PipelineBuilder.create_component(
                    comp_config
                )
            )
            logger.info("Added component %s to pipeline %s", comp_config['name'], pipeline_id)

        return pipe