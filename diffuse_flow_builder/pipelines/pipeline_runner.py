from pathlib import Path

import yaml

from diffuse_flow_builder.pipelines.pipeline import Pipeline


class PipelineRunner:

    @staticmethod
    def run_pipeline(config_path: Path, pipe: Pipeline):
        
        with open(config_path, "r", encoding="utf-8") as cf:
            p_config = yaml.safe_load(cf)["pipelines"][pipe.pipeline_id]
        
        pipe.run(
            p_config["n_runs"],
            p_config["save_intermediate_results"],
        )

