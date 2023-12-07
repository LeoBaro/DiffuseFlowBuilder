from diffuse_flow_builder.pipelines.pipeline_builder import PipelineBuilder
from diffuse_flow_builder.pipelines.pipeline_runner import PipelineRunner


def test_run_pipeline_t2i(data_dir, output_dir):
    data_dir = data_dir / "test_pipeline"
    output_dir = output_dir / "test_run_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = PipelineBuilder().build_pipeline(data_dir / "pipeline.yaml", "t2i_pipeline")

    PipelineRunner().run_pipeline(data_dir / "pipeline.yaml", pipe)

def test_run_pipeline_sr(data_dir, output_dir):
    data_dir = data_dir / "test_pipeline"
    output_dir = output_dir / "test_run_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = PipelineBuilder().build_pipeline(data_dir / "pipeline.yaml", "sr_pipeline")

    PipelineRunner().run_pipeline(data_dir / "pipeline.yaml", pipe)

def test_run_pipeline_t2i_ip(data_dir, output_dir):
    data_dir = data_dir / "test_pipeline"
    output_dir = output_dir / "test_run_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = PipelineBuilder().build_pipeline(data_dir / "pipeline.yaml", "t2i_ip_pipeline")

    PipelineRunner().run_pipeline(data_dir / "pipeline.yaml", pipe)
    