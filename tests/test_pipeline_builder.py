import pytest
from diffuse_flow_builder.pipelines.pipeline_builder import PipelineBuilder


def test_build_pipeline(data_dir, output_dir):
    data_dir = data_dir / "test_pipeline"
    output_dir = output_dir / "test_build_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        PipelineBuilder().build_pipeline(data_dir / "pipeline.yaml", "wrong_pipeline_id")
    
    pipe = PipelineBuilder().build_pipeline(data_dir / "pipeline.yaml", "t2i_pipeline")
    assert len(pipe.components) == 1

    pipe = PipelineBuilder().build_pipeline(data_dir / "pipeline.yaml", "ip_pipeline")
    assert len(pipe.components) == 2