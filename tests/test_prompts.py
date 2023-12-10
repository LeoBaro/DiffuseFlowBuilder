import yaml
import pytest
import os
from tempfile import NamedTemporaryFile
from diffuse_flow_builder.prompts.prompt import Prompt, PromptRandomizer

@pytest.fixture
def sample_prompt_dict():
    return {
        "prompt_prefix": "A photograph of",
        "prompt_subject": "a beach",
        "prompt_enanchment": "hyper realistic",
    }

@pytest.fixture
def prompt_randomizer():
    config = {
        "what": ["An image of", "A photograph of", "A picture of", "A graphic of"],
        "background": ["a lake", "a beach", "a desert", "a forest", "a mountain peak", "a city skyline", "a tropical island"],
        "view_angle": ["viewed from a low angle", "viewed from a pedestrian angle", "viewed from a moving vehicle", "viewed from a hidden vantage point"],
        "weather": ["in a clear sky", "in a cloudy day", "in a rainy day", "in a snowy landscape", "in a gentle drizzle"],
        "daytime": ["in the morning", "in the afternoon", "during the night", "at twilight", "during sunrise", "during sunset", "under moonlight"],
        "enanchment": ["ultra detailed", "hyper realistic", "high quality", "hyper detailed", "photorealistic", "8k", "4k detail post processing"],
        "lens": ["Wide-angle", "Telephoto", "24mm", "EF 70mm"]
    }
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_config:
        yaml.dump(config, temp_config)
        temp_config_path = temp_config.name

    yield PromptRandomizer(temp_config_path)

    os.remove(temp_config_path)

def test_prompt_from_dict(sample_prompt_dict):
    prompt = Prompt.from_dict(sample_prompt_dict)
    assert prompt.prefix == "A photograph of"
    assert prompt.subject == "a beach"
    assert prompt.enhancement == "hyper realistic"

def test_combine_with():
    prompt1 = Prompt("An image of", "a lake", "high quality")
    prompt2 = Prompt(None, "a forest", "ultra detailed")

    combined_prompt = prompt1.combine_with(prompt2)

    assert combined_prompt.prefix == "An image of"
    assert combined_prompt.subject == "a lake"
    assert combined_prompt.enhancement == "high quality"

def test_randomizer_sample_random(prompt_randomizer):
    sample = prompt_randomizer.sample_random("background")
    assert sample in ["a lake", "a beach", "a desert", "a forest", "a mountain peak", "a city skyline", "a tropical island"]

def test_randomizer_process_string(prompt_randomizer):
    processed_string = prompt_randomizer.process_string("An image of $background, $enanchment")
    assert "An image of" in processed_string
    assert processed_string.count("$") == 0

def test_randomizer_from_dict(prompt_randomizer, sample_prompt_dict):
    prompt = prompt_randomizer.from_dict(sample_prompt_dict)
    assert prompt.prefix == "A photograph of"
    assert prompt.subject == "a beach"
    assert prompt.enhancement == "hyper realistic"
