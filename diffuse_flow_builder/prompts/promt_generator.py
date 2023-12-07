import yaml
import random
from pathlib import Path


class PromptGenerator:

    def __init__(self, prompt_config: Path, seed=None):

        with open(prompt_config, "r") as cf:
            self.p_config = yaml.safe_load(cf)
        if seed:
            self.seed = seed

    def randomize_prompt(self):
        if self.seed:
            random.seed(666)



def randomize_background_prompt(config):
    return (
        config.PROMPT_START
        + " "
        + random.choice(config.BACKGROUND_PROMPTS)
        + " "
        + random.choice(config.VIEW_ANGLE_PROMPTS)
        + " "
        + random.choice(config.DAYTIME_PROMPTS)
    )

    # random.choice(config.WEATHER_PROMPTS) + " " + \


def randomize_enhnancement(config):
    return (
        random.choice(config.ENHANCING_PROMPTS["Lens"])
        + ", "
        + random.choice(config.ENHANCING_PROMPTS["Std"])
    )


# An image of  a forest viewed from a hidden vantage point at twilight.  24mm, photorealistic;
def extract_background_description(config, background_prompt):
    # a forest viewed from a hidden vantage point at twilight.  24mm, photorealistic;
    return background_prompt.split(config.PROMPT_START)[1]
