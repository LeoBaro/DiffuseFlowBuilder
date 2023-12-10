import random
import re
import yaml
from dataclasses import dataclass   

@dataclass
class Prompt:
    prefix:      str | None = None
    subject:     str | None = None
    enhancement: str | None = None

    def get_str_prompt(self):
        return f"{self.prefix} {self.subject}, {self.enhancement}"
    
    @staticmethod
    def from_dict(d):
        return Prompt(d["prompt_prefix"], d["prompt_subject"], d["prompt_enanchment"])
    
    def combine_with(self, prompt):

        combined_prompt = Prompt()

        if self.prefix is not None:
            combined_prompt.prefix = self.prefix
        else:
            combined_prompt.prefix = prompt.prefix

        if self.subject is not None:
            combined_prompt.subject = self.subject
        else:
            combined_prompt.subject = prompt.subject

        if self.enhancement is not None:
            combined_prompt.enhancement = self.enhancement
        else:
            combined_prompt.enhancement = prompt.enhancement

        return combined_prompt
    
class PromptRandomizer:
    def __init__(self, prompt_config, seed=None):
        with open(prompt_config, "r") as cf:
            self.p_config = yaml.safe_load(cf)
        if seed:
            self.seed = seed

    def sample_random(self, key):
        return random.choice(self.p_config[key])

    @staticmethod
    def extract_words_starting_with_dollar(input_string):
        pattern = r'\$[a-zA-Z0-9_]+'
        matches = re.findall(pattern, input_string)
        return matches

    def process_string(self, prompt_part: str):
        if "$" not in prompt_part:
            return prompt_part
        words_to_replace = [w.replace("$","") for w in PromptRandomizer.extract_words_starting_with_dollar(prompt_part)]
        mapping = {}
        for w in words_to_replace:
            mapping[w] = self.sample_random(w)
        for key, val in mapping.items():
            prompt_part = prompt_part.replace(f"${key}", val)
        return prompt_part

    def from_dict(self, d: dict):
        return Prompt(
            self.process_string(d["prompt_prefix"]),
            self.process_string(d["prompt_subject"]),
            self.process_string(d["prompt_enanchment"]),
        )