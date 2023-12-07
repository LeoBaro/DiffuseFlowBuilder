from dataclasses import dataclass

from PIL import Image

@dataclass
class ComponentOutput:
    images: list[Image.Image]
    prompts: list[str]

