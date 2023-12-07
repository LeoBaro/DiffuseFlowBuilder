
from diffuse_flow_builder.models.stable_diffusion_xl import StableDiffusionXL
from diffuse_flow_builder.models.stable_diffusion_2 import StableDiffusion2


def test_stable_diffusion_xl():
    _ = StableDiffusionXL("text_to_image", apply_refinement=False)
    _ = StableDiffusionXL("text_to_image")
    _ = StableDiffusionXL("in_painting")
    _ = StableDiffusionXL("image_to_image")

def test_stable_diffusion_2():
    _ = StableDiffusion2("text_to_image")
    _ = StableDiffusion2("in_painting")
    _ = StableDiffusion2("image_to_image")
    _ = StableDiffusion2("super_resolution")

