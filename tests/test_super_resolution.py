import pytest 
from diffusers.utils import load_image, make_image_grid

from diffuse_flow_builder.components.super_resolution import SuperResolution

def test_super_resolution_stable_diffusion_2(data_dir, output_dir):
    data_dir = data_dir / "test_super_resolution"
    output_dir = output_dir / "test_super_resolution_stable_diffusion_2"
    output_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        SuperResolution("StableDiffusionXL")
    


    sr_sd2 = SuperResolution("StableDiffusion2")

    low_res_img = load_image(str(data_dir / "low_res_cat.png"))
    low_res_img = low_res_img.resize((128, 128))
    prompt = "a white cat"


    i = sr_sd2(
        prompt = prompt, 
        image  = low_res_img
    )
    assert len(i) == 1

    make_image_grid([low_res_img.resize((512, 512)), i[0].resize((512, 512))], rows=1, cols=2).save(output_dir/"upscaled.png")

