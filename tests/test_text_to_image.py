from diffuse_flow_builder.components.text_to_image import TextToImage
from diffusers.utils import make_image_grid

def test_text_to_image_stable_diffusion_xl(output_dir):
    output_dir = output_dir / "test_text_to_image"
    output_dir.mkdir(parents=True, exist_ok=True)

    tti_sdxl = TextToImage("StableDiffusionXL")
    i = tti_sdxl(
        prompt   = "A drone is flying in the sky, photorealistic.",
        prompt_2 = "A drone is flying in the sky, photorealistic.",
        height   = 1024,
        width    = 1024,
        num_inference_steps = 50,
        guidance_scale = 5.0,
        negative_prompt = None,
        negative_prompt_2 = None,
        num_images_per_prompt = 2,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        output_type = "pil", # or np.array 
        return_dict = True   # StableDiffusionXLOutput if False
    )
    assert len(i) == 2

    make_image_grid([i[0].resize((512, 512)), i[1].resize((512, 512))], rows=1, cols=2).save(output_dir/"stable_diffusion_xl.png")


def test_text_to_image_stable_diffusion_2(output_dir):
    output_dir = output_dir / "test_text_to_image"
    output_dir.mkdir(parents=True, exist_ok=True)

    tti_sd2 = TextToImage("StableDiffusion2")
    i = tti_sd2(
        prompt   = "A drone is flying in the sky, photorealistic.",
        prompt_2 = "A drone is flying in the sky, photorealistic.",
        height   = 1024,
        width    = 1024,
        num_inference_steps = 50,
        guidance_scale = 5.0,
        negative_prompt = None,
        negative_prompt_2 = None,
        num_images_per_prompt = 2,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        output_type = "pil", # or np.array
        return_dict = True   # StableDiffusionXLOutput if False
    )
    assert len(i) == 2

    make_image_grid([i[0].resize((512, 512)), i[1].resize((512, 512))], rows=1, cols=2).save(output_dir/"stable_diffusion_2.png")

