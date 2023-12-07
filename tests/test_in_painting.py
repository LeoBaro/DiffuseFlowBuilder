from diffuse_flow_builder.components.in_painting import InPainting
from diffusers.utils import load_image, make_image_grid

def test_in_painting_stable_diffusion_xl(data_dir, output_dir):
    data_dir = data_dir / "test_in_painting"
    output_dir = output_dir / "test_in_painting"
    output_dir.mkdir(parents=True, exist_ok=True)

    ip_sdxl = InPainting("StableDiffusionXL")
    
    init_image = load_image(str(data_dir / "init.png"))
    init_image = init_image.resize((1024, 1024))

    mask_image = load_image(str(data_dir / "mask.png"))
    mask_image = mask_image.resize((1024, 1024))

    i = ip_sdxl(
        prompt     = "A drone is flying in the sky, photorealistic.",
        prompt_2   = "A drone is flying in the sky, photorealistic.",
        image      = init_image,
        mask_image = mask_image,
        strength   = 0.85,
        guidance_scale = 12.5,
        num_inference_steps = 50,
        height   = 1024,
        width    = 1024,
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


def test_in_painting_stable_diffusion_2(data_dir, output_dir):
    data_dir = data_dir / "test_in_painting"
    output_dir = output_dir / "test_in_painting"
    output_dir.mkdir(parents=True, exist_ok=True)

    ip_sd2 = InPainting("StableDiffusion2")
    
    init_image = load_image(str(data_dir / "init.png"))
    init_image = init_image.resize((1024, 1024))

    mask_image = load_image(str(data_dir / "mask.png"))
    mask_image = mask_image.resize((1024, 1024))

    i = ip_sd2(
        prompt     = "A drone is flying in the sky, photorealistic.",
        image      = init_image, 
        mask_image = mask_image, 
        guidance_scale = 12.5,
        num_inference_steps = 50,
        height   = 1024,
        width    = 1024,
        negative_prompt = None,
        num_images_per_prompt = 2,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        output_type = "pil", # or np.array 
        return_dict = True,   # StableDiffusionXLOutput if False
    )
    assert len(i) == 2

    make_image_grid([i[0].resize((512, 512)), i[1].resize((512, 512))], rows=1, cols=2).save(output_dir/"stable_diffusion_2.png")
