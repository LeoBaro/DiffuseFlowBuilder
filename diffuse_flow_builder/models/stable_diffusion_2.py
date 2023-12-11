import torch
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForInpainting,
    AutoPipelineForImage2Image
)
from diffusers import DiffusionPipeline, StableDiffusionUpscalePipeline
from compel import Compel


from diffuse_flow_builder.models.model_base import HuggingFaceModel
from diffuse_flow_builder.logger import logger

class StableDiffusion2(HuggingFaceModel):
    
    RECOMMENDED_PARAMS = {
        "height": 512,
        "width" : 512
    }
    # https://huggingface.co/stabilityai/stable-diffusion-2
    def __init__(self, task, load_refiner=False, seed=42, device="cuda", cuda_index=0, **kwargs):
        super().__init__(seed, device, cuda_index)

        self.model_id = "stabilityai/stable-diffusion-2"

        if task == "text_to_image":
            self.model = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                **kwargs
            )
        elif task == "in_painting":
            self.model = AutoPipelineForInpainting.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                **kwargs
            )
        elif task == "image_to_image":
            self.model = AutoPipelineForImage2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                **kwargs
            )
        elif task == "super_resolution":
            self.model = StableDiffusionUpscalePipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                torch_dtype=torch.float16,
                use_safetensors=True,
                **kwargs
            )

        if load_refiner:
            logger.warning("StableDiffusion2 does not support the refinement operation")

        self.generator = self.get_generator(seed, device, cuda_index)

        """
        self.compel = Compel(
            tokenizer=self.model.tokenizer,
            text_encoder=self.model.text_encoder
        )
        self.model_ready = False
        """
        
    def lazy_load(self):
        self.set_device()
        # The memory will not be released after the model is unloaded, so atm we don't use this
        # self.model.unet = torch.compile(self.model.unet, mode="reduce-overhead", fullgraph=True)
        self.model_ready = True
        logger.info("%s has been lazy-loaded!", self.model_id)

    def inference(self, **kwargs):
        if self.model_ready is False:
            self.lazy_load()

        if kwargs["override_with_recommended_parameters"]:
            logger.info(f"Overriding config params with {StableDiffusion2.RECOMMENDED_PARAMS}")
            for k,v in StableDiffusion2.RECOMMENDED_PARAMS.items():
                kwargs[k] = v

        logger.info("Making inference with kwargs='%s'", kwargs)

        return self.model(**kwargs, generator=self.generator).images, kwargs
    

