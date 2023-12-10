from diffuse_flow_builder.models.stable_diffusion_xl import StableDiffusionXL
import torch

from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForInpainting,
    AutoPipelineForImage2Image,
    DiffusionPipeline
)
from compel import Compel, ReturnedEmbeddingsType

from diffuse_flow_builder.models.model_base import HuggingFaceModel
from diffuse_flow_builder.logger import logger

class StableDiffusionXLTurbo(HuggingFaceModel):
    # https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
    # https://huggingface.co/docs/diffusers/using-diffusers/sdxl
    RECOMMENDED_PARAMS = {
        "height": 512,
        "width" : 512,
        "guidance_scale": 0.0
    }
    def __init__(self, task, seed=42, device="cuda", cuda_index=0, **kwargs):
        super().__init__(seed, device, cuda_index)

        self.model_id         = "stabilityai/sdxl-turbo"

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
        elif task == 'image_to_image':
            self.model = AutoPipelineForImage2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                **kwargs
            )

        
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
            logger.info(f"Overriding config params with {StableDiffusionXLTurbo.RECOMMENDED_PARAMS}")
            for k,v in StableDiffusionXLTurbo.RECOMMENDED_PARAMS.items():
                kwargs[k] = v

        logger.info("Making inference with prompt='%s' image='%s'", kwargs["prompt"], kwargs["image"])

        return self.model(**kwargs).images
