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

class StableDiffusionXL(HuggingFaceModel):
    # https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
    # https://huggingface.co/docs/diffusers/using-diffusers/sdxl
    RECOMMENDED_PARAMS = {
        "height": 1024,
        "width" : 1024
    }
    def __init__(self, task, seed=42, device="cuda", cuda_index=0, **kwargs):
        super().__init__(seed, device, cuda_index)

        self.model_id         = "stabilityai/stable-diffusion-xl-base-1.0"
        self.refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

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

        self.refiner = None
        """
        self.generator = self.get_generator(seed, device, cuda_index)
        self.compel = Compel(
            tokenizer=[self.model.tokenizer, self.model.tokenizer_2] ,
            text_encoder=[self.model.text_encoder, self.model.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        """
        
    def lazy_load(self, apply_refinement=False):
        self.set_device()
        # The memory will not be released after the model is unloaded, so atm we don't use this
        # self.model.unet = torch.compile(self.model.unet, mode="reduce-overhead", fullgraph=True)
        self.model_ready = True
        logger.info("%s has been lazy-loaded!", self.model_id)
        if apply_refinement:
            self.refiner = DiffusionPipeline.from_pretrained(
                self.refiner_model_id,
                text_encoder_2=self.model.text_encoder_2,
                vae=self.model.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
        # The memory will not be released after the model is unloaded, so atm we don't use this
            # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)
            self.set_device_static(self.refiner, self.device, self.cuda_index)
            logger.info("%s has been lazy-loaded!", self.refiner_model_id)


    def inference(self, **kwargs):
        if self.model_ready is False:
            self.lazy_load(apply_refinement=kwargs["apply_refinement"])

        if kwargs["override_with_recommended_parameters"]:
            logger.info(f"Overriding config params with {StableDiffusionXL.RECOMMENDED_PARAMS}")
            for k,v in StableDiffusionXL.RECOMMENDED_PARAMS.items():
                kwargs[k] = v

        logger.info("Making inference with prompt='%s' image='%s'", kwargs["prompt"], kwargs["image"])

        if self.refiner is None:
            return self.model(**kwargs).images
        else:
            gen_images = self.model(
                denoising_end=0.8,
                output_type="latent",
                **kwargs
            ).images
            images_refined = [
                self.refiner(
                    prompt=kwargs["prompt"],
                    image=img,
                    denoising_start=0.8,
                    num_inference_steps=40
                ).images[0] for img in gen_images
            ]

            return images_refined
