# Diffuse Flow Builder

DiffuseFlowBuilder is a versatile YAML-based configuration repository designed to streamline the creation and execution of stable diffusion pipelines. This powerful tool empowers users to effortlessly define and manage pipelines for essential diffusion operations, including text-to-image conversion, in-painting, and super-resolution.

## Installation
Create a conda virtual environment if you don't have Python 3.10 installed in your machine
```bash
conda env create --file environment.yml 
```

Build the software:
```bash
conda activate diffflow
pip install -e .
```

## Usage
Describe a diffusion pipeline with a yaml configuration and execute:
```python
PipelineRunner()
    .run_pipeline(
        "./pipeline.yaml", 
        PipelineBuilder().build_pipeline(
            "./pipeline.yaml", 
            "t2i_pipeline"
        )
    )
```

## Configuration file
```yaml
pipelines:

  t2i_ip_pipeline:
    
    n_runs: 2 # number of times the pipeline will run
    write_results_after_x_run: 1 # todo be implemented (to avoid output loss in case of a crash)
    save_intermediate_results: False # todo be implemented (debug purposes)
    distributed: False # todo be implemented (distributed inference)
    output_format: "png"
    
    steps:

      - name: TextToImage # The name define the task
        model: StableDiffusionXL # The model to use
        prompt: # prompt is splitted in different parts to allow flexibility with random prompt generation
          prompt_prefix: "an image of a"
          prompt_subject: "city skyline"
          prompt_enanchment: "cyberpunk style, hyper realistic, 8K"
        use_prompt_from_previous_step: False # the second step of the pipeline may want to use the output of the previous step
        combine_prompt_with_previous_step: False # the second step of the pipeline may want to use the output of the previous step. In this case, if prompt_prefix, prompt_subject or prompt_enanchment is null, it will be overridden from the corresponding prompt part of the previous step. 
        apply_refinement: False # only applies to StableDiffusionXL model 
        strength: 0.85 # only applies to StableDiffusionXL model
        guidance_scale: 5 # tune it depending on the model
        num_inference_steps: 50 # tune it depending on the model
        height: 1024 # tune it depending on the model
        width: 1024 # tune it depending on the model
        num_images_per_prompt: 1 # greater the number, greater the memory requirement
        output_dir: $ROOT/output # where $ROOT is expanded (root of repository) 

      - name: InPainting
        model: StableDiffusionXL
        image: null
        use_image_from_previous_step: True
        prompt:
          prompt_prefix: null
          prompt_subject: "flying drone"
          prompt_enanchment: null
        use_prompt_from_previous_step: False
        combine_prompt_with_previous_step: True
        apply_refinement: False
        strength: 0.85
        guidance_scale: 12.5
        num_inference_steps: 50
        height: 1024
        width: 1024
        num_images_per_prompt: 1
        mask_image: null # a path to an image 
        use_random_masks: $ROOT/data/drone_masks/masks # a directory containing n images   
        output_dir: $ROOT/output

  
  t2i_sr_pipeline:
    n_runs: 5
    write_results_after_x_run: 1
    save_intermediate_results: False
    distributed: False
    output_format: "png"
    steps:

      - name: TextToImage
        model: StableDiffusion2
        prompt:
          prompt_prefix: "an image of an"
          prompt_subject: "cat"
          prompt_enanchment: "cyberpunk style"
        apply_refinement: null
        strength: null
        guidance_scale: 15
        num_inference_steps: 50
        height: 256
        width: 256
        output_dir: $ROOT/output

      - name: SuperResolution
        model: StableDiffusionXL
        image: null
        use_image_from_previous_step: True
        prompt: null
        use_prompt_from_previous_step: True
        combine_prompt_with_previous_step: False
        guidance_scale: 15
        num_inference_steps: 50
        num_images_per_prompt: 1
        output_dir: $ROOT/output

```

## Tests
Launch tests with:
```bash
pytest --disable-warnings -xsvv --pdb tests/test_pipeline_runner.py -k t2i_ip_pipeline
```