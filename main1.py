import torch
from diffusers import StableDiffusionPipeline

class CFG:
    device = "cuda"  # Use "cuda" if GPU is available and compatible
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "CompVis/stable-diffusion-v1-4"  # Updated model ID
    hf_token = "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"  # Replace with your Hugging Face token

# Load the stable-diffusion model with Hugging Face token
pipe = StableDiffusionPipeline.from_pretrained(CFG.image_gen_model_id, use_auth_token=CFG.hf_token)
pipe = pipe.to(CFG.device)  # Move to device (CPU or GPU)

# Define the text prompt
prompt = "A beautiful sunset over a mountain range"

# Generate the image
with torch.autocast(CFG.device):
    image = pipe(prompt, generator=CFG.generator, num_inference_steps=CFG.image_gen_steps).images[0]

# Save or display the image
image.save("generated_image.png")
image.show()
