import torch
from diffusers import StableDiffusionPipeline
import os
from tqdm import tqdm

model_id = "runwayml/stable-diffusion-v1-5"
lora_path = r"C:\Users\AlanLab\Downloads\liver\liver_lora_output\checkpoint-70500"
output_dir = r"C:\Users\AlanLab\Downloads\liver\generated_70k_samples"

os.makedirs(output_dir, exist_ok=True)

print("Loading base model...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

print("Loading 70k+ LoRA weights (checkpoint-70500)...")
pipe.load_lora_weights(lora_path)
pipe.to("cuda")

prompt = "a photo of liver, normal healthy liver, uniform echogenicity, smooth homogenous texture, clear liver margins, high quality, highly detailed medical ultrasound image"
negative_prompt = "hemorrhage, bleeding, cysts, tumors, lesions, shadows, medical anomaly, blurry, text, artifacts, low resolution"

print(f"Generating 50 images... (Prompt: {prompt})")

for i in tqdm(range(50)):
    # Use a different seed for each image
    generator = torch.Generator(device="cuda").manual_seed(i)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=35,
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    
    output_path = os.path.join(output_dir, f"generated_liver_70k_{i:02d}.png")
    image.save(output_path)

print(f"Success! 50 images saved to {output_dir}")
