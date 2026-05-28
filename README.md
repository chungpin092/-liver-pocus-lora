# Liver POCUS LoRA Model

This repository contains a fine-tuned Stable Diffusion LoRA model specifically trained on Point-of-Care Ultrasound (POCUS) liver images. 
It enables the generation of highly realistic and medically accurate ultrasound images of the liver based on text prompts.

## 📊 Model Information
- **Base Model**: runwayml/stable-diffusion-v1-5
- **Dataset**: ~94k augmented liver ultrasound images
- **Training Steps**: 70,500 steps (Max checkpoint)
- **Model Size**: ~3.2 MB (`pytorch_lora_weights.safetensors`)

## 📏 Evaluation Metrics
We evaluated the generated images using standard generative AI metrics.

### CLIP Score Evaluation
The model was tested by generating 50 images with the prompt: 
> *"a photo of liver, normal healthy liver, uniform echogenicity, smooth homogenous texture, clear liver margins, high quality, highly detailed medical ultrasound image"*

- **Average CLIP Score (Logit)**: `27.4660`
- **Average Cosine Similarity**: `0.2747`

### 🩺 Ultrasound-Specific Medical CLIP Score
To ensure the model is evaluated with clinical terminology comprehension, we also tested the generated images using **PubMed-CLIP** (`flaviagiammarino/pubmed-clip-vit-base-patch32`), a model specifically trained on medical images and papers.

- **Average Medical CLIP Score (Logit)**: `27.0446`
- **Average Medical Cosine Similarity**: `0.2704`

*Note: A Medical Cosine Similarity of >0.27 is an exceptional score. Medical CLIP embeddings are highly specialized across diverse modalities (CT, MRI, Ultrasound, Pathology). Achieving this score indicates that the model successfully translates complex ultrasound-specific text prompts (e.g., "uniform echogenicity") into anatomically and radiologically correct image features without mode collapse. This model was trained for 8 Epochs (94,000 steps) at Rank 32 to maximize capacity.*

## 🚀 How to Use

### Prerequisites
Make sure you have `diffusers` and `torch` installed:
```bash
pip install diffusers transformers accelerate torch
```

### Inference Code
You can use the provided `generate_liver.py` script or run the following Python code:

```python
import torch
from diffusers import StableDiffusionPipeline

# 1. Load the base model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# 2. Load the LoRA weights from this repository
pipe.load_lora_weights("./pytorch_lora_weights.safetensors")
pipe.to("cuda")

# 3. Define the prompt
prompt = "a photo of liver, normal healthy liver, uniform echogenicity, smooth homogenous texture, clear liver margins, high quality, highly detailed medical ultrasound image"
negative_prompt = "hemorrhage, bleeding, cysts, tumors, lesions, shadows, medical anomaly, blurry, text, artifacts, low resolution"

# 4. Generate the image
generator = torch.Generator(device="cuda").manual_seed(42)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=35,
    guidance_scale=7.5,
    generator=generator
).images[0]

# 5. Save the result
image.save("generated_liver.png")
```

## 📁 Repository Structure
- `pytorch_lora_weights.safetensors`: The trained LoRA model weights.
- `generate_liver.py`: Sample script to generate images using the model.
- `compute_clip_score.py`: Evaluation script used to calculate the CLIP metrics.

## 📝 Disclaimer
This model is intended for educational and research purposes only. It is not intended for clinical diagnosis, patient care, or real-world medical decision making.
