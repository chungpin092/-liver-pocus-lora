import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Path to the 50 images
image_dir = r"C:\Users\AlanLab\Downloads\liver\generated_70k_samples"

# The exact prompt used to generate the images
prompt = "a photo of liver, normal healthy liver, uniform echogenicity, smooth homogenous texture, clear liver margins, high quality, highly detailed medical ultrasound image"

print("Loading CLIP model (openai/clip-vit-base-patch32)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Gather all image paths
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

if not image_files:
    print(f"No images found in {image_dir}")
    exit(1)

print(f"Found {len(image_files)} images. Calculating CLIP score...")

total_score = 0.0

# Calculate CLIP Score for each image
for img_path in image_files:
    image = Image.open(img_path).convert("RGB")
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Cosine similarity between image and text features (scaled by 100)
        # outputs.logits_per_image is equivalent to cosine_similarity * 100.0
        logits_per_image = outputs.logits_per_image
        score = logits_per_image.item()
        
    total_score += score

average_score = total_score / len(image_files)

print("=========================================")
print(f"Total Images Evaluated: {len(image_files)}")
print(f"Prompt: {prompt}")
print(f"Average CLIP Score (Logit scale): {average_score:.4f}")
# Cosine similarity is usually just logits / 100
print(f"Average Cosine Similarity: {average_score / 100.0:.4f}")
print("=========================================")
