import os
import json
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import cv2

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_image_abstract(image_id, is_abstract = 0):
    if is_abstract:
        return True
    else:
        return False 

def execute_removal(image_path, mask_path, is_cartoon, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    category_name = data["object"]["category_name"]

    if is_cartoon:
        model_id = "Sanster/anything-4.0-inpainting"
        prompt = "clean background, empty space, flat shading, 2D vector art, solid background color, completely empty background"
        negative_prompt = f"photorealistic, realistic, 3d render, textures, {category_name}, object, foreground, item, character, residue shape"
    else:
        model_id = "runwayml/stable-diffusion-inpainting"
        prompt = "clean background, seamlessly filled, matching surroundings, no more object"
        negative_prompt = f"cartoon, anime, illustration, blurry, objects, {category_name}, fruit"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True
    ).to(device)

    init_image = Image.open(image_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("RGB")
    orig_size = init_image.size

    edited_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image.resize((512, 512)),
        mask_image=mask_image.resize((512, 512)),
        num_inference_steps=35,
        strength=0.75,
    ).images[0]

    edited_image.resize(orig_size).save(output_path)
    print(f"✅ 对象移除成功！已保存至: {output_path}")
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    DATA_DIR = "/home/fyangbe/VQA/data"
    JSON_PATH = os.path.join(DATA_DIR, "vqa_original.json")

    if os.path.exists(JSON_PATH):
        data = load_json(JSON_PATH)
        image_id = data["image_id"]
        
        img_path = os.path.join(DATA_DIR, image_id)
        mask_path = os.path.join(DATA_DIR, f"{image_id.split('.')[0]}_mask.png")
        output_path = os.path.join(DATA_DIR, f"{image_id.split('.')[0]}_remove.png")

        if os.path.exists(mask_path):
            is_cartoon = check_image_abstract(image_id)
            print(f"[移除模块] 图像: {image_id} ")
            execute_removal(img_path, mask_path, is_cartoon, output_path)
        else:
            print(f"❌ 未找到对应的掩膜文件{mask_path}")
    else:
        print(f"❌ 未找到上游 JSON 文件: {JSON_PATH}")