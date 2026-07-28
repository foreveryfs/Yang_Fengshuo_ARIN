import os
import json
import random
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_image_abstract(image_id, is_abstract = 0):
    if is_abstract:
        return True
    else:
        return False 

def is_box_overlap(box1, box2):
    return not (box1[2] <= box2[0] or box1[0] >= box2[2] or 
                box1[3] <= box2[1] or box1[1] >= box2[3])

def find_blank_area(img_w, img_h, existing_boxes, target_w=120, target_h=120, max_tries=500):
    for _ in range(max_tries):
        xmin = random.randint(10, max(11, img_w - target_w - 10))
        ymin = random.randint(10, max(11, img_h - target_h - 10))
        xmax = xmin + target_w
        ymax = ymin + target_h
        candidate_box = [xmin, ymin, xmax, ymax]
        
        overlap = False
        for ex_box in existing_boxes:
            if is_box_overlap(candidate_box, ex_box):
                overlap = True
                break
        if not overlap:
            return candidate_box
    return [10, 10, 10 + target_w, 10 + target_h]

def execute_insertion(image_path, mask_image, prompt, is_cartoon, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if is_cartoon:
        model_id = "Sanster/anything-4.0-inpainting"
        final_prompt = f"abstract cartoon style, flat shading, 2D vector art, simple illustration, {prompt}"
        negative_prompt = "photorealistic, 3d render, realistic, hairy, textures"
    else:
        model_id = "runwayml/stable-diffusion-inpainting"
        final_prompt = f"a bright vivid {prompt}, photorealistic, high resolution, 8k, highly detailed, colorful"
        negative_prompt = "cartoon, anime, illustration, drawing, blurry, dark, pitch black, shadow"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True
    ).to(device)
    init_image = Image.open(image_path).convert("RGB")
    orig_size = init_image.size

    init_np = np.array(init_image)
    mask_np = np.array(mask_image.convert("L"))
    mask_binary = (mask_np > 0).astype(np.uint8)[:, :, None] # (H, W, 1)

    fill_color = np.array([100, 160, 100], dtype=np.uint8)
    pre_filled_np = init_np * (1 - mask_binary) + (mask_binary * fill_color)
    pre_filled_image = Image.fromarray(pre_filled_np)

    edited_image = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        image=pre_filled_image.resize((512, 512)), 
        mask_image=mask_image.resize((512, 512)),
        num_inference_steps=35,
        guidance_scale=13.0,                    
    ).images[0]

    edited_image.resize(orig_size).save(output_path)
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
        
        is_cartoon = check_image_abstract(image_id)
        print(f"Image: {image_id}")

        coordinates = data["object"]["coordinates"]
        parsed_boxes = [[b["xmin"], b["ymin"], b["xmax"], b["ymax"]] for b in coordinates]

        orig_img = Image.open(img_path)
        W, H = orig_img.size
        
        blank_box = find_blank_area(W, H, parsed_boxes, target_w=130, target_h=130)

        blank_mask_np = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(blank_mask_np, (blank_box[0], blank_box[1]), (blank_box[2], blank_box[3]), 255, -1)
        blank_mask_img = Image.fromarray(blank_mask_np)

        # Object
        target_new_obj = "a green apple" 
        output_path = os.path.join(DATA_DIR, f"{image_id.split('.')[0]}_insert.png")
        
        execute_insertion(img_path, blank_mask_img, target_new_obj, is_cartoon, output_path)
    else:
        print(f"❌ 未找到上游 JSON 文件: {JSON_PATH}")