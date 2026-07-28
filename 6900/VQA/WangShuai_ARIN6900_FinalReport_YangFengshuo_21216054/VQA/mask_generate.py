import torch
import numpy as np
import cv2
import json,os
from PIL import Image
from transformers import SamModel, SamProcessor

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def generate_sam_mask(image_path, bbox, output_mask_path):
    """
    使用 SAM 模型将矩形边界框 (Bounding Box) 转化为像素级精准的 Mask
    
    :param image_path: 原图路径
    :param bbox: 已知的矩形坐标 [xmin, ymin, xmax, ymax]
    :param output_mask_path: 输出 Mask 路径
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    raw_image = Image.open(image_path).convert("RGB")
    input_boxes = [[bbox]]
    inputs = processor(raw_image, input_boxes=input_boxes, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    # 5. 后处理提取高分辨率掩膜 (Mask)
    # SAM 会预测 3 种不同尺寸/粒度的 mask，我们直接取预测得分最高的那一张
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )
    # 提取最终的布尔矩阵 (True 代表物体像素，False 代表背景)
    best_mask = masks[0][0][0].numpy() 
    # 6. 将布尔矩阵转化为 OpenCV 可视化的黑底白色二值图 (0 和 255)
    mask_binary = (best_mask * 255).astype(np.uint8)

    cv2.imwrite(output_mask_path, mask_binary)
    print(f"✅ 精细 Mask 已成功生成并保存至: {output_mask_path}")

    del model, processor, inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    DATA_DIR = "/home/fyangbe/VQA/data"
    JSON_PATH = os.path.join(DATA_DIR, "vqa_original.json")

    if os.path.exists(JSON_PATH):
        data = load_json(JSON_PATH)
        image_id = data["image_id"]
        img_full_path = os.path.join(DATA_DIR, image_id)
    
        coordinates = data["object"]["coordinates"]
        if len(coordinates) > 0:
            first_obj = coordinates[0]
            bbox_list = [first_obj["xmin"], first_obj["ymin"], first_obj["xmax"], first_obj["ymax"]]

            output_mask_name = f"{image_id.split('.')[0]}_mask.png"
            output_mask_path = os.path.join(DATA_DIR, output_mask_name)

            generate_sam_mask(img_full_path, bbox_list, output_mask_path)
        else:
            print("⚠️ 提示：JSON中未检测到有效的物体坐标，无法生成 Mask")
    else:
         print("⚠️ 提示：未找到上游JSON文件")
