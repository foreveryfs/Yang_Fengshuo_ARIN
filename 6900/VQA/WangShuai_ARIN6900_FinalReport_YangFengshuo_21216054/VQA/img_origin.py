import torch
from PIL import Image
from transformers import pipeline
import os
import json

def run_owl_with():
    model_name = "google/owlv2-base-patch16-ensemble"
    data_dir = "/home/fyangbe/VQA/data"
    input_json_path = os.path.join(data_dir, "img2question.json")
    output_json_path = os.path.join(data_dir, "vqa_original.json")

    if not os.path.exists(input_json_path):
        print(f"❌ 错误：未找到前置 JSON 文件 {input_json_path}！")
        return
    with open(input_json_path, "r", encoding="utf-8") as f:
        qwen_data = json.load(f)
    
    image_id = qwen_data.get("image_id")
    ishuman = int(qwen_data.get("ishuman", 0))
    target_obj = qwen_data.get("target_object")

    if not target_obj:
        print("❌ 错误：未包含有效的 'target_object'！")
        return

    category_name = target_obj.get("category_name", "").strip()
    action = target_obj.get("action", "")
    if action: 
        action = action.strip()
        
    if not category_name or category_name.lower() == "null":
        print("⚠️ 提示：目标类别为空，无法执行问答生成。")
        return

    # 动态定位原图路径
    image_path = os.path.join(data_dir, image_id)
    if not os.path.exists(image_path):
        print(f"❌ 错误：在指定路径找不到图像文件 {image_path}")
        return

    #OWL2 model
    device = 0 if torch.cuda.is_available() else -1
    detector = pipeline(
        task="zero-shot-object-detection",
        model=model_name,
        local_files_only=True,
        device=device
    )

    image = Image.open(image_path).convert("RGB")

    # 如采用大词库
    """ vocab_path = "/home/fyangbe/VQA/data/lvis_vocab.txt"
    if not os.path.exists(vocab_path):
        print(f"❌ 错误：找不到外部词库文件 {vocab_path}，请先生成该文件！")
        return
        
    with open(vocab_path, "r", encoding="utf-8") as f:
        candidate_labels = [line.strip() for line in f if line.strip()]
    
    print(f"📚 成功从外部注入大词库！当前总计类别数: {len(candidate_labels)}") """

    # 4. 执行推理
    predictions = detector(
        image,
        candidate_labels=[category_name, f"a cartoon {category_name}", f"a photo of a {category_name}"],
        threshold=0.42, 
    )

    # ==标准类无关 NMS 去重逻辑 ====================
    # 1. 首先确保所有预测框按置信度（Score）从高到低严格排序
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
    keep_predictions = []

    # 定义标准的 IoU 计算函数
    def calculate_iou(box1, box2):
        # 计算重叠区域的坐标
        xmin = max(box1["xmin"], box2["xmin"])
        ymin = max(box1["ymin"], box2["ymin"])
        xmax = min(box1["xmax"], box2["xmax"])
        ymax = min(box1["ymax"], box2["ymax"])
        # 如果没有重叠，直接返回 0
        if xmax <= xmin or ymax <= ymin:
            return 0.0
        intersection = (xmax - xmin) * (ymax - ymin)
        # 计算两个框各自的面积
        area1 = (box1["xmax"] - box1["xmin"]) * (box1["ymax"] - box1["ymin"])
        area2 = (box2["xmax"] - box2["xmin"]) * (box2["ymax"] - box2["ymin"])
        # 标准分母：并集面积 (Area A + Area B - Intersection)
        union = area1 + area2 - intersection
        return intersection / union
    # 开始筛选
    for pred in predictions:
        box = pred["box"]
        is_duplicate = False
        for kept in keep_predictions:
            k_box = kept["box"]
            iou = calculate_iou(box, k_box)
            # 💡 核心策略：只要两个框的重合度 IoU 超过 0.45（哪怕它们名字不同，比如一个是 squirrel 一个是 rodent）
            # 由于我们已经提前排过序，当前进来的这个框分数一定比里面的低，说明它是同一个位置的“次优猜测”
            if iou > 0.45: 
                is_duplicate = True
                break   
        if not is_duplicate:
            keep_predictions.append(pred)

    all_coordinates = []
    for pred in keep_predictions:
        box = pred["box"]
        all_coordinates.append({
            "xmin": int(box["xmin"]),
            "ymin": int(box["ymin"]),
            "xmax": int(box["xmax"]),
            "ymax": int(box["ymax"])
        })
    
    if ishuman == 1:
        question = f"Is the {category_name} {action}?"
    else:
        question = f"What is the total number of {category_name}?"

    # 5. 打印结果
    vqa_original_data = {
        "image_id": image_id,
        "ishuman": ishuman,
        "question_id": image_id, 
        "question": question,
        "object": {
          "category_name": category_name,
          "action": action if ishuman == 1 else None,
          "count": len(keep_predictions),  # OWL 纠正后的真实数量
          "coordinates": all_coordinates   # 包含的全部坐标
        }
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(vqa_original_data, f, indent=2, ensure_ascii=False)   

if __name__ == "__main__":
    run_owl_with()