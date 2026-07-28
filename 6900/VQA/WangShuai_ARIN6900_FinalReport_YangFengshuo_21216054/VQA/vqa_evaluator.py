# coding: utf-8
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from vqa_model import get_vqa_model

# =====================================================================
DATASET_JSON_PATH = "/home/fyangbe/VQA/data/vqa_mutated.json" 
IMAGE_DIR_PATH = "/home/fyangbe/VQA/data"     
OUTPUT_RESULT_PATH = "/home/fyangbe/VQA/data/vqa_testing_results.json"

MODEL_NAME = "blip"         # 'vilt', 'blip', 'git'
BATCH_SIZE = 64             
NUM_WORKERS = 4            
THRES_CONF_DROP_POS = 0.30  
THRES_CONF_DROP_NEG = 0.30  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def find_image_path(image_dir, base_id, suffix=""):
    for ext in [".png"]:
        path = os.path.join(image_dir, f"{base_id}{suffix}{ext}")
        if os.path.exists(path):
            return path
    return os.path.join(image_dir, f"{base_id}{suffix}.png")

def get_number(ans_str):
    ans_clean = "".join([c for c in str(ans_str) if c.isdigit()])
    return int(ans_clean) if ans_clean else 0


# =====================================================================
class VQABatchDataset(Dataset):
    """JSON 数据集扁平化为单点推理任务流"""
    def __init__(self, json_path, image_dir):
        self.image_dir = image_dir
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)  

        if isinstance(raw_data, dict):
            raw_data = [raw_data]
         
        self.flat_tasks = []
        for case_idx, item in enumerate(raw_data):
            image_id = item["image_id"]
            is_human = item.get("is_human", True)
            orig_q = item["original_question"]
            mutations = item["mutations"]
            
            pos_muts = mutations.get("positive", [])
            neg_muts = mutations.get("negative", [])

            img_variants = ["orig", "remove"] if is_human else ["orig", "remove", "insert"]

            for var in img_variants:
                self.flat_tasks.append({
                    "case_idx": case_idx, "image_id": image_id, "is_human": is_human,
                    "question": orig_q, "task_type": "original", "variant": var,
                    "mut_type": None, "mut_idx": -1
                })
            
                for m_idx, mut_q in enumerate(pos_muts):
                    self.flat_tasks.append({
                        "case_idx": case_idx, "image_id": image_id, "is_human": is_human,
                        "question": mut_q, "task_type": "mutation", "variant": var,
                        "mut_type": "positive", "mut_idx": m_idx
                    })
                
                for m_idx, mut_q in enumerate(neg_muts):
                    self.flat_tasks.append({
                        "case_idx": case_idx, "image_id": image_id, "is_human": is_human,
                        "question": mut_q, "task_type": "mutation", "variant": var,
                        "mut_type": "negative", "mut_idx": m_idx
                    })


    def __len__(self):
        return len(self.flat_tasks)

    def __getitem__(self, idx):
        task = self.flat_tasks[idx]
        base_id = os.path.splitext(str(task['image_id']))[0]
        
        if task["variant"] == "orig":
            img_path = find_image_path(self.image_dir, base_id, "")
        elif task["variant"] == "remove":
            img_path = find_image_path(self.image_dir, base_id, "_remove")
        elif task["variant"] == "insert":
            img_path = find_image_path(self.image_dir, base_id, "_insert")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (300, 300), (0, 0, 0))
            
        return image, task["question"], task

def collate_fn(batch):
    images, questions, metas = [item[0] for item in batch], [item[1] for item in batch], [item[2] for item in batch]
    return images, questions, metas

# =====================================================================
def main():
    dataset = VQABatchDataset(json_path=DATASET_JSON_PATH, image_dir=IMAGE_DIR_PATH)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn,
        pin_memory=True  
    )

    model = get_vqa_model(MODEL_NAME, device=DEVICE)

    # 预准备一个内存字典，用于将 Batch 吐出的扁平数据重新拼回原层次结构
    # 结构: { case_idx: { "image_id":xx, "original":{}, "mutations_eval":[] } }
    reconstructed_results = {}

    print(f"\n 蜕变测试...", flush=True)
    for images, questions, metas in tqdm(dataloader):
        batch_outputs = model.predict_batch(images, questions)

        for (ans, conf), meta in zip(batch_outputs, metas):
            c_idx = meta["case_idx"]
            var = meta["variant"]

            if c_idx not in reconstructed_results:
                reconstructed_results[c_idx] = {
                    "image_id": meta["image_id"],
                    "is_human": meta["is_human"],
                    "orig": {"original": None, "positive": [], "negative": []},
                    "remove": {"original": None, "positive": [], "negative": []},
                    "insert": {"original": None, "positive": [], "negative": []}
                }

            container = reconstructed_results[c_idx][var]
            if meta["task_type"] == "original":
                container["original"] = {"question": meta["question"], "answer": ans, "confidence": conf}
            else:
                container[meta["mut_type"]].append({
                    "mut_idx": meta["mut_idx"], "mutated_question": meta["question"],
                    "predict_answer": ans, "confidence": conf
                })
            
    # =====================================================================
    print("\n 蜕变违背率...", flush=True)
    total_mutations_counted = 0
    total_violations_counted = 0
    detailed_cases_list = []

    for c_idx, case in sorted(reconstructed_results.items()):
        is_human = case["is_human"]
        log_case = {
            "image_id": case["image_id"],
            "is_human": is_human,
            "evaluation_results": {}
        }
        for v in ["orig", "remove", "insert"]:
            case[v]["positive"].sort(key=lambda x: x["mut_idx"])
            case[v]["negative"].sort(key=lambda x: x["mut_idx"])

        if is_human:
            for var in ["orig", "remove"]:
                base_info = case[var]["original"]
                if not base_info: continue
                base_ans, base_conf = base_info["answer"], base_info["confidence"]
                
                log_case["evaluation_results"][var] = {
                    "original_baseline": base_info, "positive_eval": [], "negative_eval": []
                }
                
                # Positive
                for mut in case[var]["positive"]:
                    conf_drop = base_conf - mut["confidence"]
                    # 置信度差异 30 以上或反转视为违规
                    is_violation = (mut["predict_answer"] != base_ans) or (conf_drop > THRES_CONF_DROP_POS)
                    
                    total_mutations_counted += 1
                    if is_violation: total_violations_counted += 1
                    
                    log_case["evaluation_results"][var]["positive_eval"].append({
                        **mut, "confidence_drop": conf_drop, "is_violation": is_violation
                    })
                    
                # Negative
                for mut in case[var]["negative"]:
                    conf_drop = base_conf - mut["confidence"]
                    # 没有发生反转或者置信度差异小于 30 视为违规
                    is_violation_cond = (mut["predict_answer"] == base_ans) and (conf_drop < THRES_CONF_DROP_NEG)
                    
                    if var == "orig":
                        total_mutations_counted += 1
                        is_violation = is_violation_cond
                        if is_violation: total_violations_counted += 1
                    else:
                        is_violation = False
                        
                    log_case["evaluation_results"][var]["negative_eval"].append({
                        **mut, "confidence_drop": conf_drop, 
                        "violation_condition_triggered": is_violation_cond,
                        "is_violation_counted": is_violation
                    })
        else:
            orig_base = case["orig"]["original"]
            if orig_base:
                orig_num_base = get_number(orig_base["answer"])
                orig_conf_base = orig_base["confidence"]
                
                for var in ["orig", "remove", "insert"]:
                    log_case["evaluation_results"][var] = {"positive_eval": [], "negative_eval": []}
                    
                    # 动态数字基准
                    if var == "orig":      expected_num = orig_num_base
                    elif var == "remove":  expected_num = orig_num_base - 1
                    elif var == "insert":  expected_num = orig_num_base + 1

                    all_qs = []
                    if case[var]["original"]:
                        all_qs.append({**case[var]["original"], "is_orig_q": True})
                    for m in case[var]["positive"]:
                        all_qs.append({**m, "is_orig_q": False})
                        
                    for q_item in all_qs:
                        pred_num = get_number(q_item.get("predict_answer", q_item.get("answer")))
                        
                        if var == "orig" and not q_item["is_orig_q"]:
                            # 考察数字改变或置信度突跌 30%
                            conf_drop = orig_conf_base - q_item["confidence"]
                            is_violation = (pred_num != expected_num) or (conf_drop > THRES_CONF_DROP_POS)
                        else:
                            conf_drop = 0.0
                            is_violation = (pred_num != expected_num)
                            
                        total_mutations_counted += 1
                        if is_violation: total_violations_counted += 1
                        
                        log_case["evaluation_results"][var]["positive_eval"].append({
                            "question": q_item.get("question", q_item.get("mutated_question")),
                            "predict_answer": q_item.get("predict_answer", q_item.get("answer")),
                            "confidence": q_item["confidence"],
                            "expected_number": expected_num,
                            "predicted_number": pred_num,
                            "is_violation": is_violation
                        })

        detailed_cases_list.append(log_case)

    global_violation_rate = (total_violations_counted / total_mutations_counted * 100) if total_mutations_counted > 0 else 0.0

    final_output = {
        "meta_summary": {
            "model_tested": MODEL_NAME,
            "total_test_points": total_mutations_counted,
            "total_violations": total_violations_counted,
            "global_violation_rate_percentage": global_violation_rate
        },
        "detailed_cases": detailed_cases_list
    }
    with open(OUTPUT_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

        
if __name__ == "__main__":
    main()