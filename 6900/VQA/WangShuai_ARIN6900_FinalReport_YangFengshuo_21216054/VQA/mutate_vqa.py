import os
import json
import re


MALE_POOL = ["boy", "man", "black-man", "white-man", "asian-man", "gentleman", "male"]
FEMALE_POOL = ["girl", "woman", "black-woman", "white-woman", "asian-woman", "lady", "female"]
# 中性人类名词池
NEUTRAL_POOL = ["person", "child", "individual", "people", "human", "baby"]

ACTION_POOL = [
    "running", "sitting", "sleeping", "jumping", 
    "eating",  "riding",  "cooking", "reading" , 
    "playing football", "painting"
]

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, json_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def replace_word(text, target, replacement):
    pattern = r'\b' + re.escape(target) + r'\b'
    new_text, count = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
    if count > 0:
        return new_text
    
    if '-' in target:
        space_target = target.replace('-', ' ')
        pattern = r'\b' + re.escape(space_target) + r'\b'
        new_text, count = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
        if count > 0:
            return new_text
    elif ' ' in target:
        hyphen_target = target.replace(' ', '-')
        pattern = r'\b' + re.escape(hyphen_target) + r'\b'
        new_text, count = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
        if count > 0:
            return new_text
            
    return text

def generate_mutations(original, is_human):
    positive = []
    negative = []
    
    original_q = original.get("question", "")
    category_name = original.get("object", {}).get("category_name", "")

    if is_human:
        # =========================================================
        # 人类相关图像：主体变化 + 动作状态变化
        # =========================================================
        in_male = category_name in MALE_POOL
        in_female = category_name in FEMALE_POOL
        #in_neutral = category_name in NEUTRAL_POOL
        #if not (in_male or in_female or in_neutral):
        #    in_neutral = True

        if in_male:
            same_pool = MALE_POOL
            opp_pool = FEMALE_POOL
        elif in_female:
            same_pool = FEMALE_POOL
            opp_pool = MALE_POOL
        else:
            same_pool = NEUTRAL_POOL
            opp_pool = MALE_POOL + FEMALE_POOL

        for word in same_pool:
            if word != category_name:
                mutated = replace_word(original_q, category_name, word)
                if mutated != original_q and mutated not in positive:
                    positive.append(mutated)

        for word in opp_pool:
            mutated = replace_word(original_q, category_name, word)
            if mutated != original_q and mutated not in negative:
                    negative.append(mutated)

        found_action = original.get("object", {}).get("action")
        if found_action:
            for action in ACTION_POOL:
                if action != found_action:
                    mutated = replace_word(original_q, found_action, action)
                    if mutated != original_q and mutated not in negative:
                        negative.append(mutated)     
    else:
        # =========================================================
        # 非人类相关图像：句式变形
        # =========================================================     
        count = original.get("count")
        if count is None:
            count = original.get("object", {}).get("count", "1")

        count_str = str(count).strip()
        obj_str = category_name.strip()

        mutated = f"How many {obj_str} are there ?"
        if mutated not in positive:
            positive.append(mutated)    

    return positive,negative                    

def process_vqa_mutations(input_json_path, output_json_path):
    if not os.path.exists(input_json_path):
        print(f"❌ 找不到上游 JSON 文件: {input_json_path}")
        return
        
    upstream_data = load_json(input_json_path)
    
    if isinstance(upstream_data, dict):
        originals = [upstream_data]
        is_single = True
    else:
        originals = upstream_data
        is_single = False
        
    output_list = []
    
    for original in originals:
        question_id = os.path.splitext(original.get("question_id"))[0]
        image_id = question_id
        original_question = original.get("question", "")
        is_human = int(original.get("ishuman", 0)) == 1
        category_name = original.get("object", {}).get("category_name", "")

        positive, negative = generate_mutations(original, is_human)
        mutated_vqa = {
            "question_id": question_id,
            "image_id": image_id,
            "is_human": is_human,
            "original_question": original_question,
            "mutations": {
                "positive": positive,
                "negative": negative
            }
        }
        output_list.append(mutated_vqa)
        
    final_output = output_list[0] if is_single else output_list
    save_json(final_output, output_json_path)
    print(f"完成蜕变！已保存至: {output_json_path}")

if __name__ == "__main__":
    DATA_DIR = "/home/fyangbe/VQA/data"
    INPUT_PATH = os.path.join(DATA_DIR, "vqa_original.json")
    OUTPUT_PATH = os.path.join(DATA_DIR, "vqa_mutated.json")
    
    process_vqa_mutations(INPUT_PATH, OUTPUT_PATH)