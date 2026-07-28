import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os

# ==========================================
# 1. 初始化模型与处理器 (直接从本地 Cache 加载)
# ==========================================
model_id = "Qwen/Qwen2-VL-2B-Instruct"

print("🔮 正在从缓存中载入 Qwen2-VL-2B-Instruct...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# ==========================================
# 2.System Prompt
# ==========================================
system_prompt = (
    "You are a strict image-to-JSON converter. Output ONLY a valid JSON object matching the schema. "
    "Do not include markdown backticks (```json) or any conversational text. Start with '{' and end with '}'.\n\n"
    
    "📋 CONSTRAINTS:\n"
    "1. Pick EXACTLY ONE primary target object category per image.\n"
    "2. If the target is a HUMAN (girl, boy, man, woman, person) -> 'ishuman' is 1, and extract their 'action'.\n"
    "3. If the target is an ANIMAL or OBJECT -> 'ishuman' is 0, and 'action' MUST strictly be null. Never give actions to animals.\n\n"
    
    "Strict Output JSON Schema:\n"
    "{\n"
    '  "ishuman": 1 or 0,\n'
    '  "target_object": {\n'
    '    "category_name": "string",\n'
    '    "action": "string" or null,\n'
    '    "count": integer\n'
    "  }\n"
    "}\n\n"
    
    "🎯 EXAMPLE 1 (Human Case):\n"
    "Input image shows a boy swimming.\n"
    "Output:\n"
    "{\n"
    '  "ishuman": 1,\n'
    '  "target_object": {\n'
    '    "category_name": "boy",\n'
    '    "action": "swimming",\n'
    '    "count": 1\n'
    "  }\n"
    "}\n\n"
    
    "🎯 EXAMPLE 2 (Non-Human Case):\n"
    "Input image shows three running chipmunks.\n"
    "Output:\n"
    "{\n"
    '  "ishuman": 0,\n'
    '  "target_object": {\n'
    '    "category_name": "chipmunks",\n'
    '    "action": null,\n'
    '    "count": 3\n'
    "  }\n"
    "}"
)

# ==========================================
# 3. 消息输入 
image_path = "/home/fyangbe/VQA/data/3.png"
output_file_path = "/home/fyangbe/VQA/data/img2question.json"
image_id = os.path.basename(image_path)
user_question = "Return the structured JSON for this image according to the schema."

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user_question}
        ]
    }
]

# ==========================================
# 4. 数据预处理
# ==========================================
# 渲染文本模板
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# 提取视觉特征输入
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)
# 将输入移动到模型对应的显卡上
inputs = inputs.to(model.device)

# ==========================================
# 5. 模型推理生成
# ==========================================
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# ==========================================
# 6. 结果清洗与展示
# 尝试解析成标准的 Python 字典，验证 JSON 合法性
try:
    # 移除可能残留的 ```json 或 ``` 标记（以防万一）
    clean_text = output_text.replace("```json", "").replace("```", "").strip()
    json_data = json.loads(clean_text)

    final_output = {
        "image_id": image_id
    }
    final_output.update(json_data) # 将模型生成的其他字段合并进来

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print("\n✅ JSON 格式验证成功！")

except json.JSONDecodeError:
    print("\n❌ 警告: 模型输出的不是标准的 JSON 格式")
    print(output_text)