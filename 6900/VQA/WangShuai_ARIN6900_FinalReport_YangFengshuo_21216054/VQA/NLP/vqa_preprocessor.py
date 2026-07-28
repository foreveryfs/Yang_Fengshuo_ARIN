import json
import os,sys
import pickle
import gensim.downloader as api

from Mutator import AnalogyMutator, ActiveMutator, create_sentence_candidates

def main():

    input_vqa_json = "OpenEnded_abstract_v002_val2017_questions.json" 
    output_mutated_json = "vqa_mutated_questions.json"
    output_image_ids_txt = "vqa_mutated_image_ids.txt"
    
    MAX_SAVED_QUESTION = 10 


    word2vec_model = api.load("word2vec-google-news-300")

    ana_mutator = AnalogyMutator("gender", model=word2vec_model)
    act_mutator = ActiveMutator("gender")

    if not os.path.exists(input_vqa_json):
        print(f"错误: 文件是否存在！")
        sys.exit(1)

    with open(input_vqa_json, "r", encoding="utf-8") as f:
        vqa_data = json.load(f)

    # 获取 VQA 标准格式中的 questions 列表
    raw_questions = vqa_data.get("questions", [])

    processed_dataset = []
    saved_count = 0
    mutated_image_ids = set()

    for idx, item in enumerate(raw_questions):
        question_text = item["question"]
        image_id = item["image_id"]
        question_id = item["question_id"]

        ana_candidates, act_candidates, has_ana, has_act, _, _ = create_sentence_candidates(
            question_text, ana_mutator, act_mutator
        )

        all_mutations = list(ana_candidates) + list(act_candidates)

        # 只要知识图谱动态判定该句子“可突变”，且生成了突变体，我们就打包这条 VQA 任务
        if len(all_mutations) > 0:
            saved_count += 1
            print(f"[{saved_count}/{MAX_SAVED_QUESTION}] ")
            print(f"  - 原始问题: '{question_text}'")
            print(f"  - 生成突变数量: {len(all_mutations)} 个")

            # 严格保留 image_id 和 question_id，确保下一阶段 PyTorch 多模态推理时图片能完全对齐
            mutated_image_ids.add(image_id)

            processed_dataset.append({
                "question_id": question_id,
                "image_id": image_id,
                "original_question": question_text,
                "mutations": all_mutations  # 存储突变出来的衍生文本列表
            })

            if saved_count >= MAX_SAVED_QUESTION:
                break

        if idx % 1000 == 0 and idx > 0:
            print(f"已扫描 {idx} 条原始数据...", flush=True)

    print("--------------------------------------------------")
    with open(output_mutated_json, "w", encoding="utf-8") as f:
        json.dump(processed_dataset, f, indent=4, ensure_ascii=False)

    with open(output_image_ids_txt, "w", encoding="utf-8") as f:
        for img_id in sorted(list(mutated_image_ids)):
            f.write(f"{img_id}\n")    

if __name__ == "__main__":
    main()