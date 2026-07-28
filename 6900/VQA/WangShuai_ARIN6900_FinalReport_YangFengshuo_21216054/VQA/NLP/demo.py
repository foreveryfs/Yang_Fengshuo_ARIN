#!/usr/bin/env python
import time
import argparse
import re
import os
parser = argparse.ArgumentParser(description='Mutate sentence and mitigate violations')

parser.add_argument('-s', help='file of seed sentence')
parser.add_argument('-k', type=int, help='mitigation parameter k')
parser.add_argument('-e', type=float, help='mitigation parameter epsilon')

args = parser.parse_args()

CSV_FILE_PATH = args.s          # 
CHUNK_SIZE = 128                # 提高分块大小，纯计数不占内存，可加速 I/O

import nltk
# nltk
nltk.data.path.insert(0, '/home/fyangbe/nltk_data')
import pandas as pd
import pickle, os
from gensim.models import KeyedVectors
from Mutator import AnalogyMutator, ActiveMutator, create_sentence_candidates
# Test 1
from fluency_scorer import FluencyScorer
import pickle
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import tensorflow as tf
import numpy as np



# load word2vec embedding
print("word2vec model loading")
# 修改为本地路径：
word2vec = KeyedVectors.load_word2vec_format('./dependency/GoogleNews-vectors-negative300.bin', binary=True)
# word2vec = api.load("word2vec-google-news-300")
print("word2vec model loaded")

# Test 1
# load scorer
p, d = "dependency/wiki103/wiki103.pt", "dependency/wiki103"
fluency_scorer = FluencyScorer(p, d)

# load cnn model
VOCAB_SIZE = 30000
MAX_LEN = 500

model = load_model('model/cnn_model.h5')
with open('model/cnn_model_tokenizer.pickle', 'rb') as f:
    imdb_tokenizer = pickle.load(f)

ana = AnalogyMutator("gender", model=word2vec)
act = ActiveMutator("gender")

# ======文本清洗工具 ====================
def clean_content(text):
    # 1. 移除可能残留的各种格式的 <br /> 标签
    text = re.sub(r'<\s*br\s*/?\s*>', ' ', text)
    
    # 2. 核心修复：把标点符号（. ! ?）前面的多余空格删掉，让 NLTK 能够正确识别句尾
    text = re.sub(r'\s+([\.!\?])', r'\1', text)
    
    # 3. 把连续的多个空格合并为一个空格
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def mutate(sentence, epsilon):
    sentence = clean_content(sentence)
    if not sentence:
        return 0, 0.0, 0, 0, 0, 0, 0, 0, 0
    
    ana_candidates, act_candidates, has_ana, has_act, ana_cnt, act_cnt = create_sentence_candidates(sentence, ana, act)
    # 保留原始所属类别标记
    ana_list = list(ana_candidates)
    act_list = list(act_candidates)
    #candidates = list(ana_candidates) + list(act_candidates)

    is_mutated = 1 if (len(ana_list) + len(act_list)) > 0 else 0

    if (len(ana_list) + len(act_list)) > args.k:
        # Test 1
        candidates_with_type = []
        for c in ana_list:
            candidates_with_type.append((fluency_scorer.score_sentence(c).item(), c, 'ana'))
        for c in act_list:
            candidates_with_type.append((fluency_scorer.score_sentence(c).item(), c, 'act'))

        candidates_with_type.sort(key=lambda x:x[0], reverse=True)
        final_selected = candidates_with_type[:args.k]

        # 重新提取截断后的候选句子
        candidates = [item[1] for item in final_selected]
        # 统计截断后真正保留下来的两类用例数
        final_ana_cnt = sum([1 for item in final_selected if item[2] == 'ana'])
        final_act_cnt = sum([1 for item in final_selected if item[2] == 'act'])
        final_has_ana = 1 if final_ana_cnt > 0 else 0
        final_has_act = 1 if final_act_cnt > 0 else 0
    
    else:
        # 如果没超过 k，直接使用全部
        candidates = ana_list + act_list
        final_ana_cnt = ana_cnt
        final_act_cnt = act_cnt
        final_has_ana = has_ana
        final_has_act = has_act

        #candidates_with_fluency = [(fluency_scorer.score_sentence(c).item(), c) for c in candidates]
        # candidates_with_fluency = [(1.0, c) for c in candidates]
        #candidates_with_fluency.sort(key=lambda x:x[0], reverse=True)
        #candidates = [c[1] for c in candidates_with_fluency[:args.k]]
    if len(candidates) == 0:
        return 0, 0.0, 0, 0, is_mutated, final_has_ana, final_has_act, final_ana_cnt, final_act_cnt
    
    sentences_seq = imdb_tokenizer.texts_to_sequences([sentence] + candidates)
    sentences_vec = sequence.pad_sequences(sentences_seq, maxlen=MAX_LEN, padding="post", value=0)
    predictions = model.predict(sentences_vec) - .5
    original_score, testcase_scores = predictions[0], predictions[1:]

    # Single sentence
    #print("# testcases (k): ", len(candidates))
    #print("original score: ", original_score + .5)
    #print(sentence)
    #num_violations = sum([1 for i in range(len(testcase_scores)) if testcase_scores[i] * original_score < 0])
    #mit_num = mitigation(predictions, epsilon)
    #print("# violations: ", num_violations)
    #print("# violations (after mitigation): ", mitigation(predictions, epsilon))

    # Batch size
    num_violations = sum([1 for i in range(len(testcase_scores)) if testcase_scores[i] * original_score < 0])
    mit_num = mitigation(predictions, epsilon)
    return len(candidates), original_score, num_violations, mit_num, is_mutated, final_has_ana, final_has_act, final_ana_cnt, final_act_cnt
    

"""def mitigation(predictions, epsilon = 2):
    num_violations = 0
    ground_truth = predictions[0]
    k = len(predictions)
    for i in predictions[1:]:
        score = np.sum(predictions) / (np.e**epsilon + k) + np.e**epsilon * i / (np.e**epsilon + k)
        if score * ground_truth < 0:
            num_violations += 1
    return num_violations"""
def mitigation(predictions, epsilon=2):
    num_violations = 0
    ground_truth = predictions[0]  # 原始句子的情感分数
    testcase_scores = predictions[1:]  # 所有扰动句子的分数
    k = len(testcase_scores)
    exp_eps = np.exp(epsilon)

    for test_score in testcase_scores:
        # 正确的缓解公式：原始句子 + 扰动句子的加权平均
        mitigated_score = (ground_truth + exp_eps * test_score) / (1 + exp_eps)
        
        # 判断缓解后是否仍然违规
        if mitigated_score * ground_truth < 0:
            num_violations += 1
    return num_violations

# Batch size or Single sentence
""" try:
    start_time = time.time()

    total_processed_reviews = 0  
    total_natural_sentences = 0   
    total_mutated_sentences = 0    
    total_cases = 0               
    total_violations = 0          
    total_mitigated = 0    

    print("# Dataset mode running: ", CSV_FILE_PATH)
    csv_reader = pd.read_csv(CSV_FILE_PATH, header=0, usecols=[0], names=['text'], chunksize=CHUNK_SIZE) 
    for chunk_idx, df_chunk in enumerate(csv_reader):
        chunk_start = time.time()    

        for review in df_chunk['text']: 
            review = str(review).strip()
            if not review or review.lower() == 'nan':
                continue
            total_processed_reviews += 1 

            cleaned_review = clean_text(review)
            review_sentences = [s for s in nltk.sent_tokenize(cleaned_review) if s.strip()]

            for sent in review_sentences:
                total_natural_sentences += 1
                cases, orig_score, vio, mit, ismutated = mutate(sent, args.e)

                if total_natural_sentences % 500 == 0:
                    current_time = time.time()
                    total_elapsed = current_time - start_time
                    print(f"Processed: {total_natural_sentences} sentences | "
                        f"Current_Time: {total_elapsed/60:.2f} min | Current_Violations: {total_violations}")

                total_cases += cases
                total_violations += vio
                total_mitigated += mit
                total_mutated_sentences += ismutated

        chunk_elapsed = time.time() - chunk_start            

    end_time = time.time()
    total_elapsed_mins = (end_time - start_time) / 60

    print("\n" + "="*6)
    print(f"Total Time: {total_elapsed_mins:.2f} min")
    print(f"Total Reviews: {total_processed_reviews}")
    print(f"Total Sentences: {total_natural_sentences}")
    print(f"Mutated Sentences: {total_mutated_sentences}")
    print(f"Total Test Cases: {total_cases}")
    print(f"Original Violations: {total_violations}")
    print(f"Mitigated Violations: {total_mitigated}")

    if total_violations > 0:
        reduction = (total_violations - total_mitigated) / total_violations
        print(f"Bias Reduction Rate: {reduction:.2%}")
    else:
        print(f"Bias Reduction Rate: N/A")
    print("="*6)

except Exception as e:
    print("\nError:", e) """


try:
    with open(CSV_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        full_text = f.read()

    cleaned_text = clean_content(full_text)
    sentences = [s.strip() for s in nltk.sent_tokenize(cleaned_text) if s.strip()]    

    if len(sentences) > 1:
        print("# Batch mode running:", CSV_FILE_PATH)

        total_cases = 0
        total_violations = 0
        total_mitigated = 0
        total_sentences = len(sentences)
        total_mutatedsentences = 0
        total_ana_sentences = 0
        total_act_sentences = 0
        total_ana_cases = 0
        total_act_cases = 0

        for idx, sent in enumerate(sentences):
            (cases, orig_score, vio, mit, is_mutated, 
             has_ana, has_act, ana_cnt, act_cnt) = mutate(sent, args.e)
            total_cases += cases
            total_violations += vio
            total_mitigated += mit
            total_mutatedsentences += is_mutated
            total_ana_sentences += has_ana
            total_act_sentences += has_act
            total_ana_cases += ana_cnt
            total_act_cases += act_cnt

            if (idx + 1) % 500 == 0:
                print(f"Done {idx+1}/{total_sentences} | total_violations: {total_violations}")

        print("\n======")
        print(f"total_sentences: {total_sentences}")
        print(f"total_mutated_sentences: {total_mutatedsentences}")
        print(f"total_cases: {total_cases}")
        print(f"total_ana_cases: {total_ana_cases}")
        print(f"total_act_cases: {total_act_cases}")
        print(f"total_violations: {total_violations}")
        print(f"total_mitigated: {total_mitigated}")
        print("======")

    else:
        # 单行句子 = 原来的 demo 模式
        s = sentences[0] if sentences else ""
        print("# Demo mode running:", CSV_FILE_PATH)
        cases, orig_score, vio, mit, _, _, _, _, _ = mutate(s, args.e)
        print("\n======")
        print("# testcases (k): ", cases)
        print("original score: ", orig_score + 0.5)
        print("# violations: ", vio)
        print("# violations (after mitigation): ", mit)
        print("======")

except Exception as e:
    print("Error reading file:", e)     