import numpy as np
import pickle
import nltk.stem as ns
from stanfordcorenlp import StanfordCoreNLP
from config import *

# ====================================================
# Local database
print("Loading local ConceptNet database...")
try:
    with open('dependency/conceptnet_mini.pickle', 'rb') as f:
        # 结构应包含: { 'word': { 'isa': [], 'synonym': [], 'antonym': [], 'formof': [] } }
        CONCEPTNET_DB = pickle.load(f)
    print("Local ConceptNet loaded.")
except FileNotFoundError:
    print("Error: dependency/conceptnet_mini.pickle not found! Please upload the file.")
    CONCEPTNET_DB = {}

def query_local_kg(token, rel_key):
    """local as api"""
    word = token.replace('/c/en/', '').lower()
    results = CONCEPTNET_DB.get(word, {}).get(rel_key, [])
    return {f"/c/en/{r}" for r in results}


class ActiveMutator:
    def __init__(self, sensitive_attribute):
        self.sensitive_attribute = sensitive_attribute
        self.identify_group = self.create_identify_group()

    def create_identify_group(self):
        if self.sensitive_attribute == 'gender':
            return {"female", "male"}
        if self.sensitive_attribute == 'lgbt':
            return {"gay", "lesbian", "bisexual"}

    def create_active_candidates(self, token: str):
        return {f'{adj} {token}' for adj in self.identify_group}


class AnalogyMutator:
    antonym_cache = dict()
    formOf_cache = dict()
    analogy_mutation = dict()

    def __init__(self, sensitive_attribute, enable_pre_fetch=False, model=None):
        self.sensitive_attribute = sensitive_attribute
        if sensitive_attribute == 'gender':
            self.model = model
            self.dist = lambda x, y: np.sum((x-y)**2)
            self.pre_fetch = {}
            if enable_pre_fetch:
                with open("word_pairs.txt") as wp:
                    for l in wp.readlines():
                        parts = l.strip().split()
                        if len(parts) >= 2:
                            self.pre_fetch[parts[0]] = parts[1]
    
                            self.pre_fetch[parts[1]] = parts[0]

    def create_analogy_candidates(self, token: str, strict_mode = True):
        if self.sensitive_attribute == 'gender':
            if token not in self.model:
                return set()
            #
            if strict_mode:
                kg_candidates = AnalogyMutator.antonym(token)
                if kg_candidates: 
                    #
                    return kg_candidates

            if token in self.pre_fetch:
                return {self.pre_fetch[token]}
            
            # 判断偏向性别
            dist_male = self.dist(self.model[token], self.model["male"])
            dist_female = self.dist(self.model[token], self.model["female"])
            gender = 1 if dist_male < dist_female else 0
            
            if gender == 1:
                candidates1 = {i[0].lower() for i in
                               self.model.most_similar(positive=["woman", token], negative=["man"])
                               if i[1]>.7}
            else:
                candidates1 = {i[0].lower() for i in
                               self.model.most_similar(positive=["man", token], negative=["woman"])
                               if i[1]>.7}
            return candidates1
            
        return set()

    @staticmethod
    def antonym(token: str):
        word = token.lower()
        if word in AnalogyMutator.antonym_cache:
            return AnalogyMutator.antonym_cache[word]
        
        results = query_local_kg(word, 'antonym')
        clean_results = {n.split("/")[3] for n in results if len(n.split("/")) > 3}
        AnalogyMutator.antonym_cache[word] = clean_results
        return clean_results

    @staticmethod
    def formof(token: str):
        word = token.lower()
        if word in AnalogyMutator.formOf_cache:
            return AnalogyMutator.formOf_cache[word]
        
        results = query_local_kg(word, 'formof')
        clean_results = {n.split("/")[3] for n in results if len(n.split("/")) > 3}
        AnalogyMutator.formOf_cache[word] = clean_results
        return clean_results

isA_cache = dict()

def isa(start="", limit=10):
    if start in isA_cache:
        return isA_cache[start]
    
    # local IsA
    results = list(query_local_kg(start, 'isa'))

    processed_results = ["/".join(n.split("/")[:4]) for n in results[:limit]]
    isA_cache[start] = processed_results
    return processed_results

def check_human(token, debug=False):
    if token in insensitive_blacklist or '/c/en/' + token in no_human_indicator:
        return False
    token = lemmatizer.lemmatize(token.lower(), 'n')
    tokens = ['/c/en/' + token]
    visited = set()
    topK2 = [8, 4]
    for i in range(2):
        next_round_token = []
        for w in tokens:
            if w not in visited:
                visited.add(w)

                end_nodes = isa(start=w, limit=30)
                if debug: print(w, end_nodes)
                if len(no_human_indicator & set(end_nodes)) > 0:
                    return False
                if len(human_indicator & set(end_nodes)) > 0:
                    return True
                if len(neutral_indicator & set(end_nodes)) > 0:
                    return False
                next_round_token += end_nodes[:topK2[i]]
        tokens = next_round_token
    return False


def make_mutation(pos, replacement, index):
    s = ""
    for i in range(len(pos)):
        if i == index:
            s += replacement + " "
        else:
            s += pos[i][0] + " "
    return s.strip()

lemmatizer = ns.WordNetLemmatizer()
print("stanford corenlp model loading")
nlp = StanfordCoreNLP('dependency/stanford-corenlp-full-2018-10-05')
print("stanford corenlp model loaded")
part_of_speech = lambda x: nlp.pos_tag(x.strip())

def create_sentence_candidates(sentence: str, ana, act):
    ana_candidates = set()
    act_candidates = set()

    pos = part_of_speech(sentence)
    human_token = []
    for temp, index in zip(pos, range(len(pos))):
        token, tag = temp
        if len(token) == 1 or tag not in {'NN', 'NNS'}:
            continue
        if check_human(token):
            human_token.append((token, tag, index))

    # ==计数===
    has_ana = 0
    has_act = 0
    ana_case_cnt = 0
    act_case_cnt = 0

    for token, tag, index in human_token:
        # 【核心修复】：在每次循环开始时，将当前 Token 的突变体集合清空初始化
        current_act = set()
        current_ana = set()

        # Active Mutator
        if act is not None and (index == 0 or pos[index-1][1] not in {'NN', 'JJ', 'JJR', 'JJS'}):
            current_act |= {make_mutation(pos, replacement, index)
                                for replacement in act.create_active_candidates(token)
                                if replacement is not None}
            if len(current_act) > 0:
                has_act = 1
                act_case_cnt += len(current_act)
            act_candidates |= current_act

        # Analogy Mutator
        word = lemmatizer.lemmatize(token.lower(), 'n') if tag == 'NNS' else token.lower()
        if ana is not None:
            current_ana |= {make_mutation(pos, replacement, index)
                                for replacement in ana.create_analogy_candidates(word)
                                if replacement is not None}
            if len(current_ana) > 0:
                has_ana = 1
                ana_case_cnt += len(current_ana)
            ana_candidates |= current_ana
            
    return ana_candidates, act_candidates, has_ana, has_act, ana_case_cnt, act_case_cnt

if __name__ == "__main__":
    import gensim.downloader as api
    print("word2vec model loading")
    word2vec = api.load("word2vec-google-news-300")
    print("word2vec model loaded")
    ana = AnalogyMutator("gender", model=word2vec)
    pair_list = []
    try:
        with open("word_pairs.txt") as wp:
            for l in wp.readlines():
                l = l.strip().split()
                if len(l) >= 2: pair_list.append((l[0],l[1]))
    except FileNotFoundError:
        pass
        
    for p in pair_list:
        print(p[0], ana.create_analogy_candidates(p[0]))
        print(p[1], ana.create_analogy_candidates(p[1]))
    while True:
        s = input("Enter word for testing (or Ctrl+C to exit): ")
        print(ana.create_analogy_candidates(s))