
import os
import types
dir_path = os.path.dirname(os.path.realpath(__file__))      # this path
# rglist -> watermark -> wmfunc
parent_dir_path = os.path.abspath(os.path.join(dir_path,    
                                               os.pardir,   
                                               os.pardir))  
import sys
sys.path.append(parent_dir_path) # add path in system


import hashlib
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import torch
import json
import numpy as np
import pickle
import random


def transfer_frequency_files(tokenizer: AutoTokenizer, json_file: str = None, target_key: str = "text", history_record_file: str = None, fre_record_file: str = None, fre_pair_file: str = None):
    
    # load history
    if history_record_file is not None:
        with open(history_record_file, "r") as f:
            fre_dict = json.load(f)
    else:
        fre_dict = {}
    
    if json_file is not None:
        for x in open(json_file, encoding="utf-8"):
            example = json.loads(x)
            text = example[target_key]
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            for id in ids:
                if id in fre_dict.keys():
                    fre_dict[id] += 1
                else:
                    fre_dict[id] = 1
    
    fre_dict = dict(sorted(fre_dict.items(), key = lambda item: item[1], reverse=True))
    
    if fre_record_file is not None:
        with open(fre_record_file, "w") as f:
            json.dump(fre_dict, f)
    
    if fre_pair_file is not None:
        key_len = len(fre_dict.keys())
        key_list = list(fre_dict.keys())
        shuffle_ids = list(range(key_len))
        random.shuffle(shuffle_ids)
        result_dict = {}
        
        for id, key in enumerate(key_list):
            sid = shuffle_ids[int(id/2)]
            result_dict[key] = 2*shuffle_ids[sid] if id%2==0 else 2*shuffle_ids[sid] + 1
        
        with open(fre_pair_file, "w") as f:
            json.dump(result_dict, f)


def run_demo():
    fre_record = "prewm/fre_record.jsonl"
    data_file = "c4-realnewslike-5000.jsonl"
    fre_pairs_file = "prewm/opt_pairs.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("opt-2.7b")
    
    transfer_frequency_files(tokenizer=tokenizer, json_file=data_file, fre_record_file=fre_record, fre_pair_file=fre_pairs_file)
    print("over")

# run_demo()
