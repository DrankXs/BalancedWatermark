
import os
import sys

import argparse
from transformers import AutoTokenizer, LlamaTokenizer
import torch
from datetime import datetime
from bw.get_frequency import transfer_frequency_files

def for_frequency(args):
    if 'llama' in args.model_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, torch_dtype=torch.float16)
    transfer_frequency_files(tokenizer=tokenizer, json_file=args.data_file, target_key=args.target_key, history_record_file=args.history_record_file, fre_record_file=args.fre_record_file, fre_pair_file=args.fre_pair_file)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="prepare watermark file")
    # task
    parser.add_argument("--task", default="fre", type=str, help="fre:cal frequency and save;")
    
    # model
    parser.add_argument("--model_path", default="opt-2.7b", type=str, help="generate model")
    
    # file 
    parser.add_argument("--data_file", default=None, type=str, help="data file for frequency statistics")
    parser.add_argument("--target_key", default="text", type=str, help="frequency pair file for seed map")
    parser.add_argument("--history_record_file", default=None, type=str, help="history record file for frequency statistics")
    parser.add_argument("--fre_record_file", default=None, type=str, help="record detail for frequency statistics")
    parser.add_argument("--fre_pair_file", default=None, type=str, help="frequency pair file for seed map")
    
    
    args = parser.parse_args()
    
    if args.task == "fre":
        for_frequency(args)
    
    