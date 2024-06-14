import torch
import os
import logging
import random
import json

def setGPU(id = '2'):
    ## set gpu, return the device we can choose
    device = torch.device('cuda:'+id if torch.cuda.is_available() else 'cpu')
    return device


def checkGPU(check_tensor:torch.Tensor, desc="Check"):
    if check_tensor.is_cuda:
        print(desc, "✔")
    else:
        print(desc, "✖")
    
   

def remove_redundancy_json_keys(input_file:str, output_file:str, remove_keys: list):
    with open(output_file, "w", encoding="utf-8") as f:
        for id, x in enumerate(open(input_file, encoding="utf-8")):
            example = json.loads(x)
            new_example = {}
            for key in example.keys():
                if key not in remove_keys:
                    new_example[key] = example[key]
            f.write(json.dumps(example)+"\n")
    


def average_device_map(src_map, devices=[0,1,2,3]):
    n_gpu = len(devices)
    per_gpu_blocks = int(len(src_map) // n_gpu)
    new_device_map = {}
    for i, key in enumerate(src_map.keys()):
        gpu_id = int(i / per_gpu_blocks)
        gpu_id = gpu_id if gpu_id < n_gpu else n_gpu - 1
        new_device_map[key] = devices[gpu_id]

    device_map = new_device_map
    return device_map

def str_to_intlist(input:str, split=","):
    input_strs = input.split(split)
    res = [int(k) for k in input_strs]
    return res