
import os
dir_path = os.path.dirname(os.path.realpath(__file__))      # this path
# semantic -> evaluate -> wmfunc
parent_dir_path = os.path.abspath(os.path.join(dir_path,    
                                               os.pardir,   
                                               os.pardir))  
import sys
sys.path.append(parent_dir_path) # add path in system


import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
# us
from load_model import load_clm_model
from utils import *

'''
    this use lm model in premodel

    DemoModel:
        gpt2: https://huggingface.co/gpt2 
            https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        opt-1.3b:https://huggingface.co/facebook/opt-1.3b
            https://arxiv.org/abs/2205.01068
'''
def compute_perplexity(
    texts, 
    model="", 
    model_name = None,
    to_load_model=True, 
    batch_size:int=1, 
    add_start_token: bool=True, 
    max_length=None):
    """compute perplexity for the text

    Args:
        text (str or a list of strs): input text
        model (optional): model_path or a AutoCausalLMModelForWM if load_model=False
        model_name (str, optional): model name. Defaults to None.
        load_model (bool, optional): _description_. Defaults to True.
        batch_size (int, optional): size of a batch to processing batch
        add_start_token (bool, optional): to compute perplexity, need when generate, 
            We follow the perplexity calculation in huggingface to use it
        max_length (_type_, optional): same as add_start_token

    Returns:
        _type_: a json dict
            "perplexities" (list float): each sentences perpelexities
            "mean_perplexity" (float): mean perplexity

    """  
    # prepare language model
    if to_load_model:
        model = load_clm_model(model, model_name=model_name)
    
    # format input to a list
    if isinstance(texts, str):
        texts = [texts]
        count = 1
    elif isinstance(texts, list):
        count = len(texts)
    else:
        print("ppl's input text must be a str or a list of strs")
    
    model.model.eval()
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    
    last_step = 0
    with tqdm(total=100) as pbar:
        for start_index in range(0, len(texts), batch_size):
            end_index = min(start_index + batch_size, len(texts))
            
            # get logits, labels and attn_mask
            
            out_logits, labels, attn_mask = model.get_text_logits(texts[start_index:end_index], add_start_token=add_start_token,max_length=max_length)
            
            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            # compute batch perplexity
            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )            
            perplexity_batch = perplexity_batch.detach().cpu().tolist()
            
            ppls.extend(perplexity_batch)

            # update progressbar
            update = (end_index-last_step)*100 / (len(texts))
            if update > 1:
                pbar.update(int(update))
                last_step = end_index 
    
            
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
        

def run_demo():
    
    text = ["today is a good day.", "apple is delicious"]
    out = compute_perplexity(text)
    print(out)
    
# run_demo()