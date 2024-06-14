'''
this file is to load torch model
'''

import os
dir_path = os.path.dirname(os.path.realpath(__file__))      # this path
# model -> wmfunc
parent_dir_path = os.path.abspath(os.path.join(dir_path,    
                                               os.pardir))  
import sys
sys.path.append(parent_dir_path) # add path in system

from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, LlamaForCausalLM, AutoModel
import torch
import torch.nn as nn
from tqdm import tqdm
import re

# us
# from constant import *
from utils import *
class BaseGenerateModel(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.tokenizer = None
        
    def generate(self, 
                    texts, 
                    batch_size: int = 1,
                    max_length: int = 256,
                    **gen_kwargs):
        """ generate some sequence by this model for the texts

        Args:
            texts (str/str list): input texts
            batch_size (int, optional): size of a batch to accelerate. Defaults to 1.
            max_length (int, optional): generate sequence max length. Defaults to 128.
            gen_kwargs: generate kwargs
        Returns:
            out_sentences (a 2d list): [[generated sentences for the first input text],...]
        """        
        # reconstruct the texts
        if isinstance(texts, str):
            texts = [texts]     # make single str to a list
        
        # init states
        self.model.eval()
        out_sentences = []
        
        # calculate start
        for start_index in range(0, len(texts), batch_size):
            end_index = min(start_index + batch_size, len(texts))

            # tokenize and convert token to ids
            input_ids = self.tokenizer(
                texts[start_index:end_index],
                return_tensors="pt", padding="longest",
                max_length=max_length,
                truncation=True,
            )
            input_ids = input_ids.input_ids
            input_ids = input_ids.to(self.model.device)
            
            # generate by seq2seqModel with parameters
            outputs = self.model.generate(
                input_ids,
                **gen_kwargs
            )
                        
            res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                                                
            # conclude the result
            beams = int(len(res) / (end_index - start_index))
            for i in range(end_index - start_index):
                out_sentences.append(res[i * beams:(i + 1) * beams])
            
                
        return out_sentences

    def get_text_logits(self, text, add_start_token: bool=True, max_length=None):
        """_summary_

        Args:
            text (a str or a list of str): input
            add_start_token (bool, optional):  Defaults to True.
            max_length (int, optional):  Defaults to None.
        Returns:
            out_logits: probablity for the token (str_counts, max_str_length, vocab_size)
            labels: text tokenized by the tokenizer, tokenid (str_counts, max_str_length)
            attn_mask: zero-one matrix, to mask the padding token
        """
        encoded_batch, attn_mask = self.prepare_input(text=text, 
                                                      add_start_token=add_start_token,
                                                      max_length=max_length)
        # get pure text input ids(labels) and get logits by this language model(AutoModelForCausalLM)
        labels = encoded_batch
        with torch.no_grad():
            try:
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits
            except Exception as e:
                print(f"{e} \n {text}")
        return out_logits, labels, attn_mask
    
    def prepare_input(self,  text, add_start_token: bool=False, max_length=None):
        # format input to a list
        if isinstance(text, str):
            text = [text]
        
        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if self.tokenizer.pad_token is None:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})
        
        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        # tokenize and convert token to ids
        encodings = self.tokenizer(
            text,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.model.device)

        # get token ids and attention mask
        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        encoded_batch = encoded_texts
        attn_mask = attn_masks

        # add start token before all string and fix the attention mask
        if add_start_token:
            bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(self.model.device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.model.device), attn_mask], dim=1
            )
       
        return encoded_batch, attn_mask 
        
'''
    CLM MODEL:
        which can be used to compute perplexity and generate;
        we use AutoModelForCausalLM to express LM MODEL;
        this model will be use for text generation;
        this model focus on causal relation ship, this means this model we can seem it as
        a Autoregressive language model;
    Task:
        evaluate.semantic.perplexity:✔
        generate:✖    
''' 
class AutoCausalLMModelForWM(BaseGenerateModel):
    def __init__(self,model_path: str = None, model_name: str=None, single_device = None, devices = None) -> None:
        """ to conclude the AutoModelForCausalLM
        Args:
            model_path (str, optional): the model path for the function from pretrained. Defaults to None.
            model_name (str, optional): the model name. Defaults to None.
        """
        self.model_type = "perplexity,generate"
        self.model_name = model_name
        self.model_path = model_path
        # init model and tokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = single_device
        # singledevice
        if devices is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True)
            # set gpu
            self.device = single_device
            self.model.to(self.device)
        else:
            if "llama" in model_path.lower():
                # llama3-8b
                device_map = self.get_device_map_for_init(model_path, devices)
                
                self.model = LlamaForCausalLM.from_pretrained(model_path, output_hidden_states=True, device_map=device_map)
            else:
                # opt must be single device
                self.model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True)
                self.model.to(self.device)
                
    def get_device_map_for_init(self, model_path, devices):
        if "llama-2-7b" in model_path.lower():
            src_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 2, 'model.layers.17': 2, 'model.layers.18': 2, 'model.layers.19': 2, 'model.layers.20': 2, 'model.layers.21': 2, 'model.layers.22': 2, 'model.layers.23': 2, 'model.layers.24': 2, 'model.layers.25': 3, 'model.layers.26': 3, 'model.layers.27': 3, 'model.layers.28': 3, 'model.layers.29': 3, 'model.layers.30': 3, 'model.layers.31': 3, 'model.norm': 3, 'lm_head': 3}
            device_map = average_device_map(src_map=src_map, devices=devices)
            return device_map
        else:
            return "auto"

    
def load_clm_model(model_path=None,model_name=None, single_device = setGPU(), devices = None):
    """_summary_

    Args:
        model_path (str, optional): the model path for the function from pretrained. Defaults to None.
        model_name (str, optional): the model name. Defaults to None.
    Returns:
        model: a AutoCausalLMModelForWM
    """    
    if model_path == None and model_name == None:
        print("model_path and model_name have to at least one to be not None")
        return None
    elif model_name==None:
        model_name = "lm"
    elif model_path == None:
        model_path = model_name

    # read model 
    model = AutoCausalLMModelForWM(model_name=model_name, model_path=model_path, single_device=single_device, devices=devices)
    
    return model
    

# TODEL
def test_lm_model():
    m = load_clm_model(model_path="./models/gpt2")
    text = ["today is a good day", "apple is delicious"]
    logits,_,_ = m.get_text_logits(text)
    print(logits.shape)
