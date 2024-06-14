
'''
    paper:A Watermark for Large Language Models
    github:https://github.com/jwkirchenbauer/lm-watermarking

    Necessory:
        model: generate sentence;
        prompt: input for generate, for example, quesiton-answer's question;   
'''

import os
dir_path = os.path.dirname(os.path.realpath(__file__))      # this path
# bw 
parent_dir_path = os.path.abspath(os.path.join(dir_path))  
import sys
sys.path.append(parent_dir_path) # add path in system

from functools import partial
from transformers import LogitsProcessorList
from typing import List

from load_model import *
from bw.watermark_processor import *
from evaluate.statistics import *
class BalancedWaterMark:
    def __init__(self, 
                 generate_model = None,
                 to_load_model: bool = False,
                 model_name: str = None,
                 model_type: str = "clm",
                 **wm_kwargs,
                 ) -> None:
        """init balanced watermark

        Args:
            generate_model (_type_, optional): a language model, must be AutoCausalLMModelForWM. Defaults to None.
            to_load_model (bool, optional): True to load model, False to use the exist model. Defaults to False.
            model_name (str, optional): pretrained model file path. Defaults to None.
            model_type (str, optional): must be clm. Defaults to "clm".

        Raises:
            TypeError: _description_
        """        
        # prepare generate model and tokenizer
        if generate_model == None:
            # default generate model is gpt2
            self.gen_model = load_clm_model("gpt2", "gpt2")
        else:
            # load causal language model or Seq2Seq language model by path, model name may be used to focus different model type
            if isinstance(generate_model, AutoCausalLMModelForWM) and not to_load_model:
                self.gen_model = generate_model
            elif to_load_model and model_type == "clm":
                self.gen_model = load_clm_model(generate_model, model_name)
            else:
                raise TypeError("you must load model by our AutoCausalLMModelForWM or AutoLMModelForWM")
        
        # init watermark logits processor
        self.watermark_processor = WatermarkLogitsProcessor(vocab=list(self.gen_model.tokenizer.get_vocab().values()),
                                                    **wm_kwargs
                                                    )
        

    def generate(self,
                 text,
                 max_new_tokens: int = 200,
                 prompt_max_length: int = None,
                 without_watermark: bool = False,
                 **gen_kwargs):
        """ generate watermark texts

        Args:
            text (_type_): input prompt, str or a text list.
            max_new_tokens (int, optional): max new token when generate. Defaults to 200.
            prompt_max_length (int, optional): max input length. Defaults to None.
            without_watermark (bool, optional): True to generate without watermark. Defaults to False.

        Returns: a dict include result detail. wmtext is the key for watermark text.
        """       
        # format input to a list
        if isinstance(text, str):
            text = [text]
        
        gen_kwargs.update(dict(max_new_tokens=max_new_tokens))
        
        
        # fix text_max_length by this generate model
        if prompt_max_length:
            pass
        elif hasattr(self.gen_model.model.config, "max_position_embedding"):
            prompt_max_length = self.gen_model.model.config.max_position_embeddings - max_new_tokens
        else:
            prompt_max_length = 2048 - max_new_tokens
        
        # get token ids by tokenizer and get final input tokens after using the prompt max length and max new tokens
        tokd_input = self.gen_model.tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=prompt_max_length).to(self.gen_model.device)
        truncation_warning = True if tokd_input["input_ids"].shape[-1] == prompt_max_length else False
        redecoded_input = self.gen_model.tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)

        # dicide whether to generate unwatermarked text
        if without_watermark:
            generate_without_watermark = partial(
                self.gen_model.model.generate,
                **gen_kwargs
            )
            output_without_watermark = generate_without_watermark(**tokd_input)
        else:
            output_without_watermark = None
        
        generate_with_watermark = partial(
            self.gen_model.model.generate,
            logits_processor=LogitsProcessorList([self.watermark_processor]), 
            **gen_kwargs
        )
        
        output_with_watermark = generate_with_watermark(**tokd_input)
        # if generate model is a decode only model
        if isinstance(self.gen_model, AutoCausalLMModelForWM):
            output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:] if output_without_watermark is not None else None
            output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]
        
        # get real word by output
        decoded_output_without_watermark = self.gen_model.tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True) if output_without_watermark is not None else None
        decoded_output_with_watermark =  self.gen_model.tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)
        
        return {
            "redecoded_input":redecoded_input,
            "truncation": int(truncation_warning),
            "text": decoded_output_without_watermark,
            "wmtext": decoded_output_with_watermark,
            "gen_kwargs": gen_kwargs
        }
        

class BalancedWDetector(WatermarkBase):
    def __init__(self,
                    tokenizer: Tokenizer = None,
                    device: torch.device = None,
                    z_threshold: float = 4.0,
                    normalizers: List[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
                    ignore_repeated_bigrams: bool = False,
                    **wm_kwargs,
                    
                    ) -> None:
        """ init detector

        Args:
            tokenizer (Tokenizer, optional): model tokenizer, to transfer input to ids. Defaults to None.
            device (torch.device, optional): use torch.device, can use gpu to acclerate. Defaults to None.
            z_threshold (float, optional): a threshold to determine the text is watermarked or not. Defaults to 4.0.
            normalizers (List[str], optional):  Defaults to ["unicode"].

        Raises:
            NotImplementedError: _description_
        
        """        
        super().__init__(
                        list(tokenizer.get_vocab().values()),
                        **wm_kwargs
                         )
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"
        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))
        
        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams: 
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."

    def detect(self, 
                text: str = None,
                tokenized_text: List[int] = None,
                return_num_tokens_scored: bool = True,
                return_num_green_tokens: bool = True,
                return_green_fraction: bool = True,
                return_green_token_mask: bool = False,
                return_z_score: bool = True,
                return_p_value: bool = True):
        """ detect text or tokenized text

        Args:
            text (str, optional): text, a str. Defaults to None.
            tokenized_text (List[int], optional): tokenize text. Defaults to None.
        Raises:
            ValueError: _description_

        Returns:
            a dict for all result detail; z_score is used with z_threshold to determine the text is watermarked or not.
        """    
        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"

        # run optional normalizers on text
        for normalizer in self.normalizers:
            text = normalizer(text)

        # get tokenized text's token ids
        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            # try to remove the bos_tok at beginning if it's there
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        input_ids = tokenized_text
        # compute with detect strategy
        if self.ignore_repeated_bigrams:
            # Method that only counts a green/red hit once per unique bigram.
            # New num total tokens scored (T) becomes the number unique bigrams.
            # We iterate over all unqiue token bigrams in the input, computing the greenlist
            # induced by the first token in each, and then checking whether the second
            # token falls in that greenlist.
            assert return_green_token_mask == False, "Can't return the green/red mask when ignoring repeats."
            bigram_table = {}
            # construct unigrams list
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())
            # decide green or red in order
            for idx, bigram in enumerate(freq.keys()):
                prefix = torch.tensor([bigram[0]], device=self.device) # expects a 1-d prefix tensor on the randperm device
                
                greenlist_ids = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = True if bigram[1] in greenlist_ids else False
            # get green counts
            green_token_count = sum(bigram_table.values())
        else:
            # we don't know which tokens are prompt, so we think at least 1 token is prompt
            num_tokens_scored = len(input_ids) - self.min_prefix_len
            if num_tokens_scored < 1:
                raise ValueError((f"Must have at least {1} token to score after "
                                f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."))
            # Standard method.
            # Since we generally need at least 1 token (for the simplest scheme)
            # we start the iteration over the token sequence with a minimum 
            # num tokens as the first prefix for the seeding scheme,
            # and at each step, compute the greenlist induced by the
            # current prefix and check if the current token falls in the greenlist.
            green_token_count, green_token_mask = 0, []
            # we construct green token num, and get known where is green token by constructing green_token_mask
            for idx in range(self.min_prefix_len, len(input_ids)):
                curr_token = input_ids[idx]
                
                greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)
        
        # we conclude some detect params
        result = {}
        if return_num_tokens_scored:
            result.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            result.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            result.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            result.update(dict(z_score=cal_z_score(green_token_count, num_tokens_scored, expected=self.gamma)))
        if return_p_value:
            z_score = result.get("z_score")
            if z_score is None:
                z_score = cal_z_score(green_token_count, num_tokens_scored,  expected=self.gamma)
            result.update(dict(p_value=cal_p_value(z_score)))
        if return_green_token_mask:
            result.update(dict(green_token_mask=green_token_mask))
            t_map = []
            tokens = input_ids.cpu().tolist()
            for i, flag in enumerate(green_token_mask):
                t_map.append({self.tokenizer._convert_id_to_token(tokens[i]):flag})
            result.update(dict(tokens=t_map))
            
        return result
         
        
        
def run_demo():
    model = load_clm_model(model_path="opt-2.7b")
    wm = BalancedWaterMark(model)
    input_text = (
        "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
        "species of turtle native to the brackish coastal tidal marshes of the "
        )
    input_text = [input_text, input_text]
    
    res = wm.generate(input_text)
    print(res)
    
    pass

# run_demo()