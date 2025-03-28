# Balanced Watermark

## Introduction
Paper [CEFW: A Comprehensive Evaluation Framework for Watermark in Large Language Models](https://arxiv.org/abs/2503.20802) has two ideas: Comprehensive Evaluation Framework for Watermark (CEFW), and Balanced Watermark (BW).

This repository (on GitHub) only supports the implementation of BW. 
The implementation of the CEFW evaluation framework allows for the free combination of corresponding evaluation methods. 


## Environment

```python
python==3.8.0
torch==1.13.0+cu117
transformers==4.35.0
scipy==1.10.1
nltk==3.8.1
tokenizers==0.14.1
homoglyphs==2.0.4
scikit-learn==1.3.2
```

## Balanced Watermark
A watermark and its corresponding detection demonstration are presented.
**gamma** represents the ratio of the green list; 
**delta** signifies the watermark strength;
**model_path** denotes the location of the text generation model. 
When both **use_balanced** and **use_fix** are set to true, the watermark is designated as *BalancedWatermark*; when both are false, the watermark is *KGW*; and when **use_fix** is true while **use_balanced** is false, the watermark is *UNIW*. 
**balanced_pair_file** is a pre-prepared word frequency file utilized for balancing Lists A and B. 
**hash_key** serves as a cryptographic key, utilized to ensure the uniqueness of the watermark.

The demo code is:
```python
from bw.integration import BalancedWaterMark, BalancedWDetector
from load_model import load_clm_model
from utils import setGPU
model = load_clm_model(model_path="opt-2.7b")
wm = BalancedWaterMark( generate_model=model, 
                        gamma=0.5,
                        delta=2.0,
                        use_balanced=True,
                        use_fix=True,
                        balanced_pair_file="prewm/opt_pairs.jsonl",
                        hash_key=15485863,
                        seeding_scheme="simple_1"
                    )
input_text = (
    "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
    "species of turtle native to the brackish coastal tidal marshes of the "
    )
input_text = [input_text]
res = wm.generate(input_text)

detector = BalancedWDetector( tokenizer=model.tokenizer,
                        device=setGPU('2'),
                        gamma=0.5,
                        use_balanced=True,
                        use_fix=True,
                        balanced_pair_file="prewm/opt_pairs.jsonl",
                        hash_key=15485863,
                        seeding_scheme="simple_1")

detect_res = detector.detect(res["wmtext"][0])
```
For accurate detection, it is imperative to ensure the consistency of several parameters:
**gamma**, **hash_key**, **balanced_pair_file**, **use_balanced**, **use_fix**.
It requires special attention that the **tokenizer** and the **balanced_pair_file** must be compatible; otherwise, the configuration of the **balanced_pair_file** would be rendered meaningless.

You can prepare a pair file **opt_pairs_demo.jsonl** by data **xxx.jsonl** for the model **opt-2.7b**.
```bash
python prepare_wm_sc.py --task fre --model_path opt-2.7b --data_file xxx.jsonl --target_key text_key --fre_pair_file prewm/opt_pairs_demo.jsonl
```