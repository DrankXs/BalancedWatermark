from bw.integration import BalancedWaterMark, BalancedWDetector
from load_model import load_clm_model
from utils import setGPU
model = load_clm_model(model_path="/home/zhangshuhao/MLFiles/PLM/opt-2.7b")
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
print(detect_res)