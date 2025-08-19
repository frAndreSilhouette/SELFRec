from SELFRec import SELFRec
from util.conf import ModelConf
import time
import re
import os
import random
import numpy as np
import torch

def fix_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_models(title, models):
    print(f"{'=' * 80}\n{title}\n{'-' * 80}")
    for category, model_list in models.items():
        print(f"{category}:\n   {'   '.join(model_list)}\n{'-' * 80}")

if __name__ == '__main__':

    fix_seeds(42) 
    models = {
        'Graph-Based Baseline Models': ['LightGCN', 'DirectAU', 'MF', 'UserKNN', 'ItemKNN'],
        'Self-Supervised Graph-Based Models': ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL', 'MixGCF'],
        'Sequential Baseline Models': ['SASRec'],
        'Self-Supervised Sequential Models': ['CL4SRec', 'BERT4Rec']
    }

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print_models("Available Models", models)

    # model = input('Please enter the model you want to run:')
    modified_models = ['LightGCN', 'MF', 'MixGCF', 'NCL', 'SGL', 'SimGCL', 'XSimGCL']
    # campus_ids = [10, 15, 34, 102, 143]
    campus_ids = [10]

    for campus_id in campus_ids:
        for model in modified_models:
            print('=' * 80)
            print(f"Running {model} on campus {campus_id}...")
            
            yaml_path = f'./conf/{model}.yaml'
            # 直接修改 yaml 文件内容
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_str = re.sub(r"campus_(\*|\d+)", f"campus_{campus_id}", f.read())
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)   

            s = time.time()
            all_models = sum(models.values(), [])
            if model in all_models:
                conf = ModelConf(yaml_path)
                rec = SELFRec(conf)
                rec.execute()
                e = time.time()
                print(f"Running time: {e - s:.2f} s")
            else:
                print('Wrong model name!')
                exit(-1)
