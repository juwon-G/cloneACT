import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)


import torch
import pickle
from utils.load import load_model_tokenizer
from utils.attack import black_attack
from utils.parser import attack_parser
import time
from datasets import load_from_disk


def main(args):

    PATH = args.path 
    if not os.path.exists(PATH):
        print("Please run the white-box attack first!")
        return
    
    # adv sentences
    with (open(PATH, "rb")) as f:
        d = pickle.load(f)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    # load target black-box NMT model and its tokenizer
    black_model, black_tokenizer = load_model_tokenizer('marian', 'en', 'de', device)

    # dataset tokenized by target tokenizer
    dataset = load_from_disk(args.dataset_name)

    
    
    attack_dict = {}

    time_begin = time.time()
    for idx in range(args.start_index, args.start_index+args.num_samples):
        best_output = black_attack(args, idx, d[idx], dataset, black_tokenizer, black_model, device)
        attack_dict[idx]= best_output 
    
    print(f'finished attack for {args.num_samples} samples in {time.time()-time_begin} seconds!')
    with open(f'{PATH[:-4]}_black_lang.pkl', 'wb') as f:
        pickle.dump(attack_dict, f)

if __name__ == '__main__':
    
    parser = attack_parser()

    args = parser.parse_args()

    main(args)
