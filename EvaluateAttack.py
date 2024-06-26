import pickle
import torch
import argparse    
from utils.eval import Eval 
from utils.convert import convert, convert_baseline
from utils.convert_target import convert_target

from tabulate import tabulate
import pandas as pd

import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):

    
   
    PATH = args.path

    if "textattack" in args.attack_alg and ".pkl" in PATH:
        with (open(PATH, "rb")) as f:
            d = (pickle.load(f))
    
    elif "textattack" in args.attack_alg:
        d = pd.read_csv(PATH)  

        if 'target' in PATH: # for targeted attacks
            d = convert_target(d,args,device,args.num_class)
        else:
            d = convert(d,args,device,args.num_class)
        
    else:
        with (open(PATH, "rb")) as f:
            d = (pickle.load(f))
        d = convert_baseline(d,args,device)
        
    
    with open(f'{PATH[:-4]}.pkl', 'wb') as f:
        pickle.dump(d, f)

    
    E = Eval(d, device, args, part = 'success')
   
    results = E.results_success
   
    



    print("\n\n")
    print("*********** RESULTS In Successful Attacks ***********")
    print("\n")               

    print("\n")

    print("Classifier 2 (sim>0.7):")
    print(tabulate(results, tablefmt='psql', showindex=False, numalign="left", floatfmt=".8f"))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation.")

    # Bookkeeping
    parser.add_argument("--path", type=str,
        help="folder for loading trained models")

    # Model
    parser.add_argument("--target_model_name", default="marian", type=str,
        choices=["marian", "mbart", "google"],
        help="target NMT model")
    parser.add_argument("--source_lang", default="en", type=str,
        choices=["en", "fr"],
        help="source language")
    parser.add_argument("--target_lang", default="fr", type=str,
        choices=["fr", "de", "zh", "en", "de2", "ru", "cs"],
        help="target language")
    
    parser.add_argument("--dataset_name", default="sst2", type=str,
        help="dataset name")

    
    # Eval setup
    parser.add_argument("--bad_perp", default=100, type=float,
        help="threshold for the bad perplexity")
    parser.add_argument("--bad_sim", default=0, type=float,
        help="threshold for the bad similaroty")
    parser.add_argument("--num_class", default=1, type=float,
        help="number of classifier used in the attack")

    # attack method
    parser.add_argument("--attack_alg", default="textattack", type=str,
        help="attack method")
    
    args = parser.parse_args()

    main(args)