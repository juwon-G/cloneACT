import argparse

def attack_parser():
    parser = argparse.ArgumentParser(description="TransFool Attack")

    # Bookkeeping
    parser.add_argument("--result_folder", default="result", type=str,
        help="folder for loading trained models")

    # Data
    parser.add_argument("--dataset_name", default="sst2_en_fr", type=str,
        # choices=["sst2_en_fr"],
        help="classification dataset to use")
    parser.add_argument("--dataset_config_name", default="fr-en", type=str,
        choices=["fr-en", "de-en"],
        help="config of the translation dataset to use")
    parser.add_argument("--num_samples", default=100, type=int,
        help="number of samples to attack")
    parser.add_argument("--start_index", default=0, type=int,
        help="starting sample index")
    parser.add_argument("--part", default="", type=str,
        help="dataset part to attack")
        


    # previous attack results
    parser.add_argument("--path", type=str,
        help="path to the results of the main attack")
    
    return parser