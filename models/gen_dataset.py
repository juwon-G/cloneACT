from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,  DataCollatorForSeq2Seq, MBart50TokenizerFast
from torch.utils.data import DataLoader
import torch 
import argparse 
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main(args):

    dataset_name = args.dataset_name
    model_name = args.model_name
    source_lang = args.source_lang
    target_lang = args.target_lang

    prefix = ''
    max_source_length = 1024
    max_target_length = 128
    padding = False
    batch_size = 32


    dataset_label = {'sst2':'sentence', 'ag_news':'text', 'rotten_tomatoes':'text'}
    val_label = {'sst2':'validation', 'ag_news':'test', 'rotten_tomatoes':'test'}

    # load mode, tokenizer, dataset
    if dataset_name == "sst2":
        dataset = load_dataset("glue",dataset_name)
    else:
        dataset = load_dataset(dataset_name)

    
    if model_name=='marian':
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    elif model_name=='mbart':
        model_name = 'facebook/mbart-large-50-one-to-many-mmt'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        dict_lang = {"en":"en_XX", "de":"de_DE", "fr":"fr_XX"}
        tokenizer.src_lang = dict_lang[source_lang]
        tokenizer.tgt_lang = dict_lang[target_lang]
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[tokenizer.tgt_lang ]
    
    tokenizer.model_max_length = 512
    model.eval()



    # preprocess function
    def preprocess_function(examples):
            inputs = [ex for ex in examples[dataset_label[dataset_name]]]
            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

            return model_inputs



    # tokenize dataset
    processed_datasets = dataset.map(
                preprocess_function,
                batched=True,)
        
    train_dataset = processed_datasets["train"]
    val_dataset = processed_datasets[val_label[dataset_name]]


    
    # French dataset
    train_dict = {'sentence':[],'label':[]}
    val_dict = {'sentence':[],'label':[]}

    # train
    for i in tqdm(range(len(train_dataset))):
        with torch.no_grad():
            generated_tokens = model.generate(torch.tensor(train_dataset[i]["input_ids"]).unsqueeze(0).to(device))
            label = [dataset['train'][i]["label"]]
            generated_tokens = generated_tokens.cpu().numpy()[0]
            decoded_preds = [tokenizer.decode(generated_tokens, skip_special_tokens=True)]
            train_dict['sentence'].extend(decoded_preds)
            train_dict['label'].extend(label)   
    dataset_fr_train = Dataset.from_dict(train_dict)

    # validation
    for i in tqdm(range(len(val_dataset))):
        with torch.no_grad():
            generated_tokens = model.generate(torch.tensor(val_dataset[i]["input_ids"]).unsqueeze(0).to(device))
            label = [dataset[val_label[dataset_name]][i]["label"]]
            generated_tokens = generated_tokens.cpu().numpy()[0]
            decoded_preds = [tokenizer.decode(generated_tokens, skip_special_tokens=True)]
            val_dict['sentence'].extend(decoded_preds)
            val_dict['label'].extend(label)
    dataset_fr_val = Dataset.from_dict(val_dict) 


    dataset_fr = DatasetDict({'train':dataset_fr_train,'validation':dataset_fr_val})
    dataset_fr.save_to_disk(f'{args.model_name}/{dataset_name}_{target_lang}')



    dataset_fr_train = dataset_fr_train.rename_column('sentence',f'sentence_{target_lang}')
    dataset_fr_train = dataset_fr_train.add_column(f'sentence_{source_lang}' , dataset['train'][dataset_label[dataset_name]])

    dataset_fr_val = dataset_fr_val.rename_column('sentence',f'sentence_{target_lang}')
    dataset_fr_val = dataset_fr_val.add_column(f'sentence_{source_lang}' , dataset[val_label[dataset_name]][dataset_label[dataset_name]])

    dataset_fr = DatasetDict({'train':dataset_fr_train,'validation':dataset_fr_val})
    dataset_fr.save_to_disk(f'{args.model_name}/{dataset_name}_{source_lang}_{target_lang}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation.")


    # Model
    parser.add_argument("--model_name", default="marian", type=str,
        help="target NMT model")
    parser.add_argument("--source_lang", default="en", type=str,
        choices=["en"],
        help="source language")
    parser.add_argument("--target_lang", default="fr", type=str,
        choices=["fr", "de"],
         help="target language")
    parser.add_argument("--dataset_name", default="sst2", type=str,
        choices=["sst2", "ag_news", "rotten_tomatoes"],
         help="target dataset")
    
    args = parser.parse_args()

    main(args)