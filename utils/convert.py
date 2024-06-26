from utils import attack_output
from datasets import load_metric
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, pipeline

path1 = "./models/results_org/"
path2 = "./results_org_tf/"



def classifier(d, name, device, args, tf=False):
        
    if tf:
        model = TFAutoModelForSequenceClassification.from_pretrained(name)
        if args.target_lang=='fr':
            tokenizer_name = "tblard/tf-allocine"
        elif args.target_lang=='de':
            tokenizer_name = "oliverguhr/german-sentiment-bert"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,truncation=True)
        tensor_type = 'tf'
    else:
        model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(name,truncation=True)
        tensor_type = 'pt'
    
    

    class_adv = []
    class_org = []

    for i in range(len(d)):
        adv_tr = d[i].adv_tr
        org_tr = d[i].org_tr

        adv_tr_t = tokenizer(adv_tr, return_tensors=tensor_type)['input_ids']
        org_tr_t = tokenizer(org_tr, return_tensors=tensor_type)['input_ids']

        if adv_tr_t.numpy().shape[1]>512:
            adv_tr_t = adv_tr_t[:,:512]
        
        if org_tr_t.numpy().shape[1]>512:
            org_tr_t = org_tr_t[:,:512]

        if tensor_type=='pt':
            class_adv.append(model(adv_tr_t.to(device))['logits'].cpu().detach().numpy().argmax())
            class_org.append(model(org_tr_t.to(device))['logits'].cpu().detach().numpy().argmax())
        
        elif tensor_type=='tf':
            class_adv.append(model(adv_tr_t)['logits'].numpy().argmax())
            class_org.append(model(org_tr_t)['logits'].numpy().argmax())

        


    ground_truth = [d[i].ground_truth for i in range(len(d))]

    attack_results = []
    for i in range(len(d)):
        # if class_org[i]['label']!=label[ground_truth[i]]:
        if class_org[i]!=ground_truth[i]:
            attack_results.append('skipped')
        
        # elif class_adv[i]['label']!=label[ground_truth[i]]:
        elif class_adv[i]!=ground_truth[i]:
            attack_results.append('success')
        
        else:
            attack_results.append('failed')

    return attack_results

    



def convert(d,args,device,num_class=1):

    attack_dict = {}

    for i in tqdm(range(len(d))):
        ex =d.loc[i]
        bleu = load_metric("sacrebleu")
        chrf = load_metric("chrf")

        if num_class==1:
            adv_output = ex.perturbed_output.split(", tensor([")
            adv_tr = adv_output[0][2:-1]
            adv_logits = [float(k) for k in adv_output[1].split("], device=")[0].split(",")]

            org_output = ex.original_output.split(", tensor([")
            org_tr = org_output[0][2:-1]
            org_logits = [float(k) for k in org_output[1].split("], device=")[0].split(",")]

            adv_bleu = bleu.compute(predictions=[adv_tr], references=[[org_tr]])['score']
            adv_chrf = chrf.compute(predictions=[adv_tr], references=[[org_tr]])['score']

            if np.argmax(org_logits)!=ex.ground_truth_output:
                attack_result = 'skipped'
            
            elif np.argmax(adv_logits)!=ex.ground_truth_output:
                attack_result = 'success'
            
            else:
                attack_result = 'failed'
        
        elif num_class==2:

            adv_output = ex.perturbed_output.split(", tensor([")
            adv_tr = adv_output[0][2:-1]
            adv_logits = [float(k) for k in adv_output[1].split("], device=")[0].split(",")]
            adv_logits1 = [float(k) for k in adv_output[2].split("], device=")[0].split(",")]

            org_output = ex.original_output.split(", tensor([")
            org_tr = org_output[0][2:-1]
            org_logits = [float(k) for k in org_output[1].split("], device=")[0].split(",")]
            org_logits1 = [float(k) for k in org_output[2].split("], device=")[0].split(",")]

            adv_bleu = bleu.compute(predictions=[adv_tr], references=[[org_tr]])['score']
            adv_chrf = chrf.compute(predictions=[adv_tr], references=[[org_tr]])['score']

            attack=False
            attack1=False
            
            if np.argmax(org_logits)!=ex.ground_truth_output and np.argmax(org_logits1)!=ex.ground_truth_output:
                attack_result = 'skipped'
            elif np.argmax(org_logits)!=ex.ground_truth_output:
                attack1=True
            elif np.argmax(org_logits1)!=ex.ground_truth_output:
                attack=True
            else:
                attack=True
                attack1=True



            if attack and attack1: 
                if np.argmax(adv_logits)!=ex.ground_truth_output and np.argmax(adv_logits1)!=ex.ground_truth_output:
                    attack_result = 'success' 
                else:
                    attack_result = 'failed'
            
            elif attack:
                if np.argmax(adv_logits)!=ex.ground_truth_output:
                    attack_result = 'success'
                else:
                    attack_result = 'failed'
            
            elif attack1:
                if np.argmax(adv_logits1)!=ex.ground_truth_output:
                    attack_result = 'success'
                else:
                    attack_result = 'failed'

        

        best  = attack_output(attack_result,attack_result,ex.perturbed_text, \
                    ex.original_text, \
                    adv_tr, \
                    org_tr, \
                    0, adv_bleu, adv_chrf,ex.num_queries,ex.ground_truth_output)
        
        attack_dict[i]= best
    
    attack_results2 = classifier(attack_dict, f'{path2}{args.dataset_name}_{args.target_lang}_{args.target_model_name}', device, args, True)
    for i in range(len(attack_results2)):
        attack_dict[i].attack_result2 = attack_results2[i]
    
    return attack_dict
        


def convert_baseline(d,args,device):


    attack_results = classifier(d, f'{path1}{args.dataset_name}_{args.target_lang}_{args.target_model_name}', device, args)
    attack_results2 = classifier(d, f'{path2}{args.dataset_name}_{args.target_lang}_{args.target_model_name}' , device, args, True)

    for i in range(len(d)):
        d[i].attack_result = attack_results[i]
        d[i].attack_result2 = attack_results2[i]




    
    
    return d
        
