
import os
from datasets import load_metric
import torch 
from utils.attack_output import attack_output
from utils.convert import classifier, path1, path2


# tokeniser decode only for zh is different
def decode(ids,tokenizer, model_name, target_lang):
        if model_name=="mbart" and target_lang=="zh":
            return tokenizer.decode(ids, skip_special_tokens=True).replace(" ","")
        else:
            return tokenizer.decode(ids, skip_special_tokens=True)



def black_attack(args, sentence_number, d_i, dataset, black_tokenizer, black_model, device):

    part = "validation"
    metric_bleu = load_metric("sacrebleu")
    metric_chrf = load_metric("chrf")
     
    
    input_ids = black_tokenizer(dataset[part][sentence_number]['sentence_en'], truncation=True)['input_ids']
    
    first_tr_ids = black_model.generate(torch.LongTensor(input_ids).unsqueeze(0).to(device))
    first_tr = [[decode(first_tr_ids[0], black_tokenizer,args.black_model_name,args.target_lang)]]
    
    
    
    adversarial_sent = d_i.adv_sent
    adversarial_input = torch.tensor(black_tokenizer(adversarial_sent)['input_ids']).unsqueeze(0).to(device)
    
    
    # error_rate = wer(adversarial_input[0],input_ids)  
    output_ids = black_model.generate(adversarial_input)
    pred_project_decode = [decode(output_ids[0],black_tokenizer,args.black_model_name,args.target_lang)]
    
    adv_bleu = metric_bleu.compute(predictions=pred_project_decode, references=first_tr)['score']
    adv_chrf = metric_chrf.compute(predictions=pred_project_decode, references=first_tr)['score']

    adv_sent = black_tokenizer.decode(adversarial_input[0].tolist(), skip_special_tokens=True).replace("▁"," ")
    
    best = d_i
    best.adv_sent = adv_sent
    best.adv_tr = pred_project_decode[0]
    best.org_tr = first_tr[0][0]
    best.adv_bleu = adv_bleu
    best.adv_chrf = adv_chrf
    best.query = 1

    attack_dict = {0:best}

    if 'sst2' in args.dataset_name:
        dataset_name = 'sst2'
    elif 'ag_news' in args.dataset_name:
        dataset_name = 'ag_news'
    elif 'rotten_tomatoes' in args.dataset_name:
        dataset_name = 'rotten_tomatoes'
    
    model_name = 'marian' if args.black_model_name=='mbart' else 'mbart'
    
    attack_results = classifier(attack_dict, f'{path1}{dataset_name}_{args.target_lang}_{model_name}', device, args)
    attack_results2 = classifier(attack_dict, f'{path2}{dataset_name}_{args.target_lang}_{model_name}', device, args, True)
    
    best.attack_result = attack_results[0]
    best.attack_result2 = attack_results2[0]

    
    return best 





