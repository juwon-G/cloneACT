#이 파일은 원본에서 수정됨

"""
Marian model + Classifier trained to generate text from text
---------------------------------------------------------------------

"""
import json
import os

import torch
import transformers

from textattack.model_args import TEXTATTACK_MODELS
from textattack.models.tokenizers import MarianTokenizer, ClassTrTokenizer


class MarianClassTr(torch.nn.Module):
    """Marian NMT text from text processed by a french sentiment classifier


    Args:
        mode (string): Name of the T5 model to use.
        output_max_length (int): The max length of the sequence to be generated.
            Between 1 and infinity.
        input_max_length (int): Max length of the input sequence.
        num_beams (int): Number of beams for beam search. Must be between 1 and
            infinity. 1 means no beam search.
        early_stopping (bool): if set to `True` beam search is stopped when at
            least `num_beams` sentences finished per batch. Defaults to `True`.
    """

    def __init__(
        self,
        mode="english_to_french",
        num_class=1,
        dataset="sst2"
    ):
        super().__init__()
        if mode=="english_to_french":
            target_lang="fr"
        elif mode == "english_to_german":
            target_lang="de"
        
        self.num_class = num_class
        self.dataset = dataset

        name = f'Helsinki-NLP/opus-mt-en-{target_lang}'
        classifier_addr = f'./models/results_org/{dataset}_{target_lang}_marian'
        
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(name)
        self.classifier = transformers.AutoModelForSequenceClassification.from_pretrained(classifier_addr)
    
        self.model.eval()
        self.classifier.eval()

        self.tokenizer = MarianTokenizer(mode)
        self.classifier_tokenizer = ClassTrTokenizer(name=classifier_addr)

        self.mode = mode
        self.tokenizer.model_max_length = 512
        

        if self.num_class==2:
            self.classifier1 = transformers.AutoModelForSequenceClassification.from_pretrained("moussaKam/barthez-sentiment-classification")
            self.classifier1.eval()
            self.classifier_tokenizer1 = ClassTrTokenizer(name="moussaKam/barthez-sentiment-classification")

    def __call__(self, device, *args, **kwargs):
        input_ids = kwargs.get('input_ids', None)
        att_mask = kwargs.get('attention_mask',None)

        output_ids = self.model.generate(input_ids = input_ids,attention_mask = att_mask)
        tmp = self.model(input_ids=input_ids,decoder_input_ids=output_ids,attention_mask=att_mask,output_attentions=True)
        org_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        tr_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in output_ids]
        NMT_output = [self.tokenizer.decode(ids) for ids in output_ids]
        batch_size=len(NMT_output)

        def resize(att, batch, org, tr):
            return tuple(layer[batch, :, :(tr if tr else org), :(org if org else tr)].clone() for layer in att)

        att_cro,att_dec,att_enc=[],[],[]
        for i in range(batch_size):
            org_size=len(org_tokens[i])
            tr_size=len(tr_tokens[i])
            att_enc.append(resize(tmp.encoder_attentions,i,org_size,0))
            att_cro.append(resize(tmp.cross_attentions,i,org_size,tr_size))
            att_dec.append(resize(tmp.decoder_attentions,i,0,tr_size))

        if self.num_class==2:
            class_output = [self.classifier1(self.classifier_tokenizer1(tr,return_tensors='pt',truncation=True)['input_ids'].to(device)).logits[0] for tr in NMT_output]
        else:
            class_output = [self.classifier(self.classifier_tokenizer(tr,return_tensors='pt',truncation=True)['input_ids'].to(device)).logits[0] for tr in NMT_output]

        del tmp
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        output = [(NMT_output[i],class_output[i],org_tokens[i],tr_tokens[i],att_enc[i],att_dec[i],att_cro[i]) for i in range(batch_size)]
        return output


    @classmethod
    def from_pretrained(cls, name_or_path):
        """

        Args:
            name_or_path (str): Name of the model (e.g. "t5-en-de") or model saved via `save_pretrained`.
        """
        if name_or_path in TEXTATTACK_MODELS:
            num_class = 1
            if "2class" in name_or_path:
                num_class = 2
            if "sst2" in name_or_path:
                dataset = "sst2"
            elif "mr" in name_or_path:
                dataset = "rotten_tomatoes"
            elif "ag" in name_or_path:
                dataset = "ag_news"
            model = cls(TEXTATTACK_MODELS[name_or_path],num_class=num_class,dataset=dataset)
            return model


    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

