"""
Marian model + Classifier trained to generate text from text
---------------------------------------------------------------------

"""
import json
import os

import torch
import transformers

from textattack.model_args import TEXTATTACK_MODELS
from textattack.models.tokenizers import MbartTokenizer, ClassTrTokenizer


class MbartClassTr(torch.nn.Module):
    """mBART50 text from text processed by a french sentiment classifier


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

        name = 'facebook/mbart-large-50-one-to-many-mmt'
        classifier_addr = f'./models/results_org/{dataset}_{target_lang}_mbart'
        
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(name)
        self.classifier = transformers.AutoModelForSequenceClassification.from_pretrained(classifier_addr)
    
        self.model.eval()
        self.classifier.eval()

        self.tokenizer = MbartTokenizer(mode)
        self.classifier_tokenizer = ClassTrTokenizer(name=classifier_addr)

        self.mode = mode
        self.tokenizer.model_max_length = 512
        self.model.config.forced_bos_token_id = self.tokenizer.tokenizer.lang_code_to_id[self.tokenizer.tokenizer.tgt_lang ]
        
        if self.num_class==2:
            self.classifier1 = transformers.AutoModelForSequenceClassification.from_pretrained("moussaKam/barthez-sentiment-classification")
            self.classifier1.eval()
            self.classifier_tokenizer1 = ClassTrTokenizer("moussaKam/barthez-sentiment-classification")

    def __call__(self, device, *args, **kwargs):
        # Generate IDs from the model.
        output_ids_list = self.model.generate(
            *args,
            **kwargs
        )

        # Convert ID tensor to string and return.
        NMT_output = [self.tokenizer.decode(ids) for ids in output_ids_list]
        class_output = [self.classifier(self.classifier_tokenizer(tr,return_tensors='pt',truncation=True)['input_ids'].to(device)).logits[0] for tr in NMT_output]
        output = [(NMT_output[i],class_output[i]) for i in range(len(NMT_output))]

        if self.num_class==2:
            class_output1 = [self.classifier1(self.classifier_tokenizer1(tr,return_tensors='pt',truncation=True)['input_ids'].to(device)).logits[0] for tr in NMT_output]
            output = [(NMT_output[i],class_output[i],class_output1[i]) for i in range(len(NMT_output))]

        
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

