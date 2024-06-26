"""
Marian model trained to generate text from text
---------------------------------------------------------------------

"""
import json
import os

import torch
import transformers

from textattack.model_args import TEXTATTACK_MODELS
from textattack.models.tokenizers import MarianTokenizer


class MarianForTextToText(torch.nn.Module):
    """A Marian model trained to generate text from text.


    Args:
        mode (string): Name of the model to use.
    """

    def __init__(
        self,
        mode="english_to_french",
        # output_max_length=20,
        # input_max_length=64,
        # num_beams=1,
        # early_stopping=True,
    ):
        super().__init__()
        # self.model = transformers.T5ForConditionalGeneration.from_pretrained("t5-base")
        if mode=="english_to_french":
            target_lang="fr"
        elif mode == "english_to_german":
            target_lang="de"

        name = f'Helsinki-NLP/opus-mt-en-{target_lang}'
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(name)
        self.model.eval()
        self.tokenizer = MarianTokenizer(mode)
        # self.model.config.forced_bos_token_id = self.tokenizer.tokenizer.lang_code_to_id[self.tokenizer.tokenizer.tgt_lang ]
        self.mode = mode
        self.tokenizer.model_max_length = 512
        
        # self.output_max_length = output_max_length
        # self.input_max_length = input_max_length
        # self.num_beams = num_beams
        # self.early_stopping = early_stopping

    def __call__(self, *args, **kwargs):
        # Generate IDs from the model.
        output_ids_list = self.model.generate(
            *args,
            **kwargs,
            # max_length=self.output_max_length,
            # num_beams=self.num_beams,
            # early_stopping=self.early_stopping,
        )
        # Convert ID tensor to string and return.
        return [self.tokenizer.decode(ids) for ids in output_ids_list]

    # def save_pretrained(self, output_dir):
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     config = {
    #         "mode": self.mode,
    #         "output_max_length": self.output_max_length,
    #         "input_max_length": self.input_max_length,
    #         "num_beams": self.num_beams,
    #         "early_stoppping": self.early_stopping,
    #     }
    #     # We don't save it as `config.json` b/c that name conflicts with HuggingFace's `config.json`.
    #     with open(os.path.join(output_dir, "t5-wrapper-config.json"), "w") as f:
    #         json.dump(config, f)
    #     self.model.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, name_or_path):
        """Load trained LSTM model by name or from path.

        Args:
            name_or_path (str): Name of the model (e.g. "t5-en-de") or model saved via `save_pretrained`.
        """
        if name_or_path in TEXTATTACK_MODELS:
            t5 = cls(TEXTATTACK_MODELS[name_or_path])
            return t5
        # else:
        #     config_path = os.path.join(name_or_path, "t5-wrapper-config.json")
        #     with open(config_path, "r") as f:
        #         config = json.load(f)
        #     t5 = cls.__new__(cls)
        #     for key in config:
        #         setattr(t5, key, config[key])
        #     t5.model = transformers.T5ForConditionalGeneration.from_pretrained(
        #         name_or_path
        #     )
        #     t5.tokenizer = T5Tokenizer(t5.mode, max_length=t5.output_max_length)
        #     return t5

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
