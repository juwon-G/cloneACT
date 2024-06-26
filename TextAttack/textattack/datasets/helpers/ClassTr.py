"""

SST2 TranslationClassification Dataset Class
------------------------------------
"""


import collections

import datasets
import numpy as np

from textattack.datasets import HuggingFaceDataset


class ClassTranslationDataset(HuggingFaceDataset):
    """Loads examples from SST2 en-fr dataset
    """

    def __init__(self, source_lang="en", target_lang="fr", dataset="sst2", model="marian", split="validation", shuffle=False):

        self._dataset = datasets.load_from_disk(f'./models/{model}/{dataset}_en_{target_lang}')[split]

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.shuffled = shuffle
        self.label_map = None
        self.output_scale_factor = None
        self.label_names = None

        if shuffle:
            self._dataset.shuffle()

    def _format_as_dict(self, raw_example):
        source = raw_example["sentence_en"]
        target = raw_example[f'sentence_{self.target_lang}']
        label = raw_example["label"]
        source_dict = collections.OrderedDict([("Source", source)])
        return (source_dict, label)
