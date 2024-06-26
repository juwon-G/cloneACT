"""

WMT14 TranslationDataset Class
------------------------------------
"""


import collections

import datasets
import numpy as np

from textattack.datasets import HuggingFaceDataset


class WMTTranslationDataset(HuggingFaceDataset):
    """Loads examples from the WMT14 translation dataset using the
    `datasets` package.

    dataset source: http://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/
    """

    def __init__(self, source_lang="en", target_lang="fr", split="test", shuffle=False):
        self._dataset = datasets.load_dataset("wmt14",f"{target_lang}-{source_lang}")[split]
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.shuffled = shuffle
        self.label_map = None
        self.output_scale_factor = None
        self.label_names = None

        if shuffle:
            self._dataset.shuffle()

    def _format_as_dict(self, raw_example):
        translations = raw_example["translation"]
        source = translations[self.source_lang]
        target = translations[self.target_lang]
        source_dict = collections.OrderedDict([("Source", source)])
        return (source_dict, target)
