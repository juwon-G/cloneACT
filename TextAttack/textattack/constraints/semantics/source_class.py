"""
BERT Score
---------------------
BERT Score is introduced in this paper (BERTScore: Evaluating Text Generation with BERT) `arxiv link`_.

.. _arxiv link: https://arxiv.org/abs/1904.09675

BERT Score measures token similarity between two text using contextual embedding.

To decide which two tokens to compare, it greedily chooses the most similar token from one text and matches it to a token in the second text.

"""

from transformers import BertTokenizer, BertForSequenceClassification

from textattack.constraints import Constraint
from textattack.shared import utils


class SourceClass(Constraint):
    """A constraint on the class of source language.

    Args:
        

        compare_against_original (bool):
            If ``True``, compare new ``x_adv`` against the original ``x``.
            Otherwise, compare it against the previous ``x_adv``.
    """

    def __init__(
        self,
        compare_against_original=True,
    ):
        super().__init__(compare_against_original)
        self.tokenizer = BertTokenizer.from_pretrained("gchhablani/bert-base-cased-finetuned-sst2")
        self.model = BertForSequenceClassification.from_pretrained("gchhablani/bert-base-cased-finetuned-sst2").to(utils.device)
        self.model.eval()

    def _class(self, starting_text, transformed_text):
        """Returns the metric similarity between the embedding of the starting
        text and the transformed text.

        Args:
            starting_text: The ``AttackedText``to use as a starting point.
            transformed_text: A transformed ``AttackedText``

        Returns:
            The similarity between the starting and transformed text using BERTScore metric.
        """
        cand = transformed_text.text
        ref = starting_text.text

        inputs_org = self.tokenizer(ref, return_tensors="pt").to(utils.device)
        inputs_adv = self.tokenizer(cand, return_tensors="pt").to(utils.device)

        logits_org = self.model(**inputs_org).logits[0]
        logits_adv = self.model(**inputs_adv).logits[0]

        return logits_org.argmax().item(), logits_adv.argmax().item()

    def _check_constraint(self, transformed_text, reference_text):
        """Return `True` if two classes are the same"""
        class_org , class_adv = self._class(reference_text, transformed_text)
        if class_org == class_adv:
            return True
        else:
            return False

    