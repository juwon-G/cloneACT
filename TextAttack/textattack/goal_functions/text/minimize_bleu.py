"""
Goal Function for Attempts to minimize the BLEU score
-------------------------------------------------------


"""

import functools

import nltk

import textattack

from datasets import load_metric

from .text_to_text_goal_function import TextToTextGoalFunction
import time


class MinimizeBleu(TextToTextGoalFunction):
    """Attempts to minimize the BLEU score between the current output
    translation and the reference translation.

        BLEU score was defined in (BLEU: a Method for Automatic Evaluation of Machine Translation).

        `ArxivURL`_

    .. _ArxivURL: https://www.aclweb.org/anthology/P02-1040.pdf

        This goal function is defined in (It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations).

        `ArxivURL2`_

    .. _ArxivURL2: https://www.aclweb.org/anthology/2020.acl-main.263
    """

    EPS = 1e-10
    thr = 0.5

    def __init__(self, *args, target_bleu=0.0, **kwargs):
        self.target_bleu = target_bleu
        super().__init__(*args, **kwargs)
        

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()
        get_bleu.cache_clear()

    def _is_goal_complete(self, model_output, _):
        score = 1 - self._get_score(model_output, _)
        # print("goal:", score <= MinimizeBleu.thr )
        return score <= MinimizeBleu.thr  #bleu_score <= (self.target_bleu + MinimizeBleu.EPS)


    def _get_score(self, model_output, _):
        # import pdb
        # pdb.set_trace()
        # model_output_at = textattack.shared.AttackedText(model_output)
        # ground_truth_at = textattack.shared.AttackedText(self.ground_truth_output)
        bleu_score = get_bleu(model_output, self.ground_truth_output)
        if self.iter==0:
            self.initial_bleu=bleu_score
            print(bleu_score)
        self.iter+=1

        
        if self.initial_bleu==0: #or self.len<10 or self.len>100:
            score = 1
        # elif time.time()-self.time>900:
        #     score = 1
        #     print("*******skipped time*******")
        else:
            if bleu_score/self.initial_bleu>1:
                score = 0
            else:
                score = 1 - bleu_score/self.initial_bleu
        return score

    def extra_repr_keys(self):
        if self.maximizable:
            return ["maximizable"]
        else:
            return ["maximizable", "target_bleu"]


@functools.lru_cache(maxsize=2**12)
def get_bleu(a, b):
    # import pdb
    # pdb.set_trace()
    bleu = load_metric("sacrebleu")
    # ref = a.words
    # hyp = b.words
    # bleu_score = nltk.translate.bleu_score.sentence_bleu([ref], hyp)
    bleu_score = bleu.compute(predictions=[a], references=[[b]])['score']
    return bleu_score
