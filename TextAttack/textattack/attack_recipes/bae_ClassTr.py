"""
BAE (BAE: BERT-Based Adversarial Examples)
============================================

"""
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassTr, UntargetedClassTrS
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapMaskedLM

from .attack_recipe import AttackRecipe


class ClassTranslationBAE(AttackRecipe):
    """

    BAE 
    """

    @staticmethod
    def build(model_wrapper, goal_function="UntargetedClassTr",alpha=0.5,thr=0.3,ground_max_score=None,max_min_score=None,dif_score=None,num_class=1,cw=False):

        transformation = WordSwapMaskedLM(
            method="bae", max_candidates=50, min_confidence=0.0
        )
        #
        # Don't modify the same word twice or stopwords.
        #
        constraints = [RepeatModification(), StopwordModification()]

        # For the R operations we add an additional check for
        # grammatical correctness of the generated adversarial example by filtering
        # out predicted tokens that do not form the same part of speech (POS) as the
        # original token t_i in the sentence.
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))

       
        use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassTr(model_wrapper,alpha=alpha,thr=thr,ground_max_score=ground_max_score,max_min_score=max_min_score,dif_score=dif_score,num_class=num_class,cw=cw)
        #
        # "We estimate the token importance Ii of each token
        # t_i ∈ S = [t1, . . . , tn], by deleting ti from S and computing the
        # decrease in probability of predicting the correct label y, similar
        # to (Jin et al., 2019).
        #
        # • "If there are multiple tokens can cause C to misclassify S when they
        # replace the mask, we choose the token which makes Sadv most similar to
        # the original S based on the USE score."
        # • "If no token causes misclassification, we choose the perturbation that
        # decreases the prediction probability P(C(Sadv)=y) the most."
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return ClassTranslationBAE(goal_function, constraints, transformation, search_method)
