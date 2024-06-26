"""

Seq2Sick
================================================
(Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples)
"""
from textattack import Attack
from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import NonOverlappingOutput, MinimizeBleu
from textattack.search_methods import GreedyWordSwapWIR, ParticleSwarmOptimization
from textattack.transformations import WordSwapEmbedding, WordSwapHowNet

from .attack_recipe import AttackRecipe


class TranslationPSO(AttackRecipe):
    """
    PSO for translation (untargeted with bleu score)
    """

    @staticmethod
    def build(model_wrapper, goal_function="non_overlapping"):

        transformation = WordSwapHowNet()
        constraints = [RepeatModification(), StopwordModification()]
        goal_function = MinimizeBleu(model_wrapper)
        search_method = ParticleSwarmOptimization(pop_size=60, max_iters=20)

        return Attack(goal_function, constraints, transformation, search_method)
