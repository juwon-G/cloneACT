"""

Particle Swarm Optimization
==================================

(Word-level Textual Adversarial Attacking as Combinatorial Optimization)

"""
from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassTr
from textattack.search_methods import ParticleSwarmOptimization
from textattack.transformations import WordSwapHowNet

from .attack_recipe import AttackRecipe


class ClassTranslationPSO(AttackRecipe):
    """
    PSO for targeted attack against translation using a classifier.
    """

    @staticmethod
    def build(model_wrapper, goal_function="UntargetedClassTr",alpha=0.5,thr=0.3,ground_max_score=None,max_min_score=None,dif_score=None,num_class=1):
        #
        # Swap words with their synonyms extracted based on the HowNet.
        #
        transformation = WordSwapHowNet()
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # Use untargeted classification for demo, can be switched to targeted one
        #
        goal_function = UntargetedClassTr(model_wrapper,alpha=alpha,thr=thr,ground_max_score=ground_max_score,max_min_score=max_min_score,dif_score=dif_score,num_class=num_class)
        #
        # Perform word substitution with a Particle Swarm Optimization (PSO) algorithm.
        #
        search_method = ParticleSwarmOptimization(pop_size=60, max_iters=20)

        return Attack(goal_function, constraints, transformation, search_method)
