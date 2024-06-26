"""

Faster Alzantot Genetic Algorithm
===================================
(Certified Robustness to Adversarial Word Substitutions)


"""

from textattack import Attack
from textattack.constraints.grammaticality.language_models import (
    LearningToWriteLanguageModel,
)
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassTr
from textattack.search_methods import AlzantotGeneticAlgorithm
from textattack.transformations import WordSwapEmbedding

from .attack_recipe import AttackRecipe


class ClassTranslationFasterGenetic(AttackRecipe):
    """
    Fast Alzantot for targeted attack against translation using a classifier.
    """

    @staticmethod
    def build(model_wrapper, goal_function="UntargetedClassTr",alpha=0.5,thr=0.3,ground_max_score=None,max_min_score=None,dif_score=None,num_class=1):
        #
        
        # threshold δ are hyperparameters. We use W = 6 and δ = 5.
        #
        #
        # Swap words with their embedding nearest-neighbors.
        #
        # Embedding: Counter-fitted Paragram Embeddings.
        #
        # "[We] fix the hyperparameter values to S = 60, N = 8, K = 4, and δ = 0.5"
        #
        transformation = WordSwapEmbedding(max_candidates=8)
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # Maximum words perturbed percentage of 20%
        #
        constraints.append(MaxWordsPerturbed(max_percent=0.2))
        #
        # Maximum word embedding euclidean distance of 0.5.
        #
        constraints.append(WordEmbeddingDistance(max_mse_dist=0.5))
        #
        # Language Model
        #
        #
        #
        constraints.append(
            LearningToWriteLanguageModel(
                window_size=6, max_log_prob_diff=5.0, compare_against_original=True
            )
        )
        # constraints.append(LearningToWriteLanguageModel(window_size=5))
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassTr(model_wrapper,alpha=alpha,thr=thr,ground_max_score=ground_max_score,max_min_score=max_min_score,dif_score=dif_score,num_class=num_class)
        #
        # Perform word substitution with a genetic algorithm.
        #
        search_method = AlzantotGeneticAlgorithm(
            pop_size=60, max_iters=40, post_crossover_check=False
        )

        return Attack(goal_function, constraints, transformation, search_method)
