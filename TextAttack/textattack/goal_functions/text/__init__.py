"""

Goal Function for Text to Text case
---------------------------------------------------------------------

"""

from .minimize_bleu import MinimizeBleu
from .non_overlapping_output import NonOverlappingOutput
from .text_to_text_goal_function import TextToTextGoalFunction
from .untargeted_class_tr import UntargetedClassTr
from .untargeted_class_tr_s import UntargetedClassTrS
from .untargeted_class_tr_BLEURT import UntargetedClassTrBLEURT
from .targeted_class_tr import targetedClassTr