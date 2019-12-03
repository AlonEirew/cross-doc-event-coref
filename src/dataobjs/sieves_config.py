from typing import List, Tuple

from src.dt_system.relation_type_enum import RelationTypeEnum


class SievesConfiguration(object):
    def __init__(self):
        """Cross document co-reference event and entity evaluation configuration settings"""

        self.__sieves_order = None
        self.__run_evaluation = False

    @property
    def sieves_order(self):
        """
        Sieve definition and Sieve running order

        Tuple[SieveType, RelationType, Threshold(float)] - define sieves to run, were

        Strict- Merge clusters only in case all mentions has current relation between them,
        Relax- Merge clusters in case (matched mentions) / len(cluster_1.mentions)) >= thresh,
        Very_Relax- Merge clusters in case (matched mentions) / (all possible pairs) >= thresh

        RelationType represent the type of sieve to run.

        """
        return self.__sieves_order

    @sieves_order.setter
    def sieves_order(self, sieves_order: List[Tuple[RelationTypeEnum, float]]):
        self.__sieves_order = sieves_order

    @property
    def run_evaluation(self):
        """Should run evaluation (True/False)"""
        return self.__run_evaluation

    @run_evaluation.setter
    def run_evaluation(self, run_evaluation: bool):
        self.__run_evaluation = run_evaluation


class EventSievesConfiguration(SievesConfiguration):
    def __init__(self):
        super(EventSievesConfiguration, self).__init__()

        self.run_evaluation = True

        self.sieves_order = [
            (RelationTypeEnum.SAME_HEAD_LEMMA, 1.0)
        ]


class EntitySievesConfiguration(SievesConfiguration):
    def __init__(self):
        super(EntitySievesConfiguration, self).__init__()

        self.run_evaluation = True

        self.sieves_order = [
            (RelationTypeEnum.SAME_HEAD_LEMMA, 1.0)
        ]
