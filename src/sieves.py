import logging
from typing import Tuple

from src.cdc_resources.relations.relation_extraction import RelationExtraction
from src.cdc_resources.relations.relation_types_enums import RelationType
from src.obj.cluster import Cluster

logger = logging.getLogger(__name__)


class SieveClusterMerger(object):
    def __init__(self, excepted_relation: Tuple[RelationType, float],
                 relation_extractor: RelationExtraction):
        """
        Args:
            excepted_relation: tuple with relation to run in sieve,
            threshold to merge clusters
            relation_extractor:
        """
        self.excepted_relation = excepted_relation[0]
        self.threshold = excepted_relation[1]
        self.relation_extractor = relation_extractor

        logger.info('init Sieve, for relation-%s with threshold=%.1f',
                    self.excepted_relation.name, self.threshold)

    def run_sieve(self, cluster_i: Cluster, cluster_j: Cluster) -> bool:
        """
        Args:
            cluster_i:
            cluster_j:

        Returns:
            bool -> indicating whether to merge clusters (True) or not (False)
        """
        matches = 0
        for mention_i in cluster_i.mentions:
            for mention_j in cluster_j.mentions:
                match_result = self.relation_extractor.extract_sub_relations(
                    mention_i, mention_j, self.excepted_relation)
                if match_result == self.excepted_relation:
                    matches += 1

        possible_pairs_len = float(len(cluster_i.mentions) * len(cluster_j.mentions))
        matches_rate = matches / possible_pairs_len

        result = False
        if matches_rate >= self.threshold:
            result = True

        return result
