from typing import List

from src.cdc_resources.relations.relation_types_enums import RelationType
from src.obj.mention_data import MentionDataLight


class RelationExtraction(object):
    def __init__(self):
        pass

    def extract_relation(self, mention_x: MentionDataLight, mention_y: MentionDataLight,
                         relation: RelationType) -> RelationType:
        """
        Base Class Check if Sub class support given relation before executing the sub class

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight
            relation: RelationType

        Returns:
            RelationType: relation in case mentions has given relation and
                RelationType.NO_RELATION_FOUND otherwise
        """
        ret_relation = RelationType.NO_RELATION_FOUND
        if relation in self.get_supported_relations():
            ret_relation = self.extract_sub_relations(mention_x, mention_y, relation)
        return ret_relation

    def extract_sub_relations(self, mention_x: MentionDataLight, mention_y: MentionDataLight,
                              relation: RelationType) -> RelationType:
        raise NotImplementedError

    @staticmethod
    def get_supported_relations() -> List[RelationType]:
        raise NotImplementedError
