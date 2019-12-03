import logging

from src.dt_system.relation_extraction import RelationExtraction
from src.dt_system.relation_type_enum import RelationTypeEnum
from src.utils.string_utils import StringUtils

logger = logging.getLogger(__name__)


class ComputedRelationExtraction(RelationExtraction):
    def __init__(self):
        super(ComputedRelationExtraction, self).__init__()

    def extract_all_relations(self, mention_x, mention_y):
        relations = set()
        mention_x_str = mention_x.tokens_str
        mention_y_str = mention_y.tokens_str

        if StringUtils.is_pronoun(mention_x_str.lower()) or StringUtils.is_pronoun(mention_y_str.lower()):
            relations.add(RelationTypeEnum.NO_RELATION_FOUND)
            return relations

        relations.add(self.extract_exact_string(mention_x, mention_y))
        relations.add(self.extract_same_head_lemma(mention_x, mention_y))

        if len(relations) == 0:
            relations.add(RelationTypeEnum.NO_RELATION_FOUND)

        return relations

    def extract_sub_relations(self, mention_x, mention_y, relation):
        '''
        :param mention_x:
        :param mention_y:
        :return:
        '''
        mention_x_str = mention_x.tokens_str
        mention_y_str = mention_y.tokens_str

        if StringUtils.is_pronoun(mention_x_str.lower()) or StringUtils.is_pronoun(mention_y_str.lower()):
            return RelationTypeEnum.NO_RELATION_FOUND

        if relation == RelationTypeEnum.EXACT_STRING:
            return self.extract_exact_string(mention_x, mention_y)
        elif relation == RelationTypeEnum.SAME_HEAD_LEMMA:
            return self.extract_same_head_lemma(mention_x, mention_y)
        else:
            return RelationTypeEnum.NO_RELATION_FOUND

    def extract_same_head_lemma(self, mention_x, mention_y):
        # if StringUtils.is_preposition(mention_x.mention_head_lemma.lower()) or \
        #         StringUtils.is_preposition(mention_y.mention_head_lemma.lower()):
        #     return RelationTypeEnum.NO_RELATION_FOUND
        if mention_x.mention_head_lemma.lower() == mention_y.mention_head_lemma.lower():
            return RelationTypeEnum.SAME_HEAD_LEMMA
        return RelationTypeEnum.NO_RELATION_FOUND

    def extract_exact_string(self, mention_x, mention_y):
        relation = RelationTypeEnum.NO_RELATION_FOUND
        mention1_str = mention_x.tokens_str
        mention2_str = mention_y.tokens_str
        if mention1_str.lower() == mention2_str.lower():
            relation = RelationTypeEnum.EXACT_STRING
        return relation

    def get_supported_relations(self):
        return [RelationTypeEnum.EXACT_STRING, RelationTypeEnum.SAME_HEAD_LEMMA]
