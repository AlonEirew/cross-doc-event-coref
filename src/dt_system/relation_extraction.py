from src.dt_system.relation_type_enum import RelationTypeEnum


class RelationExtraction(object):
    def __init__(self):
        self.cache = dict()

    def extract_relation(self, mention_x, mention_y, relation):
        ret_relation = RelationTypeEnum.NO_RELATION_FOUND
        if relation in self.get_supported_relations():
            ret_relation = self.extract_sub_relations(mention_x, mention_y, relation)
        return ret_relation

    def extract_all_relations(self, mention_x, mention_y):
        relations = set()
        relations.add(self.extract_sub_relations(mention_x, mention_y, None))
        return relations

    def get_supported_relations(self):
        raise NotImplementedError()

    def extract_sub_relations(self, mention_x, mention_y, relation):
        raise NotImplementedError()
