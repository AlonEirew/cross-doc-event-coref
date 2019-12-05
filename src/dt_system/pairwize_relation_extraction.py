import torch

from src.dt_system.relation_extraction import RelationExtraction
from src.dt_system.relation_type_enum import RelationTypeEnum
from src.utils.bert_utils import BertFromFile


class PairWizeRelationExtraction(RelationExtraction):
    def __init__(self, pairwize_file, bert_pickle):
        super(PairWizeRelationExtraction, self).__init__()
        self.pairwize_model = torch.load(pairwize_file)
        self.pairwize_model.bert_utils = BertFromFile([bert_pickle])
        self.pairwize_model.eval()

    def extract_sub_relations(self, mention_x, mention_y, relation):
        key1 = mention_x.mention_id + "_" + mention_y.mention_id
        key2 = mention_y.mention_id + "_" + mention_x.mention_id
        if key1 in self.cache:
            return self.cache[key1]
        if key2 in self.cache:
            return self.cache[key2]

        prediction, gold_labels = self.pairwize_model.predict(zip([mention_x], [mention_y]), bs=1)
        if prediction == 1:
            self.cache[key1] = RelationTypeEnum.PAIRWISE
            return RelationTypeEnum.PAIRWISE

        self.cache[key1] = RelationTypeEnum.NO_RELATION_FOUND
        return RelationTypeEnum.NO_RELATION_FOUND

    def get_supported_relations(self):
        return [RelationTypeEnum.PAIRWISE]
