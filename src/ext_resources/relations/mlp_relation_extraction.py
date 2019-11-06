import sys
from typing import List, Set

import torch

from src.ext_resources.relations.relation_extraction import RelationExtraction
from src.ext_resources.relations.relation_types_enums import RelationType
from src.dl_experiments.snli_coref_train import feat_to_vec
from src.obj.mention_data import MentionDataLight

sys.path.insert(0, 'src/systems/models/mlp_new')


class MLPRelationExtraction(RelationExtraction):
    def __init__(self, model_file, embed):
        self.model = torch.load(model_file)
        self.embed = embed

    def extract_sub_relations(self, mention_x: MentionDataLight, mention_y: MentionDataLight,
                              relation: RelationType) -> RelationType:
        sent1_feat, mention1_feat, sent2_feat, mention2_feat, true_label = feat_to_vec(mention_x, mention_y, self.embed, True)
        prediction = self.model.predict(sent1_feat, mention1_feat, sent2_feat, mention2_feat)
        if prediction == 1:
            return RelationType.OTHER
        else:
            return RelationType.NO_RELATION_FOUND

    def extract_all_relations(self, mention_x: MentionDataLight,
                              mention_y: MentionDataLight) -> Set[RelationType]:
        relations = set()
        relations.add(self.extract_all_relations(mention_x, mention_y, RelationType.OTHER))
        return relations

    @staticmethod
    def get_supported_relations() -> List[RelationType]:
        return [RelationType.OTHER]
