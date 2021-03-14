import torch
from enum import Enum


class RelationTypeEnum(Enum):
    NO_RELATION_FOUND = 0
    EXACT_STRING = 1
    SAME_HEAD_LEMMA = 2
    PAIRWISE = 3


class RelationExtraction(object):
    def __init__(self):
        self.cache = dict()

    def predict(self, batch_features, bs):
        prediction = list()
        for pair in batch_features:
            prediction.append(self.solve(pair[0], pair[1]))

        return torch.tensor(prediction), None

    def solve(self, mention_x, mention_y):
        raise NotImplementedError()

    def get_supported_relation(self):
        raise NotImplementedError()

    @staticmethod
    def get_extract_method(extract_method_str: str):
        if extract_method_str == "pairwize":
            return RelationTypeEnum.PAIRWISE
        elif extract_method_str == "head_lemma":
            return RelationTypeEnum.SAME_HEAD_LEMMA
        elif extract_method_str == "exact_string":
            return RelationTypeEnum.EXACT_STRING
        raise ValueError("Extract method=" + extract_method_str + " not supported")


class ExactStringRelationExtractor(RelationExtraction):
    def __init__(self):
        super(ExactStringRelationExtractor, self).__init__()

    def solve(self, mention_x, mention_y):
        mention1_str = mention_x.tokens_str
        mention2_str = mention_y.tokens_str
        return 1 if mention1_str.lower() == mention2_str.lower() else 0

    def get_supported_relation(self):
        return RelationTypeEnum.EXACT_STRING


class HeadLemmaRelationExtractor(RelationExtraction):
    def __init__(self):
        super(HeadLemmaRelationExtractor, self).__init__()

    def solve(self, mention_x, mention_y):
        return 1 if mention_x.mention_head_lemma.lower() == mention_y.mention_head_lemma.lower() else 0

    def get_supported_relation(self):
        return RelationTypeEnum.SAME_HEAD_LEMMA
