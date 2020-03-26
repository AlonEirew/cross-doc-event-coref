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

    def solve(self, mention_x, mention_y):
        raise NotImplementedError()

    def get_supported_relation(self):
        raise NotImplementedError()


class ExactStringRelationExtractor(RelationExtraction):
    def __init__(self):
        super(ExactStringRelationExtractor, self).__init__()

    def solve(self, mention_x, mention_y):
        relation = RelationTypeEnum.NO_RELATION_FOUND
        mention1_str = mention_x.tokens_str
        mention2_str = mention_y.tokens_str
        if mention1_str.lower() == mention2_str.lower():
            relation = RelationTypeEnum.EXACT_STRING
        return relation

    def get_supported_relation(self):
        return RelationTypeEnum.EXACT_STRING


class HeadLemmaRelationExtractor(RelationExtraction):
    def __init__(self):
        super(HeadLemmaRelationExtractor, self).__init__()

    def solve(self, mention_x, mention_y):
        # if StringUtils.is_preposition(mention_x.mention_head_lemma.lower()) or \
        #         StringUtils.is_preposition(mention_y.mention_head_lemma.lower()):
        #     return RelationTypeEnum.NO_RELATION_FOUND
        if mention_x.mention_head_lemma.lower() == mention_y.mention_head_lemma.lower():
            return RelationTypeEnum.SAME_HEAD_LEMMA
        return RelationTypeEnum.NO_RELATION_FOUND

    def get_supported_relation(self):
        return RelationTypeEnum.SAME_HEAD_LEMMA

    def predict(self, batch_features, bs):
        prediction = list()
        for pair in batch_features:
            rel = self.solve(pair[0], pair[1])
            if rel == RelationTypeEnum.SAME_HEAD_LEMMA:
                prediction.append(1)
            else:
                prediction.append(0)

        return torch.tensor(prediction), None


class PairWizeRelationExtraction(RelationExtraction):
    def __init__(self, pairwize_file, pairthreshold=1.0):
        super(PairWizeRelationExtraction, self).__init__()
        self.pairwize_model = torch.load(pairwize_file)
        self.pairwize_model.eval()
        self.pairthreshold = pairthreshold

    def solve(self, mention_x, mention_y):
        key1 = mention_x.mention_id + "_" + mention_y.mention_id
        key2 = mention_y.mention_id + "_" + mention_x.mention_id
        if key1 in self.cache:
            return self.cache[key1]
        if key2 in self.cache:
            return self.cache[key2]

        prediction, gold_labels = self.pairwize_model.predict(zip([mention_x], [mention_y]), bs=1)
        if prediction.item() > self.pairthreshold:
            self.cache[key1] = RelationTypeEnum.PAIRWISE
            return RelationTypeEnum.PAIRWISE

        self.cache[key1] = RelationTypeEnum.NO_RELATION_FOUND
        return RelationTypeEnum.NO_RELATION_FOUND

    def get_supported_relation(self):
        return RelationTypeEnum.PAIRWISE
