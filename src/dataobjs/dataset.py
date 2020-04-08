import logging
import pickle
from itertools import combinations

import enum
import random
import re

from src.dataobjs.topics import Topics, Topic

logger = logging.getLogger(__name__)


# creating enumerations using class
class Split(enum.Enum):
    Test = 1
    Dev = 2
    Train = 3
    NA = 4


class POLARITY(enum.Enum):
    POSITIVE = 1
    NEGATIVE = 2


class DataSet(object):
    def __init__(self, name="DataSetSuper", ratio=-1):
        self.ratio = ratio
        self.name = name

    def load_pos_neg_pickle(self, pos_file, neg_file):
        logger.info("Loading pos file-" + pos_file)
        logger.info("Loading neg file-" + neg_file)

        pos_pairs = pickle.load(open(pos_file, "rb"))
        neg_pairs = pickle.load(open(neg_file, "rb"))

        if self.ratio > 0:
            if len(neg_pairs) > (len(pos_pairs) * self.ratio):
                neg_pairs = neg_pairs[0:len(pos_pairs) * self.ratio]

        logger.info('pos-' + str(len(pos_pairs)))
        logger.info('neg-' + str(len(neg_pairs)))
        return self.create_features_from_pos_neg(pos_pairs, neg_pairs)

    def get_pairwise_feat(self, data_file, to_topics=False):
        topics_ = Topics()
        topics_.create_from_file(data_file, keep_order=True)
        logger.info('Create pos/neg examples')
        # Create positive and negative pair within the same ECB+ topic
        positive_, negative_ = self.create_pos_neg_pairs(topics_, to_topics)

        if self.ratio > 0:
            if len(negative_) > (len(positive_) * self.ratio):
                negative_ = negative_[0:len(positive_) * self.ratio]

        logger.info('pos-' + str(len(positive_)))
        logger.info('neg-' + str(len(negative_)))
        return positive_, negative_

    @classmethod
    def create_pos_neg_pairs(cls, topics, to_topic):
        raise NotImplementedError("Method implemented only in subclasses")

    def load_datasets(self, split_file):
        logger.info('Create Features:' + self.name)
        positive_, negative_ = self.get_pairwise_feat(split_file)
        split_feat = self.create_features_from_pos_neg(positive_, negative_)
        return split_feat

    @staticmethod
    def create_features_from_pos_neg(positive_exps, negative_exps):
        feats = list()
        feats.extend(positive_exps)
        feats.extend(negative_exps)
        # feats.extend(random.sample(negative_exps, len(positive_exps) * 2))
        random.shuffle(feats)
        return feats

    @staticmethod
    def check_and_add_pair(_map, _pairs, mention1, mention2):
        if mention1 is None or mention2 is None:
            return False

        mentions_key1 = mention1.mention_id + '_' + mention2.mention_id
        mentions_key2 = mention2.mention_id + '_' + mention1.mention_id
        if mentions_key1 not in _map and mentions_key2 not in _map:
            _pairs.append((mention1, mention2))
            _map[mentions_key1] = True
            _map[mentions_key2] = True
            return True
        return False


class EcbDataSet(DataSet):
    def __init__(self):
        super(EcbDataSet, self).__init__(name="ECB")

    @classmethod
    def create_pos_neg_pairs(cls, topics, to_topic):
        if not to_topic:
            topics = cls.from_ecb_subtopic_to_topic(topics)

        # create positive examples
        positive_pairs = cls.create_pairs(topics, POLARITY.POSITIVE)
        # create negative examples
        negative_pairs = cls.create_pairs(topics, POLARITY.NEGATIVE)
        random.shuffle(negative_pairs)

        return positive_pairs, negative_pairs

    @classmethod
    def create_pairs(cls, topics, polarity):
        _map = dict()
        _pairs = list()
        # create positive examples
        for topic in topics.topics_dict.values():
            for mention1 in topic.mentions:
                for mention2 in topic.mentions:
                    if mention1.mention_id != mention2.mention_id:
                        if polarity == POLARITY.POSITIVE and mention1.coref_chain == mention2.coref_chain:
                            cls.check_and_add_pair(_map, _pairs, mention1, mention2)
                        elif polarity == POLARITY.NEGATIVE and mention1.coref_chain != mention2.coref_chain:
                            cls.check_and_add_pair(_map, _pairs, mention1, mention2)

        return _pairs

    @staticmethod
    def from_ecb_subtopic_to_topic(topics):
        new_topics = Topics()
        for sub_topic in topics.topics_dict.values():
            id_num_groups = re.search(r"\b(\d+)\D+", str(sub_topic.topic_id))
            if id_num_groups is not None:
                id_num = id_num_groups.group(1)
                ret_topic = new_topics.get_topic_by_id(id_num)
                if ret_topic is None:
                    ret_topic = Topic(id_num)
                    new_topics.topics_dict[ret_topic.topic_id] = ret_topic

                ret_topic.mentions.extend(sub_topic.mentions)
            else:
                return topics

        return new_topics


class WecDataSet(DataSet):
    def __init__(self, ratio=-1, split=Split.NA):
        super(WecDataSet, self).__init__("WEC", ratio)
        self.split = split

    def create_pos_neg_pairs(self, topics, sub_topics):
        positive_pairs = EcbDataSet.create_pairs(topics, POLARITY.POSITIVE)

        if self.split == Split.Train:
            clusters = topics.convert_to_clusters()
            negative_pairs = self.create_neg_pairs_wec(clusters)
        else:
            negative_pairs = EcbDataSet.create_pairs(topics, POLARITY.NEGATIVE)
        return positive_pairs, negative_pairs

    @classmethod
    def create_neg_pairs_wec(cls, clusters):
        all_mentions = []
        index = -1
        all_ment_index = -1
        found = True
        while found:
            found = False
            index += 1
            all_mentions.append([])
            all_ment_index += 1
            for _, mentions_list in clusters.items():
                if len(mentions_list) > index:
                    all_mentions[all_ment_index].append(mentions_list[index])
                    found = True

        # create WEC negative challenging examples
        list_combined_pairs = list()
        for mention_list in all_mentions:
            if len(mention_list) > 2:
                cls.create_combinations(mention_list, list_combined_pairs)

        return list_combined_pairs

    @staticmethod
    def create_combinations(mentions, list_combined_pairs):
        MAX_SELECT = 1000000
        pair_list = list(combinations(mentions, 2))
        if len(pair_list) > MAX_SELECT:
            pair_list = random.sample(pair_list, MAX_SELECT)

        list_combined_pairs.extend(pair_list)
