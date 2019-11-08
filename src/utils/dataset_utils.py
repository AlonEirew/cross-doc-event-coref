import enum
import logging
import re

import random

from src.obj.topics import Topics, Topic

logger = logging.getLogger(__name__)


# creating enumerations using class
class SPLIT(enum.Enum):
    TEST = 1
    VALIDATION = 2
    TRAIN = 3


class POLARITY(enum.Enum):
    POSITIVE = 1
    NEGATIVE = 2


class DATASET(enum.Enum):
    ECB = 1
    WEC = 2


def load_datasets(split_file, alpha, split, dataset):
    logger.info('Create Features:' + str(split))
    positive_, negative_ = get_feat(split_file, alpha, split, dataset)
    split_feat = create_features_from_pos_neg(positive_, negative_)
    return split_feat


def get_feat(data_file, alpha, split, dataset):
    topics_ = Topics()
    topics_.create_from_file(data_file, keep_order=True)

    logger.info('Create pos/neg examples')
    if dataset == DATASET.ECB:
        positive_, negative_ = create_pos_neg_pairs_ecb(from_subtopic_to_topic(topics_), alpha, split)
    else:
        positive_, negative_ = create_pos_neg_pairs_wec(topics_, alpha, split)

    return positive_, negative_


def create_pos_neg_pairs_wec(topics, alpha, split_type):
    positives_map = dict()
    positive_pairs = list()
    negative_map = dict()
    negative_pairs = list()
    clusters = convert_to_clusters(topics)

    # create positive examples
    for _, mentions_list in clusters.items():
        for mention1 in mentions_list:
            for mention2 in mentions_list:
                if mention1.mention_id != mention2.mention_id:
                    # if len(mention1.mention_context) > 100 or len(mention2.mention_context) > 100:
                    #     continue
                    mentions_key1 = mention1.mention_id + '_' + mention2.mention_id
                    mentions_key2 = mention2.mention_id + '_' + mention1.mention_id
                    if mentions_key1 not in positives_map and mentions_key2 not in positives_map:
                        positive_pairs.append((mention1, mention2))
                        positives_map[mentions_key1] = True
                        positives_map[mentions_key2] = True

    # create WEC negative examples
    for _, mentions_list1 in clusters.items():
        for _, mentions_list2 in clusters.items():
            index1 = random.randint(0, len(mentions_list1) - 1)
            index2 = random.randint(0, len(mentions_list2) - 1)
            if mentions_list1[index1].coref_chain != mentions_list2[index2].coref_chain:
                mentions_key1 = mentions_list1[index1].mention_id + '_' + mentions_list2[index2].mention_id
                mentions_key2 = mentions_list2[index2].mention_id + '_' + mentions_list1[index1].mention_id
                if mentions_key1 not in negative_map and mentions_key2 not in negative_map:
                    negative_pairs.append((mentions_list1[index1], mentions_list2[index2]))
                    negative_map[mentions_key1] = True
                    negative_map[mentions_key2] = True

        if split_type == SPLIT.TRAIN:
            if len(negative_pairs) > (len(positive_pairs) * alpha):
                break

    logger.info('pos-' + str(len(positive_pairs)))
    logger.info('neg-' + str(len(negative_pairs)))
    return positive_pairs, negative_pairs


def convert_to_clusters(topics):
    clusters = dict()
    for topic in topics.topics_list:
        for mention in topic.mentions:
            if mention.coref_chain not in clusters:
                clusters[mention.coref_chain] = list()
            clusters[mention.coref_chain].append(mention)
        # break
    return clusters


def create_pos_neg_pairs_ecb(topics, alpha, split_type):
    # topic = topics.topics_list[0]
    new_topics = from_subtopic_to_topic(topics)
    # create positive examples
    positive_pairs = create_pairs(new_topics, POLARITY.POSITIVE)
    # create negative examples
    negative_pairs = create_pairs(new_topics, POLARITY.NEGATIVE)

    if split_type == SPLIT.TRAIN:
        if len(negative_pairs) > (len(positive_pairs) * alpha):
            negative_pairs = negative_pairs[0:len(positive_pairs) * alpha]

    logger.info('pos-' + str(len(positive_pairs)))
    logger.info('neg-' + str(len(negative_pairs)))
    return positive_pairs, negative_pairs


def create_pairs(topics, polarity):
    _map = dict()
    _pairs = list()
    # create positive examples
    for topic in topics.topics_list:
        for mention1 in topic.mentions:
            for mention2 in topic.mentions:
                if mention1.mention_id != mention2.mention_id:
                    mentions_pairkey1 = mention1.mention_id + '_' + mention2.mention_id
                    mentions_pairkey2 = mention2.mention_id + '_' + mention1.mention_id
                    if polarity == POLARITY.POSITIVE and \
                            mention1.coref_chain == mention2.coref_chain and \
                            mentions_pairkey1 not in _map and mentions_pairkey2 not in _map:
                        app_pair(_map, _pairs, mention1, mention2, mentions_pairkey1, mentions_pairkey2)
                    elif polarity == POLARITY.NEGATIVE and \
                            mention1.coref_chain != mention2.coref_chain and \
                            mentions_pairkey1 not in _map and mentions_pairkey2 not in _map:
                        app_pair(_map, _pairs, mention1, mention2, mentions_pairkey1, mentions_pairkey2)

    return _pairs


def app_pair(_map, _pairs, mention1, mention2, mentions_pairkey1, mentions_pairkey2):
    _pairs.append((mention1, mention2))
    _map[mentions_pairkey1] = True
    _map[mentions_pairkey2] = True


def from_subtopic_to_topic(topics):
    new_topics = Topics()
    for sub_topic in topics.topics_list:
        id_num_groups = re.search(r"\b(\d+)\D+", str(sub_topic.topic_id))
        if id_num_groups is not None:
            id_num = id_num_groups.group(1)
            ret_topic = new_topics.get_topic_by_id(id_num)
            if ret_topic is None:
                ret_topic = Topic(id_num)
                new_topics.topics_list.append(ret_topic)

            ret_topic.mentions.extend(sub_topic.mentions)
        else:
            return topics

    return new_topics


def create_features_from_pos_neg(positive_exps, negative_exps):
    feats = list()
    feats.extend(positive_exps)
    feats.extend(negative_exps)
    # feats.extend(random.sample(negative_exps, len(positive_exps) * 2))
    random.shuffle(feats)
    return feats
