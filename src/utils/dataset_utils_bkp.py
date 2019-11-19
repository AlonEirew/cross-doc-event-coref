import enum
import logging

import math
import re

import random

from src.dataobjs.topics import Topics, Topic

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
        positive_, negative_ = create_pos_neg_pairs_ecb(from_subtopic_to_topic(topics_))
    else:
        positive_, negative_ = create_pos_neg_pairs_wec(topics_, alpha)

    if split == SPLIT.TRAIN:
        if len(negative_) > (len(positive_) * alpha):
            negative_ = negative_[0:len(positive_) * alpha]

    logger.info('pos-' + str(len(positive_)))
    logger.info('neg-' + str(len(negative_)))
    return positive_, negative_


def create_pos_neg_pairs_wec(topics, alpha):
    positives_map = dict()
    positive_pairs = list()
    clusters = convert_to_clusters(topics)

    # create positive examples
    for _, mentions_list in clusters.items():
        for mention1 in mentions_list:
            for mention2 in mentions_list:
                if mention1.mention_id != mention2.mention_id:
                    check_and_add_pair(positives_map, positive_pairs, mention1, mention2)

    # create WEC negative challenging examples
    negative_map = dict()
    negative_pairs_lemma = list()
    negative_pairs_no_lemma = list()
    threash_same_lema = math.ceil((len(positive_pairs) * alpha) / 50)
    threash_not_lema = math.ceil((len(positive_pairs) * alpha * 49) / 50)
    cond = True
    index = -1
    while cond:
        index += 1
        found_at_list_1 = False
        for _, mentions_list1 in clusters.items():
            for _, mentions_list2 in clusters.items():
                if mentions_list1[0].coref_chain != mentions_list2[0].coref_chain:
                    if len(negative_pairs_no_lemma) <= threash_not_lema:
                        mention1, mention2 = check_clusters_for_pairs(mentions_list1, mentions_list2, index, lambda a, b: a != b)
                        if check_and_add_pair(negative_map, negative_pairs_no_lemma, mention1, mention2):
                            found_at_list_1 = True
                    if len(negative_pairs_lemma) <= threash_same_lema:
                        mention11, mention22 = check_clusters_for_pairs(mentions_list1, mentions_list2, index, lambda a, b: a == b)
                        if check_and_add_pair(negative_map, negative_pairs_lemma, mention11, mention22):
                            found_at_list_1 = True

        if not found_at_list_1:
            cond = False

    logger.info("Found neg_same_lemma pairs-" + str(len(negative_pairs_lemma)))
    logger.info("Found neg_no_same_lemma pairs-" + str(len(negative_pairs_no_lemma)))

    negative_pairs = negative_pairs_lemma + negative_pairs_no_lemma
    random.shuffle(negative_pairs)
    return positive_pairs, negative_pairs


def check_clusters_for_pairs(mentions_list1, mentions_list2, index, condition):
    if index < len(mentions_list1) and index < len(mentions_list2):
        if condition(mentions_list1[index].mention_head, mentions_list2[index].mention_head):
            return mentions_list1[index], mentions_list2[index]

    return None, None


def convert_to_clusters(topics):
    clusters = dict()
    for topic in topics.topics_list:
        for mention in topic.mentions:
            if mention.coref_chain not in clusters:
                clusters[mention.coref_chain] = list()
            clusters[mention.coref_chain].append(mention)
        # break
    return clusters


def create_pos_neg_pairs_ecb(topics):
    # topic = topics.topics_list[0]
    new_topics = from_subtopic_to_topic(topics)
    # create positive examples
    positive_pairs = create_pairs(new_topics, POLARITY.POSITIVE)
    # create negative examples
    negative_pairs = create_pairs(new_topics, POLARITY.NEGATIVE)
    random.shuffle(negative_pairs)

    return positive_pairs, negative_pairs


def create_pairs(topics, polarity):
    _map = dict()
    _pairs = list()
    # create positive examples
    for topic in topics.topics_list:
        for mention1 in topic.mentions:
            for mention2 in topic.mentions:
                if mention1.mention_id != mention2.mention_id:
                    if polarity == POLARITY.POSITIVE and mention1.coref_chain == mention2.coref_chain:
                        check_and_add_pair(_map, _pairs, mention1, mention2)
                    elif polarity == POLARITY.NEGATIVE and mention1.coref_chain != mention2.coref_chain and \
                        not mention1.is_singleton and not mention2.is_singleton:
                        check_and_add_pair(_map, _pairs, mention1, mention2)

    return _pairs


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
