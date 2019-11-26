import enum
import logging
import multiprocessing
import pickle
from itertools import combinations

import random
import re

from src.dataobjs.topics import Topics, Topic

logger = logging.getLogger(__name__)


# creating enumerations using class
class SPLIT(enum.Enum):
    Test = 1
    Dev = 2
    Train = 3


class POLARITY(enum.Enum):
    POSITIVE = 1
    NEGATIVE = 2


class DATASET(enum.Enum):
    ECB = 1
    WEC = 2


def load_pos_neg_pickle(pos_file, neg_file, alpha):
    logger.info("Loading pos file-" + pos_file)
    logger.info("Loading neg file-" + neg_file)

    pos_pairs = pickle.load(open(pos_file, "rb"))
    neg_pairs = pickle.load(open(neg_file, "rb"))

    if alpha > 0:
        if len(neg_pairs) > (len(pos_pairs) * alpha):
            neg_pairs = neg_pairs[0:len(pos_pairs) * alpha]

    logger.info('pos-' + str(len(pos_pairs)))
    logger.info('neg-' + str(len(neg_pairs)))
    return create_features_from_pos_neg(pos_pairs, neg_pairs)


def load_datasets(split_file, alpha, dataset):
    logger.info('Create Features:' + dataset.name)
    positive_, negative_ = get_feat(split_file, alpha, dataset)
    split_feat = create_features_from_pos_neg(positive_, negative_)
    return split_feat


def get_feat(data_file, alpha, dataset, multiprocess=False):
    topics_ = Topics()
    topics_.create_from_file(data_file, keep_order=True)

    logger.info('Create pos/neg examples')
    if dataset == DATASET.ECB:
        positive_, negative_ = create_pos_neg_pairs_ecb(from_subtopic_to_topic(topics_))
    else:
        clusters = convert_to_clusters(topics_)
        positive_ = create_pos_pairs_wec(clusters)
        negative_ = create_neg_pairs_wec(clusters, multiprocess)

    if alpha > 0:
        if len(negative_) > (len(positive_) * alpha):
            negative_ = negative_[0:len(positive_) * alpha]

    logger.info('pos-' + str(len(positive_)))
    logger.info('neg-' + str(len(negative_)))
    return positive_, negative_


def create_pos_pairs_wec(clusters):
    positives_map = dict()
    positive_pairs = list()

    # create positive examples
    for _, mentions_list in clusters.items():
        for mention1 in mentions_list:
            for mention2 in mentions_list:
                if mention1.mention_id != mention2.mention_id:
                    check_and_add_pair(positives_map, positive_pairs, mention1, mention2)

    return positive_pairs


def create_neg_pairs_wec(clusters, multiprocess):
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
    multiprocessing.set_start_method('spawn')
    list_combined_pairs = list()
    pool = []
    for mention_list in all_mentions:
        if multiprocess:
            combined = multiprocessing.Process(target=create_combinations, args=(mention_list, 2, list_combined_pairs))
            combined.start()
            pool.append(combined)
            print()
        else:
            create_combinations(mention_list, 2, list_combined_pairs)

    for worker in pool:
        worker.join()

    return list_combined_pairs


def create_combinations(mentions, r, list_combined_pairs):
    name = multiprocessing.current_process().name
    print(name, 'Starting')
    MAX_SELECT = 500000
    pair_list = list(combinations(mentions, r))
    if len(pair_list) > MAX_SELECT:
        pair_list = random.sample(pair_list, MAX_SELECT)

    list_combined_pairs.extend(pair_list)


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
                    elif polarity == POLARITY.NEGATIVE and mention1.coref_chain != mention2.coref_chain:
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
