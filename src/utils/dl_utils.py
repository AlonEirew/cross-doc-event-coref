import logging
import random

from src.obj.topics import Topics


logger = logging.getLogger(__name__)


def get_feat(data_file, alpha):
    topics_ = Topics()
    topics_.create_from_file(data_file, keep_order=True)

    logger.info('Create pos/neg examples')
    positive_, negative_ = create_pos_neg_pairs(topics_, alpha)
    features = create_features_from_pos_neg(positive_, negative_)

    return features


def create_pos_neg_pairs(topics, alpha):
    clusters = dict()
    positive_pairs = list()
    negative_pairs = list()
    # topic = topics.topics_list[0]
    for topic in topics.topics_list:
        for mention in topic.mentions:
            if mention.coref_chain not in clusters:
                clusters[mention.coref_chain] = list()
            clusters[mention.coref_chain].append(mention)
        break

    # create positive examples
    for _, mentions_list in clusters.items():
        for mention1 in mentions_list:
            for mention2 in mentions_list:
                if mention1.mention_id != mention2.mention_id:
                    if len(mention1.mention_context) > 100 or len(mention2.mention_context) > 100:
                        continue

                    positive_pairs.append((mention1, mention2))

    # create negative examples
    for _, mentions_list1 in clusters.items():
        for _, mentions_list2 in clusters.items():
            index1 = random.randint(0, len(mentions_list1) - 1)
            index2 = random.randint(0, len(mentions_list2) - 1)
            if mentions_list1[index1].coref_chain != mentions_list2[index2].coref_chain:
                negative_pairs.append((mentions_list1[index1], mentions_list2[index2]))
        if len(negative_pairs) > (len(positive_pairs) * alpha):
            break

    logger.info('pos-' + str(len(positive_pairs)))
    logger.info('neg-' + str(len(negative_pairs)))
    return positive_pairs, negative_pairs


def create_features_from_pos_neg(positive_exps, negative_exps):
    feats = list()
    feats.extend(positive_exps)
    feats.extend(negative_exps)
    # feats.extend(random.sample(negative_exps, len(positive_exps) * 2))
    random.shuffle(feats)
    return feats
