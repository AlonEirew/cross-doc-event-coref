import pickle
import random

import numpy as np

from src import LIBRARY_ROOT
from src.ext_resources.embedding.embed_elmo import ElmoEmbedding, ElmoEmbeddingOffline
from src.obj.topics import Topics


def create_export_features_to_file(examples, embed, out_pickle_file):
    feat_to_vec_list = list()
    for mention1, mention2 in examples:
        features = feat_to_vec(mention1, mention2, embed)
        feat_to_vec_list.append(features)

    pickle.dump(feat_to_vec_list, open(out_pickle_file, "wb"))


def feat_to_vec(mention1, mention2, embed):
    sentence1_words = ' '.join(mention1.mention_context)
    sentence2_words = ' '.join(mention2.mention_context)
    context1_full_vec = None
    context2_full_vec = None
    if sentence1_words in embed.embeder:
        context1_full_vec = embed.embeder[sentence1_words]
        mention1_vec = ElmoEmbedding.get_mention_vec_from_sent(context1_full_vec, mention1.tokens_number)
        # mention1_feat = np.vstack((context1_full_vec, mention1_vec))

    if sentence2_words in embed.embeder:
        context2_full_vec = embed.embeder[sentence2_words]
        mention2_vec = ElmoEmbedding.get_mention_vec_from_sent(context2_full_vec, mention2.tokens_number)
        # mention2_feat = np.vstack((context2_full_vec, mention2_vec))

    gold_label = 1 if mention1.coref_chain == mention2.coref_chain else 0
    # ret_gold = torch.tensor([gold_label])
    # ret_sent1 = torch.from_numpy(np.vstack(context1_full_vec))
    # ret_sent2 = torch.from_numpy(np.vstack(context2_full_vec))
    # ret_ment1 = torch.from_numpy(mention1_vec)
    # ret_ment2 = torch.from_numpy(mention2_vec)
    ret_sent1 = np.vstack(context1_full_vec)
    ret_sent2 = np.vstack(context2_full_vec)

    # if use_cuda:
    #     ret_sent1 = ret_sent1.cuda()
    #     ret_sent2 = ret_sent2.cuda()
    #     ret_ment1 = ret_ment1.cuda()
    #     ret_ment2 = ret_ment2.cuda()
    #     ret_gold = ret_gold.cuda()

    return [ret_sent1, mention1_vec, ret_sent2, mention2_vec, gold_label]


def create_features_from_pos_neg(positive_exps, negative_exps):
    feats = list()
    feats.extend(positive_exps)
    feats.extend(negative_exps)
    # feats.extend(random.sample(negative_exps, len(positive_exps) * 2))
    # random.shuffle(feats)
    return feats


def create_pos_neg_pairs(topics):
    clusters = dict()
    positive_pairs = list()
    negative_pairs = list()
    topic = topics.topics_list[0]
    for mention in topic.mentions:
        if mention.coref_chain not in clusters:
            clusters[mention.coref_chain] = list()
        clusters[mention.coref_chain].append(mention)

    # create positive examples
    for coref, mentions_list in clusters.items():
        for mention1 in mentions_list:
            for mention2 in mentions_list:
                if mention1.mention_id != mention2.mention_id:
                    if len(mention1.mention_context) > 100 or len(mention2.mention_context) > 100:
                        continue

                    positive_pairs.append((mention1, mention2))

    # create negative examples
    for coref1, mentions_list1 in clusters.items():
        for coref2, mentions_list2 in clusters.items():
            index1 = random.randint(0, len(mentions_list1) - 1)
            index2 = random.randint(0, len(mentions_list2) - 1)
            if mentions_list1[index1].coref_chain != mentions_list2[index2].coref_chain:
                negative_pairs.append((mentions_list1[index1], mentions_list2[index2]))
        if len(negative_pairs) > len(positive_pairs):
            break

    print('pos-' + str(len(positive_pairs)))
    print('neg-' + str(len(negative_pairs)))
    return positive_pairs, negative_pairs


def extract_features(json_file):
    topics = Topics()
    topics.create_from_file(json_file, keep_order=True)

    print('Create pos/neg examples for-' + json_file)
    positive, negative = create_pos_neg_pairs(topics)
    feats = create_features_from_pos_neg(positive, negative)
    # print('Total Train examples-' + str(len(feats)))

    return feats


if __name__ == '__main__':
    # _event_train_file = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Train_Event_gold_mentions.json'
    # _event_dev_file = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Dev_Event_gold_mentions.json'
    # _event_test_file = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Test_Event_gold_mentions.json'
    _event_small_file = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Small_Event_gold_mentions.json'

    # _event_train_feat_pickle = str(LIBRARY_ROOT) + '/resources/corpora/wiki/feats/WIKI_Train_Event_Features.csv'
    # _event_dev_feat_pickle = str(LIBRARY_ROOT) + '/resources/corpora/wiki/feats/WIKI_Dev_Event_Features.csv'
    # _event_test_feat_pickle = str(LIBRARY_ROOT) + '/resources/corpora/wiki/feats/WIKI_Test_Event_Features.csv'
    _event_small_feat_pickle = str(LIBRARY_ROOT) + '/resources/corpora/wiki/feats/WIKI_Small_Event_Features.pickle'

    _bert_file = str(
        LIBRARY_ROOT) + '/resources/preprocessed_external_features/embedded/wiki_all_embed_bert_all_layers.pickle'

    bert = ElmoEmbeddingOffline(_bert_file)

    # train_feats = extract_features(_event_train_file)
    # dev_feats = extract_features(_event_dev_file)
    # test_feats = extract_features(_event_test_file)
    small_feats = extract_features(_event_small_file)

    # create_export_features_to_file(train_feats, bert, _event_train_feat_csv)
    # create_export_features_to_file(dev_feats, bert, _event_dev_feat_csv)
    # create_export_features_to_file(test_feats, bert, _event_test_feat_csv)
    create_export_features_to_file(small_feats, bert, _event_small_feat_pickle)
