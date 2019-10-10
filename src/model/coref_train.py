import random
import time

import numpy as np
import torch
from torch import optim

from src.cdc_resources.embedding.embed_elmo import ElmoEmbedding, ElmoEmbeddingOffline
from src.model.coref_model import CorefModel
from src.obj.topics import Topics


def train_classifier(model, train_feats, dev_feats, learning_rate, iterations, embed, use_cuda):
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.05)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    total_count = 0
    for epoch in range(iterations):
        cum_loss = 0.0
        # start = int(round(time.time() * 1000))
        pair_count = 0
        for mention1, mention2 in train_feats:
            if pair_count == 5:
                break
            optimizer.zero_grad()
            pair_count += 1
            batch1_start = int(round(time.time() * 1000))
            sent1_feat, mention1_feat, sent2_feat, mention2_feat, true_label = feat_to_vec(mention1, mention2, embed, use_cuda)
            output = model(sent1_feat, mention1_feat, sent2_feat, mention2_feat)
            loss = model.loss_fun(output.view(1, -1), true_label)
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
            total_count += 1

            if total_count % 100 == 0:
                batch1_end = int(round(time.time() * 1000))
                batch1_took = batch1_end - batch1_start
                log(epoch, total_count, pair_count, cum_loss, batch1_took, 0.0)

        accuracy_dev = accuracy_on_dataset(model, dev_feats, embed, use_cuda)
        print('DEV Accuracy at END:')
        log(epoch, total_count, pair_count, cum_loss, 1.0, accuracy_dev)

        accuracy_train = accuracy_on_dataset(model, train_feats, embed, use_cuda)
        print('TRAIN Accuracy at END:')
        log(epoch, total_count, pair_count, cum_loss, 1.0, accuracy_train)


def create_pos_neg_pairs(topics):
    mentions_pairs = set()
    positive_pairs = list()
    negative_pairs = list()
    for topic in topics.topics_list:
        for mention1 in topic.mentions:
            for mention2 in topic.mentions:
                if mention1.mention_id == mention2.mention_id:
                    continue

                key = mention1.mention_id + '&' + mention2.mention_id
                key_reverse = mention2.mention_id + '&' + mention1.mention_id

                if key in mentions_pairs or key_reverse in mentions_pairs:
                    continue

                mentions_pairs.add(key)
                gold_label = mention1.coref_chain == mention2.coref_chain
                if gold_label:
                    positive_pairs.append((mention1, mention2))
                else:
                    negative_pairs.append((mention1, mention2))

    print('pos-' + str(len(positive_pairs)))
    print('neg-' + str(len(negative_pairs)))
    return positive_pairs, negative_pairs


def feat_to_vec(mention1, mention2, embed, use_cuda):
    sentence1_words = ' '.join(mention1.mention_context)
    sentence2_words = ' '.join(mention2.mention_context)
    if sentence1_words in embed.embeder:
        context1_full_vec = embed.embeder[sentence1_words]
        mention1_vec = ElmoEmbedding.get_mention_vec_from_sent(context1_full_vec, mention1.tokens_number)
        # mention1_feat = np.vstack((context1_full_vec, mention1_vec))

    if sentence2_words in embed.embeder:
        context2_full_vec = embed.embeder[sentence2_words]
        mention2_vec = ElmoEmbedding.get_mention_vec_from_sent(context2_full_vec, mention2.tokens_number)
        # mention2_feat = np.vstack((context2_full_vec, mention2_vec))

    gold_label = 1 if mention1.coref_chain == mention2.coref_chain else 0
    ret_gold = torch.tensor([gold_label])
    ret_sent1 = torch.from_numpy(np.vstack(context1_full_vec))
    ret_sent2 = torch.from_numpy(np.vstack(context2_full_vec))
    ret_ment1 = torch.from_numpy(mention1_vec)
    ret_ment2 = torch.from_numpy(mention2_vec)

    if use_cuda:
        ret_sent1 = ret_sent1.cuda()
        ret_sent2 = ret_sent2.cuda()
        ret_ment1 = ret_ment1.cuda()
        ret_ment2 = ret_ment2.cuda()
        ret_gold = ret_gold.cuda()

    return ret_sent1, ret_ment1, ret_sent2, ret_ment2, ret_gold


def log(epoch, total_count, pair_count, cum_loss, took, accuracy):
    if accuracy != 0.0:
        print('%d: %d: loss: %.3f: Accuracy: %.5f: epoch-took: %dmilli' %
              (epoch + 1, total_count, cum_loss / pair_count, accuracy, took))
    else:
        print('%d: %d: loss: %.3f: epoch-took: %dmilli' %
              (epoch + 1, total_count, cum_loss / pair_count, took))


def accuracy_on_dataset(model, features, embedd, use_cuda):
    good = bad = 0.0
    for mention1, mention2 in features:
        sent1_feat, mention1_feat, sent2_feat, mention2_feat, true_label = feat_to_vec(mention1, mention2, embedd, use_cuda)
        predictions = model.predict(sent1_feat, mention1_feat, sent2_feat, mention2_feat)

        if predictions == true_label.item():
            good += 1
        else:
            bad += 1

    return good / (good + bad)


def create_features_from_pos_neg(positive_exps, negative_exps):
    feats = list()
    feats.extend(positive_exps)
    feats.extend(random.sample(negative_exps, len(positive_exps) * 2))
    random.shuffle(feats)
    return feats


def get_feat(train_file, dev_file):
    topics_train = Topics()
    topics_train.create_from_file(train_file, keep_order=True)
    topics_dev = Topics()
    topics_dev.create_from_file(dev_file, keep_order=True)

    print('Create Train pos/neg examples')
    positive_train, negative_train = create_pos_neg_pairs(topics_train)
    train_feats = create_features_from_pos_neg(positive_train, negative_train)
    print('Total Train examples-' + str(len(train_feats)))

    print('Create Test pos/neg examples')
    positive_dev, negative_dev = create_pos_neg_pairs(topics_dev)
    dev_feats = create_features_from_pos_neg(positive_dev, negative_dev)
    print('Total Dev examples-' + str(len(dev_feats)))

    return train_feats, dev_feats


def joint_feats(event_feat, entity_feat):
    print('Create Joint features:')
    joint_feat = list()
    joint_feat.extend(event_feat)
    joint_feat.extend(entity_feat)
    random.shuffle(joint_feat)
    print('Joint features size-' + str(len(joint_feat)))
    return joint_feat


def run_train(event_train_file, event_dev_file, entity_train_file, entity_dev_file,
              bert_file, learning_rate, iterations, model_out, run_type, joint, use_cuda):

    print('Create Events Features:')
    event_train_feat, event_dev_feat = get_feat(event_train_file, event_dev_file)
    print('Create Entity Features:')
    entity_train_feat, entity_dev_feat = get_feat(entity_train_file, entity_dev_file)

    bert = ElmoEmbeddingOffline(bert_file)
    model = CorefModel(MODEL_SIZE, MODEL_SIZE, MODEL_SIZE)

    if use_cuda and torch.cuda.is_available():
        model.cuda()

    if joint:
        joint_train_feat = joint_feats(event_train_feat, entity_train_feat)
        joint_dev_feat = joint_feats(event_dev_feat, entity_dev_feat)
        train_classifier(model, joint_train_feat, joint_dev_feat, learning_rate, iterations, bert, use_cuda)
    else:
        if run_type == 'event':
            train_classifier(model, event_train_feat, event_dev_feat, learning_rate, iterations, bert, use_cuda)
        elif run_type == 'entity':
            train_classifier(model, entity_train_feat, entity_dev_feat, learning_rate, iterations, bert, use_cuda)
        else:
            raise TypeError('No such run_type=' + run_type)

    if model_out:
        torch.save(model, model_out)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--trainFile', type=str, help='train file', required=True)
    # parser.add_argument('--testFile', type=str, help='test file', required=True)
    # parser.add_argument('--modelFile', type=str, help='model output location', required=False)
    # parser.add_argument('--bertFile', type=str, help='preprocessed_external_features glove file', required=False)
    # parser.add_argument('--lr', type=str, help='learning rate', required=True)
    # parser.add_argument('--iter', type=str, help='num of iterations', required=True)
    # parser.add_argument('--cuda', type=str, help='use cuda device', required=True)
    # args = parser.parse_args()

    EMBED_SIZE = 1024
    MODEL_SIZE = 1024

    _event_train_file = 'data/interim/kian/gold_mentions_with_context/ECB_Train_Event_gold_mentions.json'
    _event_dev_file = 'data/interim/kian/gold_mentions_with_context/ECB_Dev_Event_gold_mentions.json'
    _entity_train_file = 'data/interim/kian/gold_mentions_with_context/ECB_Train_Entity_gold_mentions.json'
    _entity_dev_file = 'data/interim/kian/gold_mentions_with_context/ECB_Dev_Entity_gold_mentions.json'
    _bert_file = 'dumps/embedded/ecb_all_embed_bert_all_layers.pickle'
    _model_out = 'models/mlp_test_model'

    _learning_rate = 0.01
    _iterations = 1
    _joint = True
    _type = 'entity'

    _use_cuda = False  # args.cuda in ['True', 'true', 'yes', 'Yes']
    if _use_cuda:
        print(torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(1)

    random.seed(1)
    np.random.seed(1)

    run_train(_event_train_file, _event_dev_file, _entity_train_file, _entity_dev_file,
              _bert_file, _learning_rate, _iterations, _model_out, _type, _joint, _use_cuda)
