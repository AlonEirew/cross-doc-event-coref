import random
import time

import numpy as np
import torch
from torch import optim

from src import LIBRARY_ROOT
from src.ext_resources.embedding.embed_elmo import ElmoEmbedding, ElmoEmbeddingOffline
from src.dl_model.coref_model import CorefModel
from src.obj.topics import Topics


def train_classifier(model, train_feats, dev_feats, learning_rate, iterations, embed, use_cuda):
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.05)
    # optimizer = optim.SGD(dl_model.parameters(), lr=learning_rate)

    total_count = 0
    for epoch in range(iterations):
        cum_loss = 0.0
        start = int(round(time.time() * 1000))
        pair_count = 0
        dataset_size = len(train_feats)
        batch_size = 50
        end_index = batch_size
        for start_index in range(0, dataset_size, batch_size):
            if end_index > dataset_size:
                end_index = dataset_size

            batch_features = train_feats[start_index:end_index].copy()
            pair_count += 1
            batch1_start = int(round(time.time() * 1000))
            sent1_feat, mention1_feat, sent2_feat, mention2_feat, true_label = feat_to_vec(batch_features, embed, use_cuda)

            optimizer.zero_grad()
            output = model(sent1_feat, mention1_feat, sent2_feat, mention2_feat)
            loss = model.loss_fun(output, true_label)
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
            total_count += batch_size
            end_index += batch_size

            if total_count % 500 == 0:
                batch1_end = int(round(time.time() * 1000))
                batch1_took = batch1_end - batch1_start
                log(epoch, total_count, pair_count, cum_loss, batch1_took, 0.0)

        accuracy_dev = accuracy_on_dataset(model, dev_feats, embed, use_cuda)
        end = int(round(time.time() * 1000))
        took = end - start
        print('DEV Accuracy at END:')
        log(epoch, total_count, pair_count, cum_loss, took, accuracy_dev)

        accuracy_train = accuracy_on_dataset(model, train_feats, embed, use_cuda)
        end = int(round(time.time() * 1000))
        took = end - start
        print('TRAIN Accuracy at END:')
        log(epoch, total_count, pair_count, cum_loss, took, accuracy_train)


def feat_to_vec(batch_features, embed, use_cuda):
    ret_sents1 = list()
    ret_ments1 = list()
    ret_sents2 = list()
    ret_ments2 = list()
    ret_golds = list()
    longest_array_context = 0
    for mention1, mention2 in batch_features:
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

        if context1_full_vec is None or context2_full_vec is None:
            continue

        if len(context2_full_vec) < len(context1_full_vec):
            max_context = len(context1_full_vec)
        else:
            max_context = len(context2_full_vec)

        if max_context > longest_array_context:
            longest_array_context = max_context

        ret_golds.append(gold_label)
        ret_sents1.append(np.vstack(context1_full_vec))
        ret_sents2.append(np.vstack(context2_full_vec))
        ret_ments1.append(mention1_vec)
        ret_ments2.append(mention2_vec)

    for i in range(len(ret_sents1)):
        if len(ret_sents1[i]) < longest_array_context:
            ret_sents1[i] = np.pad(ret_sents1[i], ((0, longest_array_context - len(ret_sents1[i])), (0, 0)), 'constant')

    for i in range(len(ret_sents2)):
        if len(ret_sents2[i]) < longest_array_context:
            ret_sents2[i] = np.pad(ret_sents2[i], ((0, longest_array_context - len(ret_sents2[i])), (0, 0)), 'constant')

    ret_golds = torch.tensor(ret_golds)
    ret_sents1 = torch.from_numpy(np.vstack((ret_sents1, )))
    ret_sents2 = torch.from_numpy(np.vstack((ret_sents2, )))
    ret_ments1 = torch.from_numpy(np.vstack(ret_ments1))
    ret_ments2 = torch.from_numpy(np.vstack(ret_ments2))

    if use_cuda:
        ret_sents1 = ret_sents1.cuda()
        ret_sents2 = ret_sents2.cuda()
        ret_ments1 = ret_ments1.cuda()
        ret_ments2 = ret_ments2.cuda()
        ret_golds = ret_golds.cuda()

    return ret_sents1, ret_ments1, ret_sents2, ret_ments2, ret_golds


def log(epoch, total_count, pair_count, cum_loss, took, accuracy):
    if accuracy != 0.0:
        print('%d: %d: loss: %.10f: Accuracy: %.10f: epoch-took: %dmilli' %
              (epoch + 1, total_count, cum_loss / pair_count, accuracy, took))
    else:
        print('%d: %d: loss: %.10f: batch-took: %dmilli' %
              (epoch + 1, total_count, cum_loss / pair_count, took))


def accuracy_on_dataset(model, features, embedd, use_cuda):
    dataset_size = len(features)
    batch_size = 1000
    end_index = batch_size
    labels = list()
    predictions = list()
    for start_index in range(0, dataset_size, batch_size):
        if end_index > dataset_size:
            end_index = dataset_size

        batch_features = features[start_index:end_index].copy()
        sent1_feat, mention1_feat, sent2_feat, mention2_feat, batch_label = feat_to_vec(batch_features, embedd, use_cuda)
        batch_predictions = model.predict(sent1_feat, mention1_feat, sent2_feat, mention2_feat)
        predictions.append(batch_predictions)
        labels.append(batch_label)
        end_index += batch_size

    labels = torch.cat(labels)
    predictions = torch.cat(predictions)
    return torch.mean((labels == predictions).float())


def create_features_from_pos_neg(positive_exps, negative_exps):
    feats = list()
    feats.extend(positive_exps)
    feats.extend(negative_exps)
    # feats.extend(random.sample(negative_exps, len(positive_exps) * 2))
    random.shuffle(feats)
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


def get_feat(train_file, dev_file):
    topics_train = Topics()
    topics_train.create_from_file(train_file, keep_order=True)
    topics_dev = Topics()
    topics_dev.create_from_file(dev_file, keep_order=True)

    print('Create Train pos/neg examples')
    positive_train, negative_train = create_pos_neg_pairs(topics_train)
    train_feats = create_features_from_pos_neg(positive_train, negative_train)
    # print('Total Train examples-' + str(len(train_feats)))

    print('Create Test pos/neg examples')
    positive_dev, negative_dev = create_pos_neg_pairs(topics_dev)
    dev_feats = create_features_from_pos_neg(positive_dev, negative_dev)
    # print('Total Dev examples-' + str(len(dev_feats)))

    return train_feats, dev_feats


def run_train(train_file, dev_file, bert_file, learning_rate, iterations, model_out, use_cuda):

    print('Create Features:')
    train_feat, dev_feat = get_feat(train_file, dev_file)

    bert = ElmoEmbeddingOffline(bert_file)
    model = CorefModel(MODEL_SIZE, MODEL_SIZE, MODEL_SIZE)

    if use_cuda and torch.cuda.is_available():
        model.cuda()

    train_classifier(model, train_feat, dev_feat, learning_rate, iterations, bert, use_cuda)

    if model_out:
        torch.save(model, model_out)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--trainFile', type=str, help='train file', required=True)
    # parser.add_argument('--testFile', type=str, help='test file', required=True)
    # parser.add_argument('--modelFile', type=str, help='dl_model output location', required=False)
    # parser.add_argument('--bertFile', type=str, help='preprocessed_external_features glove file', required=False)
    # parser.add_argument('--lr', type=str, help='learning rate', required=True)
    # parser.add_argument('--iter', type=str, help='num of iterations', required=True)
    # parser.add_argument('--cuda', type=str, help='use cuda device', required=True)
    # args = parser.parse_args()

    EMBED_SIZE = 1024
    MODEL_SIZE = 1024

    _event_train_file = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Train_Event_gold_mentions.json'
    _event_dev_file = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Dev_Event_gold_mentions.json'

    _bert_file = str(LIBRARY_ROOT) + '/resources/preprocessed_external_features/embedded/wiki_all_embed_bert_all_layers.pickle'
    _model_out = str(LIBRARY_ROOT) + '/saved_models/wiki_trained_model'

    _learning_rate = 0.01
    _iterations = 1
    _joint = False
    _type = 'event'

    _use_cuda = False  # args.cuda in ['True', 'true', 'yes', 'Yes']
    if _use_cuda:
        print(torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(1)

    random.seed(1)
    np.random.seed(1)

    run_train(_event_train_file, _event_dev_file, _bert_file, _learning_rate, _iterations, _model_out, _use_cuda)
