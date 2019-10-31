import logging
import random
import time

import numpy as np
import torch
from torch import optim

from src import LIBRARY_ROOT
from src.dl_model.snli_coref_model import SnliCorefModel
from src.ext_resources.embedding.embed_elmo import ElmoEmbedding, ElmoEmbeddingOffline
from src.utils.dl_utils import get_feat

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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

            if total_count % 100 == 0:
                batch1_end = int(round(time.time() * 1000))
                batch1_took = batch1_end - batch1_start
                log(epoch, total_count, pair_count, cum_loss, batch1_took, 0.0)

                accuracy_dev = accuracy_on_dataset(model, dev_feats, embed, use_cuda)
                end = int(round(time.time() * 1000))
                took = end - start
                logger.info('DEV Accuracy at END:')
                log(epoch, total_count, pair_count, cum_loss, took, accuracy_dev)

        accuracy_train = accuracy_on_dataset(model, train_feats, embed, use_cuda)
        end = int(round(time.time() * 1000))
        took = end - start
        logger.info('TRAIN Accuracy at END:')
        log(epoch, total_count, pair_count, cum_loss, took, accuracy_train)


def feat_to_vec(batch_features, embed, use_cuda):
    ret_sents1 = list()
    ret_ments1 = list()
    ret_sents2 = list()
    ret_ments2 = list()
    ret_golds = list()
    longest_context = 0
    longest_mention = 0
    for mention1, mention2 in batch_features:
        sentence1_words = ' '.join(mention1.mention_context)
        sentence2_words = ' '.join(mention2.mention_context)
        context1_full_vec = None
        context2_full_vec = None
        mention1_vec = None
        mention2_vec = None
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

        ret_golds.append(gold_label)
        ret_sents1.append(np.vstack(context1_full_vec))
        ret_sents2.append(np.vstack(context2_full_vec))
        ret_ments1.append(np.vstack(mention1_vec))
        ret_ments2.append(np.vstack(mention2_vec))

        if len(context2_full_vec) < len(context1_full_vec):
            max_context = len(context1_full_vec)
        else:
            max_context = len(context2_full_vec)

        if len(mention2_vec) < len(mention1_vec):
            max_mention = len(mention1_vec)
        else:
            max_mention = len(mention2_vec)

        if max_context > longest_context:
            longest_context = max_context

        if max_mention > longest_mention:
            longest_mention = max_mention

    padding_vector_list(ret_sents1, longest_context)
    padding_vector_list(ret_sents2, longest_context)
    padding_vector_list(ret_ments1, longest_mention)
    padding_vector_list(ret_ments2, longest_mention)

    ret_golds = torch.tensor(ret_golds)
    ret_sents1 = torch.from_numpy(np.vstack((ret_sents1, )))
    ret_sents2 = torch.from_numpy(np.vstack((ret_sents2, )))
    ret_ments1 = torch.from_numpy(np.vstack((ret_ments1, )))
    ret_ments2 = torch.from_numpy(np.vstack((ret_ments2, )))

    if use_cuda:
        ret_sents1 = ret_sents1.cuda()
        ret_sents2 = ret_sents2.cuda()
        ret_ments1 = ret_ments1.cuda()
        ret_ments2 = ret_ments2.cuda()
        ret_golds = ret_golds.cuda()

    return ret_sents1, ret_ments1, ret_sents2, ret_ments2, ret_golds


def padding_vector_list(vector_list, vector_length):
    for i in range(len(vector_list)):
        if len(vector_list[i]) < vector_length:
            vector_list[i] = np.pad(vector_list[i], ((0, vector_length - len(vector_list[i])), (0, 0)), 'constant')


def log(epoch, total_count, pair_count, cum_loss, took, accuracy):
    if accuracy != 0.0:
        logger.info('%d: %d: loss: %.10f: Accuracy: %.10f: epoch-took: %dmilli' %
              (epoch + 1, total_count, cum_loss / pair_count, accuracy, took))
    else:
        logger.info('%d: %d: loss: %.10f: batch-took: %dmilli' %
              (epoch + 1, total_count, cum_loss / pair_count, took))


def accuracy_on_dataset(model, features, embedd, use_cuda):
    dataset_size = len(features)
    batch_size = 200
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

    return torch.mean((torch.cat(labels) == torch.cat(predictions)).float())


def run_train(train_file, dev_file, bert_file, learning_rate, iterations, model_out, use_cuda):
    logger.info('Create Features:')
    train_feat = get_feat(train_file)
    dev_feat = get_feat(dev_file)

    bert = ElmoEmbeddingOffline(bert_file)
    model = SnliCorefModel(MODEL_SIZE)

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

    _event_train_file = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Dev_Event_gold_mentions.json'
    _event_dev_file = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Dev_Event_gold_mentions.json'

    _bert_file = str(LIBRARY_ROOT) + '/resources/preprocessed_external_features/embedded/wiki_all_embed_bert_all_layers.pickle'
    _model_out = str(LIBRARY_ROOT) + '/saved_models/wiki_trained_model'

    _learning_rate = 0.01
    _iterations = 1
    _joint = False
    _type = 'event'

    _use_cuda = False  # args.cuda in ['True', 'true', 'yes', 'Yes']
    if _use_cuda:
        logger.info(torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(1)

    random.seed(1)
    np.random.seed(1)

    run_train(_event_train_file, _event_dev_file, _bert_file, _learning_rate, _iterations, _model_out, _use_cuda)
