import datetime
import logging

import numpy as np
import random
import torch

from src import LIBRARY_ROOT
from src.pairwize_model.model import PairWiseModel
from src.utils.bert_utils import BertFromFile
from src.utils.dataset_utils import SPLIT, load_datasets, DATASET
from src.utils.log_utils import create_logger


def train_pairwise(bert_utils, pairwize_model, train, validation, batch_size, epochs=4, lr=1e-5, use_cuda=True):
    loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(pairwize_model.parameters(), lr)
    dataset_size = len(train)

    for epoch in range(epochs): #, desc="Epoch"
        pairwize_model.train()
        end_index = batch_size
        random.shuffle(train)

        cum_loss, count_btch = (0.0, 0.0)
        for start_index in range(0, dataset_size, batch_size): #, desc="Batches"
            if end_index > dataset_size:
                end_index = dataset_size

            optimizer.zero_grad()

            batch_features = train[start_index:end_index].copy()
            embeded_features, gold_labels = get_bert_rep(batch_features, bert_utils, use_cuda)
            output = pairwize_model(embeded_features)

            loss = loss_func(output, gold_labels)
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            end_index += batch_size
            count_btch += 1

            logger.info("%d: %d: loss: %.10f:" % (epoch + 1, end_index, cum_loss / count_btch))

        pairwize_model.eval()
        dev_accuracy = accuracy_on_dataset(bert_utils, pairwize_model, validation, use_cuda)
        logger.info("%s: %d: Accuracy: %.10f" %
                    ("Dev-Acc", epoch + 1, dev_accuracy.item()))

        train_accuracy = accuracy_on_dataset(bert_utils, pairwize_model, train, use_cuda)
        logger.info("%s: %d: Accuracy: %.10f" %
                    ("Train-Acc", epoch + 1, train_accuracy.item()))


def accuracy_on_dataset(bert_utils, pairwize_model, features, use_cuda):
    dataset_size = len(features)
    batch_size = 1000
    end_index = batch_size
    labels = list()
    predictions = list()
    for start_index in range(0, dataset_size, batch_size):
        if end_index > dataset_size:
            end_index = dataset_size

        batch_features = features[start_index:end_index].copy()
        batch, batch_label = get_bert_rep(batch_features, bert_utils, use_cuda)
        batch_predictions = pairwize_model.predict(batch)

        predictions.append(batch_predictions)
        labels.append(batch_label)
        end_index += batch_size

    return torch.mean((torch.cat(labels) == torch.cat(predictions)).float())


def get_bert_rep(batch_features, bert_utils, use_cuda):
    batch_result = list()
    batch_labels = list()
    for mention1, mention2 in batch_features:
        # (1, 768), (1, 169)
        hidden1 = bert_utils.get_mention_mean_rep(mention1)
        if type(hidden1) == tuple:
            hidden1, _ = hidden1
        # (1, 768)
        hidden2 = bert_utils.get_mention_mean_rep(mention2)
        if type(hidden2) == tuple:
            hidden2, _ = hidden2

        # (1, 768), (1,169)
        span1_span2 = hidden1 * hidden2

        # 768 * 2 + 1 = (1, 2304)
        concat_result = torch.cat((hidden1.reshape(-1), hidden2.reshape(-1), span1_span2.reshape(-1))).reshape(1, -1)
        gold_label = 1 if mention1.coref_chain == mention2.coref_chain else 0

        batch_result.append(concat_result)
        batch_labels.append(gold_label)

    ret_result = torch.cat(batch_result)
    ret_golds = torch.tensor(batch_labels)
    
    if use_cuda:
        ret_result = ret_result.cuda()
        ret_golds = ret_golds.cuda()

    return ret_result, ret_golds


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dataset = DATASET.WEC
    context_set = "single_sent_full_context"

    _lr = 1e-7
    _batch_size = 32
    _alpha = 1
    _iterations = 1
    _use_cuda = True  # args.cuda in ["True", "true", "yes", "Yes"]
    save_model = True

    running_timestamp = "train_" + str(datetime.datetime.now().time().strftime("%H%M%S%m%d%Y"))
    params_str = "_lr" + str(_lr) + "_bs" + str(_batch_size) + "_a" + str(_alpha) + "_itr" + str(_iterations)
    log_file = str(LIBRARY_ROOT) + "/logging/" + running_timestamp + "_" + params_str + ".log"
    logger = create_logger(__name__, log_file)

    _event_train_file = str(LIBRARY_ROOT) + "/resources/corpora/" + context_set + "/" + dataset.name + "_Train_Event_gold_mentions.json"
    _event_validation_file = str(LIBRARY_ROOT) + "/resources/corpora/bkp_single_sent/ECB_Dev_Event_gold_mentions.json"

    _model_out = str(LIBRARY_ROOT) + "/saved_models/" + dataset.name +"_trained_model"

    bert_files = [str(LIBRARY_ROOT) + "/resources/corpora/" + context_set + "/" + dataset.name +"_Train_Event_gold_mentions.pickle",
                  # str(LIBRARY_ROOT) + "/resources/corpora/" + context_set + "/WEC_Dev_Event_gold_mentions.pickle",
                  str(LIBRARY_ROOT) + "/resources/corpora/" + context_set + "/ECB_Test_Event_gold_mentions.pickle",
                  str(LIBRARY_ROOT) + "/resources/corpora/" + context_set + "/ECB_Dev_Event_gold_mentions.pickle"
                  ]

    # _bert_utils = BertPretrainedUtils()
    _bert_utils = BertFromFile(bert_files)
    _pairwize_model = PairWiseModel(2304, 250, 2)

    if _use_cuda:
        logger.info(torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(0)
        _pairwize_model.cuda()

    _train = load_datasets(_event_train_file, _alpha, SPLIT.TRAIN, dataset)
    _validation = load_datasets(_event_validation_file, _alpha, SPLIT.VALIDATION, DATASET.ECB)

    train_pairwise(_bert_utils, _pairwize_model, _train, _validation, _batch_size, _iterations, _lr, _use_cuda)

    if save_model:
        print("Saving the model to-" +_model_out)
        torch.save(_pairwize_model, _model_out)
