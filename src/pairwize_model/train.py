import logging

import numpy as np
import random
import torch

from src import LIBRARY_ROOT
from src.pairwize_model.model import PairWiseModel
from src.utils.bert_utils import BertFromFile
from src.utils.dataset_utils import SPLIT, DATASET, load_datasets
from src.utils.log_utils import create_logger_with_fh

logger = logging.getLogger(__name__)


def train_pairwise(bert_utils, pairwize_model, train, validation, batch_size, epochs=4, lr=1e-5, use_cuda=True, report_fs=None):
    loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(pairwize_model.parameters(), lr, weight_decay=0.01)
    dataset_size = len(train)

    for epoch in range(epochs): #, desc="Epoch"
        pairwize_model.train()
        end_index = batch_size
        random.shuffle(train)

        # DELETE
        # dev_accuracy, dev_precision, dev_recall, dev_f1 = accuracy_on_dataset(bert_utils, pairwize_model, validation, use_cuda)
        # dev_report = "%s: %d: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" % \
        #              ("Dev-Acc", epoch + 1, dev_accuracy.item(), dev_precision, dev_recall, dev_f1)
        # logger.info(dev_report)
        #
        # train_accuracy, train_precision, train_recall, train_f1 = accuracy_on_dataset(bert_utils, pairwize_model, train, use_cuda)
        # train_report = "%s: %d: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" % \
        #                ("Train-Acc", epoch + 1, train_accuracy.item(), train_precision, train_recall, train_f1)
        # logger.info(train_report)
        # DELETE

        cum_loss, count_btch = (0.0, 0)
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

            if count_btch % 100 == 0:
                report = "%d: %d: loss: %.10f:" % (epoch + 1, end_index, cum_loss / count_btch)
                logger.info(report)
                if report_fs is not None:
                    report_fs.write(report + "\n")

            if count_btch % 10000 == 0:
                pairwize_model.eval()
                dev_accuracy, dev_precision, dev_recall, dev_f1 = accuracy_on_dataset(bert_utils, pairwize_model, validation, use_cuda)
                dev_report = "%s: %d: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" % \
                             ("Dev-Acc", epoch + 1, dev_accuracy.item(), dev_precision, dev_recall, dev_f1)
                logger.info(dev_report)
                pairwize_model.train()

        pairwize_model.eval()
        dev_accuracy, dev_precision, dev_recall, dev_f1 = accuracy_on_dataset(bert_utils, pairwize_model, validation, use_cuda)
        dev_report = "%s: %d: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" % \
                     ("Dev-Acc", epoch + 1, dev_accuracy.item(), dev_precision, dev_recall, dev_f1)
        logger.info(dev_report)

        train_accuracy, train_precision, train_recall, train_f1 = accuracy_on_dataset(bert_utils, pairwize_model, train, use_cuda)
        train_report = "%s: %d: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" % \
                       ("Train-Acc", epoch + 1, train_accuracy.item(), train_precision, train_recall, train_f1)
        logger.info(train_report)

        if report_fs is not None:
            report_fs.write(dev_report + "\n")
            report_fs.write(train_report + "\n")


def accuracy_on_dataset(bert_utils, pairwize_model, features, use_cuda):
    dataset_size = len(features)
    batch_size = 10000
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

    all_labels = torch.cat(labels).bool()
    all_predictions = torch.cat(predictions).bool()
    return get_measurements(all_labels, all_predictions)


def get_measurements(all_labels, all_predictions):
    accuracy = torch.mean((all_labels == all_predictions).float())

    tp = torch.sum(all_labels & all_predictions).float().item()
    # tn = torch.sum(~all_labels & ~all_predictions)
    fn = torch.sum(all_labels & ~all_predictions).float().item()
    fp = torch.sum(~all_labels & all_predictions).float().item()
    tpfp = tp + fp
    tpfn = tp + fn
    precision, recall, f1 = (0.0, 0.0, 0.0)
    if tpfp != 0:
        precision = tp / tpfp
    if tpfn != 0:
        recall = tp / tpfn
    if precision != 0 or recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1


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


def run_experiment(train_file, valid_file, test_file, bert_utils, pairwize_model, batch_size, iterations,
                   lr, alpha, dataset_type, use_cuda, save_model=False, model_out=None, report_fs=None):

    train_feat = load_datasets(train_file, alpha, SPLIT.TRAIN, dataset_type)
    validation_feat = load_datasets(valid_file, alpha, SPLIT.VALIDATION, DATASET.ECB)

    train_pairwise(bert_utils, pairwize_model, train_feat, validation_feat, batch_size, iterations, lr, use_cuda, report_fs)

    if save_model:
        print("Saving the model to-" + model_out)
        torch.save(_pairwize_model, model_out)


def init_basic_training_resources(context_set, dataset, use_cuda, fine_tune=False, model_in=None):
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    bert_files = [str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + dataset.name + "_Train_Event_gold_mentions.pickle",
                  str(LIBRARY_ROOT) + "/resources/" + context_set + "/ECB_Dev_Event_gold_mentions.pickle",
                  ]

    bert_utils = BertFromFile(bert_files)

    if fine_tune:
        logger.info("Loading model to fine tune-" + model_in)
        pairwize_model = torch.load(model_in)
    else:
        pairwize_model = PairWiseModel(2304, 250, 2)

    event_train_file = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + dataset.name + "_Train_Event_gold_mentions.json"
    event_validation_file = str(LIBRARY_ROOT) + "/resources/" + context_set + "/ECB_Dev_Event_gold_mentions.json"
    event_test_file = None

    if use_cuda:
        # print(torch.cuda.get_device_name(1))
        torch.cuda.manual_seed(1)
        pairwize_model.cuda()

    return event_train_file, event_validation_file, event_test_file, bert_utils, pairwize_model


if __name__ == '__main__':
    _dataset = DATASET.ECB
    _context_set = "single_sent_full_context_mean"

    _lr = 1e-6
    _batch_size = 32
    _alpha = 10
    _iterations = 20
    _use_cuda = True
    _save_model = False
    _fine_tune = True

    log_params_str = "train_ds" + _dataset.name + "_lr" + str(_lr) + "_bs" + str(_batch_size) + "_a" + str(_alpha) + "_itr" + str(_iterations)
    create_logger_with_fh(log_params_str)

    _model_out = str(LIBRARY_ROOT) + "/saved_models/" + _dataset.name + "_trained_model_" + str(_alpha)
    _model_in = str(LIBRARY_ROOT) + "/saved_models/WEC_trained_model_1"

    if _save_model and _fine_tune and _model_out == _model_in:
        raise Exception('Fine Tune & Save model set with same model file for in & out')

    _event_train_file, _event_validation_file, _event_test_file, _bert_utils, _pairwize_model = \
        init_basic_training_resources(_context_set, _dataset, _use_cuda, fine_tune=_fine_tune, model_in=_model_in)

    run_experiment(_event_train_file, _event_validation_file, None, _bert_utils, _pairwize_model, _batch_size, _iterations, _lr
                   , _alpha, _dataset, _use_cuda, save_model=_save_model, model_out=_model_out)
