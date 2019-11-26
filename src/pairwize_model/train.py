import logging

import numpy as np
import random
import torch

from src import LIBRARY_ROOT
from src.pairwize_model.model import PairWiseModel, PairWiseModelKenton
from src.utils.bert_utils import BertFromFile
from src.utils.dataset_utils import SPLIT, DATASET, load_datasets, load_pos_neg_pickle
from src.utils.log_utils import create_logger_with_fh

logger = logging.getLogger(__name__)


def train_pairwise(bert_utils, pairwize_model, train, validation, batch_size, epochs=4,
                   lr=1e-5, use_cuda=True, save_model=False, model_out=None, best_model_to_save=0.1):
    loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = torch.nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(pairwize_model.parameters(), lr, weight_decay=0.01)
    optimizer = torch.optim.Adam(pairwize_model.parameters(), lr, weight_decay=0.01)
    dataset_size = len(train)

    INTERVAL = 10000
    accum_count_btch = 0
    best_result_for_save = best_model_to_save
    improvement_seen = False
    non_improved_epoch_count = 0
    for epoch in range(epochs): #, desc="Epoch"
        pairwize_model.train()
        end_index = batch_size
        random.shuffle(train)

        cum_loss, count_btch = (0.0, 0)
        for start_index in range(0, dataset_size, batch_size): #, desc="Batches"
            if end_index > dataset_size:
                end_index = dataset_size

            optimizer.zero_grad()

            batch_features = train[start_index:end_index].copy()
            bs = end_index - start_index
            embeded_features, gold_labels = pairwize_model.get_bert_rep(batch_features, bert_utils, use_cuda, bs)
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

            if count_btch % INTERVAL == 0:
                accum_count_btch += 1
                pairwize_model.eval()
                # accuracy_on_dataset("Train", epoch + 1, bert_utils, pairwize_model, train, use_cuda)
                _, _, _, dev_f1 = accuracy_on_dataset("Dev", accum_count_btch, bert_utils, pairwize_model, validation, use_cuda)
                # accuracy_on_dataset(accum_count_btch / 10000, bert_utils, pairwize_model, test, use_cuda)
                pairwize_model.train()

                if best_result_for_save < dev_f1:
                    if save_model:
                        logger.info("Found better model saving")
                        torch.save(pairwize_model, model_out)
                        improvement_seen = True
                        best_result_for_save = dev_f1
                        non_improved_epoch_count = 0
                elif improvement_seen:
                    if non_improved_epoch_count == 10:
                        logger.info("No Improvement for 10 ephochs, ending test...")
                        return best_result_for_save
                    else:
                        non_improved_epoch_count += 1

        pairwize_model.eval()
        # accuracy_on_dataset("Train", epoch + 1, bert_utils, pairwize_model, train, use_cuda)
        _, _, _, dev_f1 = accuracy_on_dataset("Dev", epoch + 1, bert_utils, pairwize_model, validation, use_cuda)
        # accuracy_on_dataset(accum_count_btch / 10000, bert_utils, pairwize_model, test, use_cuda)
        pairwize_model.train()

        if best_result_for_save < dev_f1:
            if save_model:
                logger.info("Found better model saving")
                torch.save(pairwize_model, model_out)
                best_result_for_save = dev_f1
                non_improved_epoch_count = 0
                improvement_seen = True
        elif improvement_seen:
            if non_improved_epoch_count == 10:
                logger.info("No Improvement for 10 ephochs, ending test...")
                break
            else:
                non_improved_epoch_count += 1

    return best_result_for_save


def accuracy_on_dataset(testset, epoch, bert_utils, pairwize_model, features, use_cuda):
    dataset_size = len(features)
    batch_size = 10000
    end_index = batch_size
    labels = list()
    predictions = list()
    for start_index in range(0, dataset_size, batch_size):
        if end_index > dataset_size:
            end_index = dataset_size

        batch_features = features[start_index:end_index].copy()
        batch_size = end_index - start_index
        batch, batch_label = pairwize_model.get_bert_rep(batch_features, bert_utils, use_cuda, batch_size)
        batch_predictions = pairwize_model.predict(batch)

        predictions.append(batch_predictions)
        labels.append(batch_label)
        end_index += batch_size

    all_labels = torch.cat(labels).bool()
    all_predictions = torch.cat(predictions).bool()

    measurements = get_measurements(testset, epoch, all_labels, all_predictions)
    return measurements


def get_measurements(testset, epoch, all_labels, all_predictions):
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

    logger.info("%s: %d: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" % \
                (testset + "-Acc", epoch, accuracy.item(), precision, recall, f1))

    return accuracy, precision, recall, f1


def init_basic_training_resources(context_set, train_dataset, dev_dataset, alpha,
                                  use_cuda=True, fine_tune=False, model_in=None):
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    bert_files = [str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + train_dataset.name + "_Train_Event_gold_mentions.pickle",
                  str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + dev_dataset.name + "_Dev_Event_gold_mentions.pickle"
                  ]

    bert_utils = BertFromFile(bert_files)

    if fine_tune:
        logger.info("Loading model to fine tune-" + model_in)
        pairwize_model = torch.load(model_in)
    else:
        pairwize_model = PairWiseModelKenton(20736, 150, 2)

    event_train_file_pos = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + train_dataset.name + "_Train_Event_gold_mentions_PosPairs.pickle"
    event_train_file_neg = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + train_dataset.name + "_Train_Event_gold_mentions_NegPairs.pickle"
    event_validation_file_pos = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + dev_dataset.name + "_Dev_Event_gold_mentions_PosPairs.pickle"
    event_validation_file_neg = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + dev_dataset.name + "_Dev_Event_gold_mentions_NegPairs.pickle"

    train_feat = load_pos_neg_pickle(event_train_file_pos, event_train_file_neg, alpha)

    if dev_dataset == DATASET.ECB:
        validation_feat = load_pos_neg_pickle(event_validation_file_pos, event_validation_file_neg, -1)
    else:
        validation_feat = load_pos_neg_pickle(event_validation_file_pos, event_validation_file_neg, 16)

    if use_cuda:
        # print(torch.cuda.get_device_name(1))
        torch.cuda.manual_seed(1)
        pairwize_model.cuda()

    return train_feat, validation_feat, bert_utils, pairwize_model


if __name__ == '__main__':
    _train_dataset = DATASET.ECB
    _dev_dataset = DATASET.ECB
    _context_set = "single_sent_clean_kenton"

    _lr = 1e-7
    _batch_size = 32
    _alpha = 7
    _iterations = 5
    _use_cuda = True
    _save_model = False
    _fine_tune = False

    log_params_str = "train_ds" + _train_dataset.name + "_lr" + str(_lr) + "_bs" + str(_batch_size) + "_a" + \
                     str(_alpha) + "_itr" + str(_iterations)
    create_logger_with_fh(log_params_str)

    _model_out = str(LIBRARY_ROOT) + "/saved_models/" + _train_dataset.name + "_" + _dev_dataset.name + "_best_trained_model"
    _model_in = str(LIBRARY_ROOT) + "/saved_models/WEC_trained_model_1"

    if _save_model and _fine_tune and _model_out == _model_in:
        raise Exception('Fine Tune & Save model set with same model file for in & out')

    logger.info("train_set=" + _train_dataset.name + ", dev_set=" + _dev_dataset.name + ", lr=" + str(_lr) + ", bs=" +
                str(_batch_size) + ", ratio=1:" + str(_alpha) + ", itr=" + str(_iterations))

    _event_train_feat, _event_validation_feat, _bert_utils, _pairwize_model = \
        init_basic_training_resources(_context_set, _train_dataset, _dev_dataset, _alpha, _use_cuda, fine_tune=_fine_tune, model_in=_model_in)

    train_pairwise(_bert_utils, _pairwize_model, _event_train_feat, _event_validation_feat, _batch_size, _iterations, _lr
                   , _use_cuda, save_model=_save_model, model_out=_model_out, best_model_to_save=0.3)
