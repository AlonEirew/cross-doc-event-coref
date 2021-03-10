import logging

import numpy as np
import random
import torch

from src.configuration import Configuration, ConfigType
from src.pairwize_model.model import PairWiseModelKenton
from src.utils.embed_utils import EmbedFromFile
from src.utils.eval_utils import get_confusion_matrix, get_prec_rec_f1
from src.utils.log_utils import create_logger_with_fh

logger = logging.getLogger(__name__)


def train_pairwise(pairwize_model, train, validation, batch_size, epochs=4,
                   lr=1e-5, save_model=False, model_out=None, best_model_to_save=0.1):
    # loss_func = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(pairwize_model.parameters(), lr, weight_decay=configuration.weight_decay)
    # optimizer = AdamW(pairwize_model.parameters(), lr)
    dataset_size = len(train)

    best_result_for_save = best_model_to_save
    improvement_seen = False
    non_improved_epoch_count = 0

    for epoch in range(epochs):
        pairwize_model.train()
        end_index = batch_size
        random.shuffle(train)

        cum_loss, count_btch = (0.0, 1)
        for start_index in range(0, dataset_size, batch_size):
            if end_index > dataset_size:
                end_index = dataset_size

            optimizer.zero_grad()

            batch_features = train[start_index:end_index].copy()
            bs = end_index - start_index
            prediction, gold_labels = pairwize_model(batch_features, bs)

            loss = loss_func(prediction, gold_labels.reshape(-1, 1).float())
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            end_index += batch_size
            count_btch += 1

            if count_btch % 100 == 0:
                report = "%d: %d: loss: %.10f:" % (epoch + 1, end_index, cum_loss / count_btch)
                logger.info(report)

        pairwize_model.eval()
        # accuracy_on_dataset("Train", epoch + 1, pairwize_model, train)
        _, _, _, dev_f1 = accuracy_on_dataset("Dev", epoch + 1, pairwize_model, validation)
        # accuracy_on_dataset(accum_count_btch / 10000, embed_utils, pairwize_model, test, use_cuda)
        pairwize_model.train()

        if best_result_for_save < dev_f1:
            if save_model:
                logger.info("Found better model saving")
                torch.save(pairwize_model, model_out + "iter_" + str(epoch + 1))
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


def accuracy_on_dataset(testset, epoch, pairwize_model, features, batch_size=10000):
    all_labels, all_predictions = run_inference(pairwize_model, features, batch_size=batch_size)
    accuracy = torch.mean((all_labels == all_predictions).float())
    tn, fp, fn, tp = get_confusion_matrix(all_labels, all_predictions)
    precision, recall, f1 = get_prec_rec_f1(tp, fp, fn)

    logger.info("%s: %d: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" % \
                (testset + "-Acc", epoch, accuracy.item(), precision, recall, f1))

    return accuracy, precision, recall, f1


def run_inference(pairwize_model, features, round_pred=True, batch_size=10000):
    dataset_size = len(features)
    end_index = batch_size
    labels = list()
    predictions = list()
    for start_index in range(0, dataset_size, batch_size):
        if end_index > dataset_size:
            end_index = dataset_size

        batch_features = features[start_index:end_index].copy()
        batch_size = end_index - start_index
        batch_predictions, batch_label = pairwize_model.predict(batch_features, batch_size)

        if round_pred:
            batch_predictions = torch.round(batch_predictions.reshape(-1)).long()

        predictions.append(batch_predictions.detach())
        labels.append(batch_label.detach())
        end_index += batch_size

    all_labels = torch.cat(labels).cpu()
    all_predictions = torch.cat(predictions).cpu()

    return all_labels, all_predictions


def init_basic_training_resources():
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    embed_utils = EmbedFromFile(configuration.embed_files, configuration.model_size)

    if configuration.fine_tune:
        logger.info("Loading model to fine tune-" + configuration.load_model_file)
        pairwize_model = torch.load(configuration.load_model_file)
        pairwize_model.set_embed_utils(embed_utils)
    else:
        # pairwize_model = PairWiseModelKenton(20736, 150, 2)
        pairwize_model = PairWiseModelKenton(9 * embed_utils.embed_size, configuration.hidden_n, 1,
                                             embed_utils, configuration.use_cuda)

    train_feat = configuration.train_dataset.load_pos_neg_pickle(configuration.event_train_file_pos,
                                                                 configuration.event_train_file_neg)
    validation_feat = configuration.dev_dataset.load_pos_neg_pickle(configuration.event_validation_file_pos,
                                                                    configuration.event_validation_file_neg)

    if configuration.use_cuda:
        # print(torch.cuda.get_device_name(1))
        pairwize_model.cuda()

    return train_feat, validation_feat, pairwize_model


if __name__ == '__main__':
    configuration = Configuration(ConfigType.Train)

    log_params_str = "train_ds" + configuration.train_dataset.name + "_dev_ds" + configuration.dev_dataset.name + \
                     "_lr" + str(configuration.learning_rate) + "_bs" + str(configuration.batch_size) + "_a" + \
                     str(configuration.ratio) + "_itr" + str(configuration.iterations)
    create_logger_with_fh(log_params_str)

    if configuration.save_model and configuration.fine_tune and \
            configuration.save_model_file == configuration.load_model_file:
        raise Exception('Fine Tune & Save model set with same model file for in & out')

    logger.info("train_set=" + configuration.train_dataset.name + ", dev_set=" + configuration.dev_dataset.name +
                ", lr=" + str(configuration.learning_rate) + ", bs=" + str(configuration.batch_size) +
                ", ratio=1:" + str(configuration.ratio) + ", itr=" + str(configuration.iterations) +
                ", hidden_n=" + str(configuration.hidden_n) + ", weight_decay=" + str(configuration.weight_decay))

    _event_train_feat, _event_validation_feat, _pairwize_model = init_basic_training_resources()

    train_pairwise(_pairwize_model, _event_train_feat, _event_validation_feat, configuration.batch_size,
                   configuration.iterations, configuration.learning_rate, save_model=configuration.save_model,
                   model_out=configuration.save_model_file, best_model_to_save=configuration.save_model_threshold)