import logging

import numpy as np
import random
import torch

from src import LIBRARY_ROOT, configuration
from src.pairwize_model.model import PairWiseModelKenton
from src.pairwize_model.train import train_pairwise
from src.utils.embed_utils import BertFromFile
from src.utils.log_utils import create_logger_with_fh

logger = logging.getLogger(__name__)


def init_basic_training_resources():
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    bert_files = [str(LIBRARY_ROOT) + "/resources/final_dataset/WEC_Train_Event_gold_mentions.pickle",
                  str(LIBRARY_ROOT) + "/resources/final_dataset/ECB_Train_Event_gold_mentions.pickle",
                  str(LIBRARY_ROOT) + "/resources/final_dataset/ECB_Dev_Event_gold_mentions.pickle"]

    bert_utils = BertFromFile(bert_files)
    # bert_utils = BertPretrainedUtils(-1, finetune=True, use_cuda=use_cuda, pad=True)

    if configuration.train_fine_tune:
        logger.info("Loading model to fine tune-" + configuration.train_load_model_file)
        pairwize_model = torch.load(configuration.train_load_model_file)
    else:
        # pairwize_model = PairWiseModelKenton(20736, 150, 2)
        pairwize_model = PairWiseModelKenton(20736, configuration.train_hidden_n, 2, bert_utils, configuration.use_cuda)

    event_train_file_pos_wec = str(LIBRARY_ROOT) + "/resources/final_dataset/" + \
                           "WEC_Train_Event_gold_mentions_PosPairs.pickle"
    event_train_file_neg_wec = str(LIBRARY_ROOT) + "/resources/final_dataset/" + \
                           "WEC_Train_Event_gold_mentions_NegPairs.pickle"

    event_train_file_pos_ecb = str(LIBRARY_ROOT) + "/resources/final_dataset/" + \
                               "ECB_Train_Event_gold_mentions_PosPairs.pickle"
    event_train_file_neg_ecb = str(LIBRARY_ROOT) + "/resources/final_dataset/" + \
                               "ECB_Train_Event_gold_mentions_NegPairs.pickle"

    train_feat_wec = load_pos_neg_pickle(event_train_file_pos_wec, event_train_file_neg_wec, configuration.train_ratio)
    train_feat_ecb = load_pos_neg_pickle(event_train_file_pos_ecb, event_train_file_neg_ecb, configuration.train_ratio)
    train_feat = train_feat_wec
    train_feat.extend(train_feat_ecb)
    logger.info("Total Train feat WEC+ECB=" + str(len(train_feat)))
    random.shuffle(train_feat)

    validation_feat = load_pos_neg_pickle(configuration.train_event_validation_file_pos, configuration.train_event_validation_file_neg, -1)

    if configuration.use_cuda:
        # print(torch.cuda.get_device_name(1))
        torch.cuda.manual_seed(1)
        pairwize_model.cuda()

    return train_feat, validation_feat, bert_utils, pairwize_model


if __name__ == '__main__':
    log_params_str = "mix_ds" + configuration.train_dataset.name + "_dev_ds" + configuration.dev_dataset.name + \
                     "_lr" + str(configuration.train_learning_rate) + "_bs" + str(configuration.train_batch_size) + "_a" + \
                     str(configuration.train_ratio) + "_itr" + str(configuration.train_iterations)
    create_logger_with_fh(log_params_str)

    if configuration.train_save_model and configuration.train_fine_tune and \
            configuration.train_save_model_file == configuration.train_load_model_file:
        raise Exception('Fine Tune & Save model set with same model file for in & out')

    logger.info("train_set=" + configuration.train_dataset.name + ", dev_set=" + configuration.dev_dataset.name +
                ", lr=" + str(configuration.train_learning_rate) + ", bs=" + str(configuration.train_batch_size) +
                ", ratio=1:" + str(configuration.train_ratio) + ", itr=" + str(configuration.train_iterations) +
                ", hidden_n=" + str(configuration.train_hidden_n) + ", weight_decay=" + str(configuration.train_weight_decay))

    _event_train_feat, _event_validation_feat, _bert_utils, _pairwize_model = init_basic_training_resources()

    train_pairwise(_pairwize_model, _event_train_feat, _event_validation_feat, configuration.train_batch_size,
                   configuration.train_iterations, configuration.train_learning_rate, save_model=configuration.train_save_model,
                   model_out=configuration.train_save_model_file, best_model_to_save=configuration.train_save_model_threshold)
