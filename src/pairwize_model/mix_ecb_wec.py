import logging

import numpy as np
import random
import torch

from src import LIBRARY_ROOT
from src.pairwize_model import configuration
from src.pairwize_model.model import PairWiseModelKenton
from src.pairwize_model.train import train_pairwise
from src.utils.bert_utils import BertFromFile
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

    if configuration.fine_tune:
        logger.info("Loading model to fine tune-" + configuration.load_model_file)
        pairwize_model = torch.load(configuration.load_model_file)
    else:
        # pairwize_model = PairWiseModelKenton(20736, 150, 2)
        pairwize_model = PairWiseModelKenton(20736, configuration.hidden_n, 2, bert_utils, configuration.use_cuda)

    event_train_file_pos_wec = str(LIBRARY_ROOT) + "/resources/final_dataset/" + \
                           "WEC_Train_Event_gold_mentions_PosPairs.pickle"
    event_train_file_neg_wec = str(LIBRARY_ROOT) + "/resources/final_dataset/" + \
                           "WEC_Train_Event_gold_mentions_NegPairs.pickle"

    event_train_file_pos_ecb = str(LIBRARY_ROOT) + "/resources/final_dataset/" + \
                               "ECB_Train_Event_gold_mentions_PosPairs.pickle"
    event_train_file_neg_ecb = str(LIBRARY_ROOT) + "/resources/final_dataset/" + \
                               "ECB_Train_Event_gold_mentions_NegPairs.pickle"

    train_feat_wec = load_pos_neg_pickle(event_train_file_pos_wec, event_train_file_neg_wec, configuration.ratio)
    train_feat_ecb = load_pos_neg_pickle(event_train_file_pos_ecb, event_train_file_neg_ecb, configuration.ratio)
    train_feat = train_feat_wec
    train_feat.extend(train_feat_ecb)
    logger.info("Total Train feat WEC+ECB=" + str(len(train_feat)))
    random.shuffle(train_feat)

    validation_feat = load_pos_neg_pickle(configuration.event_validation_file_pos, configuration.event_validation_file_neg, -1)

    if configuration.use_cuda:
        # print(torch.cuda.get_device_name(1))
        torch.cuda.manual_seed(1)
        pairwize_model.cuda()

    return train_feat, validation_feat, bert_utils, pairwize_model


if __name__ == '__main__':
    log_params_str = "mix_ds" + configuration.train_dataset.name + "_dev_ds" + configuration.dev_dataset.name + \
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

    _event_train_feat, _event_validation_feat, _bert_utils, _pairwize_model = init_basic_training_resources()

    train_pairwise(_pairwize_model, _event_train_feat, _event_validation_feat, configuration.batch_size,
                   configuration.iterations, configuration.learning_rate , save_model=configuration.save_model,
                   model_out=configuration.save_model_file, best_model_to_save=configuration.save_model_threshold)
