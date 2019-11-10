import datetime

import torch

from src import LIBRARY_ROOT
from src.pairwize_model.train import accuracy_on_dataset
from src.utils.bert_utils import BertFromFile
from src.utils.dataset_utils import SPLIT, DATASET, get_feat, create_features_from_pos_neg
from src.utils.log_utils import create_logger

if __name__ == '__main__':
    dataset = DATASET.ECB
    context_set = "single_sent_full_context"

    running_timestamp = "inference_" + str(datetime.datetime.now().time().strftime("%H%M%S%m%d%Y"))
    log_file = str(LIBRARY_ROOT) + "/logging/" + running_timestamp + "_" + dataset.name + ".log"
    logger = create_logger(__name__, log_file)

    _event_test_file = str(LIBRARY_ROOT) + "/resources/corpora/" + context_set + "/ECB_Test_Event_gold_mentions.json"
    positive_, negative_ = get_feat(_event_test_file, -1, SPLIT.TEST, DATASET.ECB)
    _model_out = str(LIBRARY_ROOT) + "/saved_models/" + dataset.name + "_trained_model"
    _bert_utils = BertFromFile([str(LIBRARY_ROOT) + "/resources/corpora/" + context_set + "/ECB_Test_Event_gold_mentions.pickle"])

    print("Loading the model to-" + _model_out)
    _pairwize_model = torch.load(_model_out)
    _pairwize_model.eval()

    _use_cuda = True

    test_pos_accuracy = accuracy_on_dataset(_bert_utils, _pairwize_model, positive_, _use_cuda)
    logger.info("%s: %d: Accuracy: %.10f" % ("Test-Pos-Acc", 0, test_pos_accuracy.item()))

    test_neg_accuracy = accuracy_on_dataset(_bert_utils, _pairwize_model, negative_, _use_cuda)
    logger.info("%s: %d: Accuracy: %.10f" % ("Test-Neg-Acc", 0, test_neg_accuracy.item()))

    split_feat = create_features_from_pos_neg(positive_, negative_)
    test_accuracy = accuracy_on_dataset(_bert_utils, _pairwize_model, split_feat, _use_cuda)
    logger.info("%s: %d: Accuracy: %.10f" % ("Test-Acc", 0, test_accuracy.item()))
