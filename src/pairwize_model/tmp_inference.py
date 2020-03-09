import logging

import torch

from src import LIBRARY_ROOT
from src.dataobjs.dataset import WecDataSet
from src.utils.bert_utils import BertFromFile
from src.utils.log_utils import create_logger_with_fh

logger = logging.getLogger(__name__)


def accuracy_on_dataset_local(message, itr, bert_utils, pairwize_model, features, use_cuda):
    dataset_size = len(features)
    batch_size = 10000
    end_index = batch_size
    labels = list()
    predictions = list()
    pairs_tp = list()
    pairs_fp = list()
    for start_index in range(0, dataset_size, batch_size):
        if end_index > dataset_size:
            end_index = dataset_size

        batch_features = features[start_index:end_index].copy()
        bs = end_index - start_index
        batch, batch_label = pairwize_model.get_bert_rep(batch_features, bert_utils, use_cuda, bs)
        batch_predictions = pairwize_model.predict(batch)

        for i in range(0, len(batch_features)):
            if batch_predictions[i] == 1 and batch_label[i] == 1:
                mention1, mention2 = batch_features[i]
                pairs_tp.append(mention1.tokens_str + "=" + mention2.tokens_str)
            elif batch_predictions[i] == 1 and batch_label[i] == 0:
                mention1, mention2 = batch_features[i]
                pairs_fp.append(mention1.tokens_str + "=" + mention2.tokens_str)

        predictions.append(batch_predictions)
        labels.append(batch_label)
        end_index += batch_size

    all_labels = torch.cat(labels).bool()
    all_predictions = torch.cat(predictions).bool()

    accuracy = torch.mean((all_labels == all_predictions).float())

    tp = torch.sum(all_labels & all_predictions).float().item()
    tn = torch.sum(~all_labels & ~all_predictions).float().item()
    fn = torch.sum(all_labels & ~all_predictions).float().item()
    fp = torch.sum(~all_labels & all_predictions).float().item()

    logger.info("###########################################################")
    logger.info("True-Positives=" + str(tp) + ", False-Positives=" + str(fp) +
                ", False-Negatives=" + str(fn) + ", True-Negatives=" + str(tn))
    logger.info("###########################################################")

    tpfp = tp + fp
    tpfn = tp + fn
    precision, recall, f1 = (0.0, 0.0, 0.0)
    if tpfp != 0:
        precision = tp / tpfp
    if tpfn != 0:
        recall = tp / tpfn
    if precision != 0 or recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1, pairs_tp, pairs_fp


if __name__ == '__main__':
    log_param_str = "inference_validated.log"
    create_logger_with_fh(log_param_str)

    _model_in = str(LIBRARY_ROOT) + "/saved_models/ECB_WEC_best_trained_model_a6"
    _bert_utils = BertFromFile([
        str(LIBRARY_ROOT) + "/resources/single_sent_clean_kenton/WEC_Dev_Event_gold_mentions.pickle",
        str(LIBRARY_ROOT) + "/resources/single_sent_clean_kenton/WEC_Test_Event_gold_mentions.pickle"
    ])

    print("Loading the model to-" + _model_in)
    _pairwize_model = torch.load(_model_in)
    _pairwize_model.eval()

    _use_cuda = True

    split_feat = WecDataSet().load_datasets(str(LIBRARY_ROOT) + "/resources/validated/WEC_CLEAN_JOIN.json", -1)
    accuracy, precision, recall, f1, pairs_tp, pairs_fp = accuracy_on_dataset_local("", 0, _bert_utils, _pairwize_model, split_feat, use_cuda=_use_cuda)

    pairs_tp_out_file = str(
        LIBRARY_ROOT) + "/reports/pairs_eval/TP_ECB_WEC_CLEAN_JOIN_paris.txt"
    pairs_fp_out_file = str(
        LIBRARY_ROOT) + "/reports/pairs_eval/FP_ECB_WEC_CLEAN_JOIN_paris.txt"
    with open(pairs_tp_out_file, 'w') as f:

        for item in pairs_tp:
            f.write("%s\n" % item)

    with open(pairs_fp_out_file, 'w') as f:
        for item in pairs_fp:
            f.write("%s\n" % item)
