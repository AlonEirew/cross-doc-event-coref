import logging

import torch

from src import LIBRARY_ROOT
from src.pairwize_model.train import accuracy_on_dataset
from src.utils.bert_utils import BertFromFile
from src.utils.dataset_utils import SPLIT, DATASET, get_feat, create_features_from_pos_neg, load_pos_neg_pickle
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
    pairs_tn = list()
    pairs_fn = list()
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
            elif batch_predictions[i] == 0 and batch_label[i] == 0:
                mention1, mention2 = batch_features[i]
                pairs_tn.append(mention1.tokens_str + "=" + mention2.tokens_str)
            elif batch_predictions[i] == 1 and batch_label[i] == 0:
                mention1, mention2 = batch_features[i]
                pairs_fp.append(mention1.tokens_str + "=" + mention2.tokens_str)
            elif batch_predictions[i] == 0 and batch_label[i] == 1:
                mention1, mention2 = batch_features[i]
                pairs_fn.append(mention1.tokens_str + "=" + mention2.tokens_str)

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

    return accuracy, precision, recall, f1, pairs_tp, pairs_fp, pairs_tn, pairs_fn


if __name__ == '__main__':
    dataset = DATASET.ECB
    split = SPLIT.Train
    alpha = 4
    context_set = "single_sent_clean_kenton"

    log_param_str = "inference_" + dataset.name + ".log"
    create_logger_with_fh(log_param_str)

    _event_test_file_pos = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + dataset.name + "_" + split.name + "_Event_gold_mentions_PosPairs.pickle"
    _event_test_file_neg = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + dataset.name + "_" + split.name + "_Event_gold_mentions_NegPairs.pickle"

    _model_in = str(LIBRARY_ROOT) + "/saved_models/ECB_ECB_finetune_a4"
    # _bert_utils = BertFromFile([str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + dataset.name + "_" + split.name + "_Event_gold_mentions.pickle"])

    # pairs_tp_out_file = str(LIBRARY_ROOT) + "/reports/__pairs_eval/TP_WEC_" + dataset.name + "_" + split.name + "_paris.txt"
    # pairs_fp_out_file = str(LIBRARY_ROOT) + "/reports/__pairs_eval/FP_WEC_" + dataset.name + "_" + split.name + "_paris.txt"
    # pairs_tn_out_file = str(LIBRARY_ROOT) + "/reports/__pairs_eval/TN_WEC_" + dataset.name + "_" + split.name + "_paris.txt"
    # pairs_fn_out_file = str(LIBRARY_ROOT) + "/reports/__pairs_eval/FN_WEC_" + dataset.name + "_" + split.name + "_paris.txt"

    print("Loading the model to-" + _model_in)
    _pairwize_model = torch.load(_model_in)
    _pairwize_model.eval()

    _use_cuda = True

    # test_pos_accuracy, test_pos_precision, test_pos_recall, test_pos_f1 = accuracy_on_dataset(_bert_utils, _pairwize_model, positive_, _use_cuda)
    # logger.info("%s: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" %
    #             ("POS-Dev-Acc", test_pos_accuracy.item(), test_pos_precision, test_pos_recall, test_pos_f1))
    #
    # test_neg_accuracy, test_neg_precision, test_neg_recall, test_neg_f1 = accuracy_on_dataset(_bert_utils, _pairwize_model, negative_, _use_cuda)
    # logger.info("%s: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" %
    #             ("NEG-Dev-Acc", test_neg_accuracy.item(), test_neg_precision, test_neg_recall, test_neg_f1))

    split_feat = load_pos_neg_pickle(_event_test_file_pos, _event_test_file_neg, alpha)
    # _, _, _, _, pairs_tp, pairs_fp, pairs_tn, pairs_fn = accuracy_on_dataset("", 0, _bert_utils, _pairwize_model, split_feat, use_cuda=_use_cuda)
    accuracy_on_dataset("", 0, _pairwize_model, split_feat)

    # with open(pairs_tp_out_file, 'w') as f:
    #     for item in pairs_tp:
    #         f.write("%s\n" % item)
    #
    # with open(pairs_fp_out_file, 'w') as f:
    #     for item in pairs_fp:
    #         f.write("%s\n" % item)

    # with open(pairs_tn_out_file, 'w') as f:
    #     for item in pairs_tn:
    #         f.write("%s\n" % item)
    #
    # with open(pairs_tp_out_file, 'w') as f:
    #     for item in pairs_tp:
    #         f.write("%s\n" % item)
