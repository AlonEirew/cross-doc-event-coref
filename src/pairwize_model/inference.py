import logging
from os import path

import torch

from src import LIBRARY_ROOT
from src.configuration import Configuration, ConfigType
from src.pairwize_model.train import accuracy_on_dataset, run_inference
from src.utils.eval_utils import get_confusion_matrix, get_prec_rec_f1
from src.utils.log_utils import create_logger_with_fh
from src.utils.string_utils import StringUtils

logger = logging.getLogger(__name__)


def accuracy_on_dataset_local(pairwize_model, features, extract_method):
    all_labels, all_predictions = run_inference(pairwize_model, features, round_pred=False)
    round_predictions, pairs_fn, pairs_fp, pairs_tn, pairs_tp = extract_method(features, all_labels, all_predictions)
    accuracy = torch.mean((all_labels == round_predictions).float())
    tn, fp, fn, tp = get_confusion_matrix(all_labels, round_predictions)

    logger.info("###########################################################")
    logger.info("True-Positives=" + str(tp) + ", False-Positives=" + str(fp) +
                ", False-Negatives=" + str(fn) + ", True-Negatives=" + str(tn))
    logger.info("###########################################################")

    precision, recall, f1 = get_prec_rec_f1(tp, fp, fn)

    logger.info("%s: %d: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" % \
                ("EXPR" + "-Acc", 0, accuracy.item(), precision, recall, f1))

    return accuracy, precision, recall, f1, pairs_fn, pairs_fp, pairs_tn, pairs_tp


def extract_on_mention(batch_features, batch_label, batch_predictions):
    pairs_fn, pairs_fp, pairs_tn, pairs_tp = [], [], [], []
    batch_predictions_round = torch.round(batch_predictions.reshape(-1)).long()
    for i in range(0, len(batch_features)):
        if batch_predictions_round[i] == 1 and batch_label[i] == 1:
            mention1, mention2 = batch_features[i]
            pairs_tp.append((mention1.tokens_str, mention1.mention_id,
                             mention2.tokens_str, mention2.mention_id, batch_predictions[i].item()))
        elif batch_predictions_round[i] == 0 and batch_label[i] == 0:
            mention1, mention2 = batch_features[i]
            pairs_tn.append((mention1.tokens_str, mention1.mention_id,
                             mention2.tokens_str, mention2.mention_id, batch_predictions[i].item()))
        elif batch_predictions_round[i] == 1 and batch_label[i] == 0:
            mention1, mention2 = batch_features[i]
            pairs_fp.append((mention1.tokens_str, mention1.mention_id,
                             mention2.tokens_str, mention2.mention_id, batch_predictions[i].item()))
        elif batch_predictions_round[i] == 0 and batch_label[i] == 1:
            mention1, mention2 = batch_features[i]
            pairs_fn.append((mention1.tokens_str, mention1.mention_id,
                             mention2.tokens_str, mention2.mention_id, batch_predictions[i].item()))

    return batch_predictions_round, pairs_fn, pairs_fp, pairs_tn, pairs_tp


def extract_on_head(batch_features, batch_label, batch_predictions):
    pairs_fn, pairs_fp, pairs_tn, pairs_tp = [], [], [], []
    batch_predictions_round = torch.round(batch_predictions.reshape(-1)).long()
    for i in range(0, len(batch_features)):
        if batch_predictions_round[i] == 1 and batch_label[i] == 1:
            mention1, mention2 = batch_features[i]
            pairs_tp.append((mention1.mention_head_lemma, mention2.mention_head_lemma, batch_predictions[i].item()))
        elif batch_predictions_round[i] == 0 and batch_label[i] == 0:
            mention1, mention2 = batch_features[i]
            pairs_tn.append((mention1.mention_head_lemma, mention2.mention_head_lemma, batch_predictions[i].item()))
        elif batch_predictions_round[i] == 1 and batch_label[i] == 0:
            mention1, mention2 = batch_features[i]
            pairs_fp.append((mention1.mention_head_lemma, mention2.mention_head_lemma, batch_predictions[i].item()))
        elif batch_predictions_round[i] == 0 and batch_label[i] == 1:
            mention1, mention2 = batch_features[i]
            pairs_fn.append((mention1.mention_head_lemma, mention2.mention_head_lemma, batch_predictions[i].item()))

    return batch_predictions_round, pairs_fn, pairs_fp, pairs_tn, pairs_tp


def filter_same_head(list_to_filter):
    final = list()
    for tup in list_to_filter:
        ment1, ment2, score = tup
        # ment1_lst = [StringUtils.get_lemma(str) for str in ment1.lower().split(" ")]
        # ment2_lst = [StringUtils.get_lemma(str) for str in ment2.lower().split(" ")]
        if not ment1 == ment2:
            final.append(tup)

    return final


def print_tlist(file_path, list_of_tuples):
    list_of_tuples.sort(key=lambda x: x[4], reverse=True)
    with open(file_path, 'w') as fs:
        for item in list_of_tuples:
            str_t = item[0] + "||" + item[1] + "||" + item[2] + "||" + item[3] + "||" + str(item[4])
            fs.write("%s\n" % str_t)


if __name__ == '__main__':
    configuration = Configuration(ConfigType.Inference)
    dataset = configuration.test_dataset
    split = configuration.split
    ratio = configuration.ratio
    _model_in = configuration.load_model_file

    log_param_str = "inference_" + dataset.name + ".log"
    create_logger_with_fh(log_param_str)

    _event_test_file_pos = configuration.event_test_file_pos
    _event_test_file_neg = configuration.event_test_file_neg

    basename = path.basename(path.splitext(_model_in)[0])
    pairs_tp_out_file = str(LIBRARY_ROOT) + "/reports/pairs_final/TP_" + basename + "_" + split.name + "_paris.txt"
    pairs_fp_out_file = str(LIBRARY_ROOT) + "/reports/pairs_final/FP_" + basename + "_" + split.name + "_paris.txt"
    pairs_tn_out_file = str(LIBRARY_ROOT) + "/reports/pairs_final/TN_" + basename + "_" + split.name + "_paris.txt"
    pairs_fn_out_file = str(LIBRARY_ROOT) + "/reports/pairs_final/FN_" + basename + "_" + split.name + "_paris.txt"

    print("Loading the model to-" + _model_in)
    _pairwize_model = torch.load(_model_in)
    _pairwize_model.eval()

    _use_cuda = True

    positive_ = dataset.load_pos_pickle(_event_test_file_pos)
    negative_ = dataset.load_neg_pickle(_event_test_file_neg)
    split_feat = dataset.create_features_from_pos_neg(positive_, negative_)

    # test_pos_accuracy, test_pos_precision, test_pos_recall, test_pos_f1 = accuracy_on_dataset("", 0, _pairwize_model, positive_)
    # logger.info("%s: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" %
    #             ("POS-Dev-Acc", test_pos_accuracy.item(), test_pos_precision, test_pos_recall, test_pos_f1))
    #
    # test_neg_accuracy, test_neg_precision, test_neg_recall, test_neg_f1 = accuracy_on_dataset("", 0, _pairwize_model, negative_)
    # logger.info("%s: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" %
    #             ("NEG-Dev-Acc", test_neg_accuracy.item(), test_neg_precision, test_neg_recall, test_neg_f1))

    _, _, _, _, pairs_fn, pairs_fp, pairs_tn, pairs_tp = accuracy_on_dataset_local(_pairwize_model, split_feat, extract_on_mention)
    # accuracy_on_dataset("", 0, _pairwize_model, split_feat)

    # pairs_tp = filter_same_head(pairs_tp)
    # pairs_fp = filter_same_head(pairs_fp)
    # pairs_tn = filter_same_head(pairs_tn)
    # pairs_fn = filter_same_head(pairs_fn)

    print_tlist(pairs_tp_out_file, pairs_tp)
    print_tlist(pairs_fp_out_file, pairs_fp)
    print_tlist(pairs_tn_out_file, pairs_tn)
    print_tlist(pairs_fn_out_file, pairs_fn)
