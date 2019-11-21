import logging

import torch

from src import LIBRARY_ROOT
from src.pairwize_model.train import get_measurements
from src.utils.dataset_utils import get_feat, DATASET, SPLIT, create_features_from_pos_neg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    context_set = "final_set_clean_min"
    event_validation_file = str(LIBRARY_ROOT) + "/resources/" + context_set + "/ECB_Train_Event_gold_mentions.json"

    positive_, negative_ = get_feat(event_validation_file, 1, SPLIT.Train, DATASET.ECB)
    features = create_features_from_pos_neg(positive_, negative_)
    accuracy_on_dataset(features)


def accuracy_on_dataset(features):
    labels = list()
    predictions = list()
    for mention1, mention2 in features:
        pred = 1 if mention1.mention_head == mention2.mention_head else 0
        gold_label = 1 if mention1.coref_chain == mention2.coref_chain else 0
        labels.append(pred)
        predictions.append(gold_label)

    all_labels = torch.tensor(labels).bool()
    all_predictions = torch.tensor(predictions).bool()
    accuracy, precision, recall, f1 = get_measurements(all_labels, all_predictions)

    dev_report = "%s: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" % \
                 ("Dev-Acc", accuracy.item(), precision, recall, f1)

    logger.info(dev_report)


if __name__ == '__main__':
    main()
