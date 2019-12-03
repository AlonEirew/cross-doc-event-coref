import logging

import torch

from src import LIBRARY_ROOT
from src.pairwize_model.train import get_measurements
from src.utils.dataset_utils import get_feat, DATASET, SPLIT, create_features_from_pos_neg, load_datasets

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    context_set = "final_dataset"
    event_validation_file = str(LIBRARY_ROOT) + "/resources/" + context_set + "/WEC_Dev_Event_gold_mentions.json"

    features = load_datasets(event_validation_file, 20, DATASET.WEC)
    accuracy_on_dataset(features)


def accuracy_on_dataset(features):
    labels = list()
    predictions = list()
    for mention1, mention2 in features:
        pred = 1 if mention1.mention_head_lemma.lower() == mention2.mention_head_lemma.lower() else 0
        gold_label = 1 if mention1.coref_chain == mention2.coref_chain else 0
        labels.append(pred)
        predictions.append(gold_label)

    all_labels = torch.tensor(labels).bool()
    all_predictions = torch.tensor(predictions).bool()
    get_measurements("ECB-Test", -1, all_labels, all_predictions)


if __name__ == '__main__':
    main()
