import logging

import torch

from src import LIBRARY_ROOT
from src.dataobjs.dataset import EcbDataSet
from src.helper_scripts.generate_pairs import get_feat_alternative
from src.pairwize_model.train import get_measurements_bool_clasification

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    context_set = "validated"
    event_validation_file = str(LIBRARY_ROOT) + "/resources/" + context_set + "/WEC_Test_Full_Event_gold_mentions_validated.json"
    ################ NO TOPICS ######################
    # positive_, negative_ = get_feat_alternative(event_validation_file)
    # logger.info('pos-' + str(len(positive_)))
    # logger.info('neg-' + str(len(negative_)))
    # features = create_features_from_pos_neg(positive_, negative_)
    ################ NO TOPICS ######################
    dataset = EcbDataSet()
    features = dataset.load_datasets(event_validation_file, -1)
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
    get_measurements_bool_clasification("ECB-Test", -1, all_labels, all_predictions)


if __name__ == '__main__':
    main()
