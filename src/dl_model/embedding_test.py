import random

import numpy as np
import torch

from src import LIBRARY_ROOT


logger = logging.getLogger(__name__)


def run_train(train_file, dev_file, bert_file, learning_rate, iterations, model_out, use_cuda):

    print('Create Features:')
    train_feat, dev_feat = get_feat(train_file, dev_file)

    bert = ElmoEmbeddingOffline(bert_file)
    model = CorefModel(MODEL_SIZE)

    if use_cuda and torch.cuda.is_available():
        model.cuda()

    train_classifier(model, train_feat, dev_feat, learning_rate, iterations, bert, use_cuda)

    if model_out:
        torch.save(model, model_out)


if __name__ == '__main__':
    EMBED_SIZE = 1024
    MODEL_SIZE = 1024

    _event_train_file = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Small_Event_gold_mentions.json'
    _event_dev_file = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Small_Event_gold_mentions.json'

    _bert_file = str(LIBRARY_ROOT) + '/resources/preprocessed_external_features/embedded/wiki_small_embed_bert_all_layers.pickle'
    _model_out = str(LIBRARY_ROOT) + '/saved_models/wiki_trained_model'

    _learning_rate = 0.01
    _iterations = 1
    _joint = False
    _type = 'event'

    _use_cuda = False  # args.cuda in ['True', 'true', 'yes', 'Yes']
    if _use_cuda:
        print(torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(1)

    random.seed(1)
    np.random.seed(1)

    run_train(_event_train_file, _event_dev_file, _bert_file, _learning_rate, _iterations, _model_out, _use_cuda)
