"""

Usage:
    generate_pairs_predictions.py --tmf=<TestMentionsFile> --tef=<TestEmbedFile> --mf=<ModelFile> --out=<OurPredFile>
            [--cuda=<y>] [--topic=<type>] [--em=<ExtractMethod>]

Options:
    -h --help                   Show this screen.
    --cuda=<y>                  True/False - Whether to use cuda device or not [default: True]
    --topic=<type>              subtopic/topic/corpus - relevant only to ECB+, take pairs only from the same sub-topic, topic or corpus wide [default: subtopic]
    --em=<ExtractMethod>        pairwize/head_lemma/exact_string - model type to run [default: pairwize]
"""

import logging
import ntpath
import pickle
from itertools import product

import numpy as np
import random
import torch
from docopt import docopt

from src.coref_system.relation_extraction import HeadLemmaRelationExtractor, RelationTypeEnum
from src.coref_system.relation_extraction import RelationExtraction
from src.dataobjs.dataset import TopicConfig
from src.dataobjs.topics import Topics
from src.utils.embed_utils import EmbedFromFile

logger = logging.getLogger(__name__)
MAX_ALLOWED_BATCH_SIZE = 20000


def generate_pred_matrix(inference_model, topic):
    all_pairs = list(product(topic.mentions, repeat=2))
    pairs_chunks = [all_pairs]
    if len(all_pairs) > MAX_ALLOWED_BATCH_SIZE:
        pairs_chunks = [all_pairs[i:i + MAX_ALLOWED_BATCH_SIZE] for i in
                        range(0, len(all_pairs), MAX_ALLOWED_BATCH_SIZE)]
    predictions = np.empty(0)
    with torch.no_grad():
        for chunk in pairs_chunks:
            chunk_predictions, _ = inference_model.predict(chunk, bs=len(chunk))
            predictions = np.append(predictions, chunk_predictions.detach().cpu().numpy())
    predictions = 1 - predictions
    pred_matrix = predictions.reshape(len(topic.mentions), len(topic.mentions))
    return pred_matrix


def predict_and_save():
    all_predictions = list()
    for topic in _event_topics.topics_dict.values():
        logger.info("Evaluating Topic No:" + str(topic.topic_id))
        all_predictions.append((topic, generate_pred_matrix(_model, topic)))
    logger.info("Generating prediction file-" + _outfile)
    pickle.dump(all_predictions, open(_outfile, "wb"))


def get_pairwise_model():
    pairwize_model = torch.load(_model_file)
    pairwize_model.set_embed_utils(EmbedFromFile([_embed_file]))

    if _use_cuda:
        pairwize_model.cuda()

    pairwize_model.eval()
    return pairwize_model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    print(_arguments)
    _mentions_file = _arguments.get("--tmf")
    _embed_file = _arguments.get("--tef")
    _model_file = _arguments.get("--mf")
    _outfile = _arguments.get("--out")
    _use_cuda = True if _arguments.get("--cuda").lower() == "true" else False
    _topic_arg = _arguments.get("--topic")
    _extract_method_str = _arguments.get("--em")

    _topic_config = Topics.get_topic_config(_topic_arg)
    _extract_method = RelationExtraction.get_extract_method(_extract_method_str)

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    logger.info("loading model from-" + ntpath.basename(_model_file))
    _event_topics = Topics()
    _event_topics.create_from_file(_mentions_file, True)

    if _topic_config == TopicConfig.Corpus and len(_event_topics.topics_dict) > 1:
        _event_topics.to_single_topic()

    _cluster_algo = None
    _model = None
    if _extract_method == RelationTypeEnum.PAIRWISE:
        _model = get_pairwise_model()
    elif _extract_method == RelationTypeEnum.SAME_HEAD_LEMMA:
        _model = HeadLemmaRelationExtractor()

    logger.info("Running agglomerative clustering with model:" + type(_model).__name__)
    predict_and_save()
