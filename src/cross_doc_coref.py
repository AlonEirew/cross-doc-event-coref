"""

Usage:
    cross_doc_coref.py --tmf=<TestMentionsFile> --tef=<TestEmbedFile> --mf=<ModelFile> [--cuda=<y>] [--topic=<type>]
            [--em=<ExtractMethod>] [--alt=<AverageLinkThresh>]

Options:
    -h --help                   Show this screen.
    --cuda=<y>                  True/False - Whether to use cuda device or not [default: True]
    --topic=<type>              subtopic/topic/corpus - relevant only to ECB+, take pairs only from the same sub-topic, topic or corpus wide [default: subtopic]
    --em=<ExtractMethod>        pairwize/head_lemma/exact_string - model type to run [default: pairwize]
    --alt=<AverageLinkThresh>   The link threshold for the clustering algorithm [default: 0.7]
"""

import logging
import ntpath
import random

import numpy as np
import torch

from docopt import docopt
from dataobjs.cluster import Clusters
from dataobjs.topics import Topics
from dataobjs.dataset import TopicConfig
from coref_system.relation_extraction import HeadLemmaRelationExtractor, RelationTypeEnum
from utils.clustering_utils import agglomerative_clustering
from utils.embed_utils import EmbedFromFile
from utils.io_utils import write_coref_scorer_results
from coref_system.relation_extraction import RelationExtraction


logger = logging.getLogger(__name__)


def cluster_and_print():
    all_mentions = list()
    logger.info('Running event coref resolution for average_link_thresh=' + str(_average_link_thresh))
    for topic in _event_topics.topics_dict.values():
        logger.info("Evaluating Topic No:" + str(topic.topic_id))
        all_mentions.extend(agglomerative_clustering(_model, topic, _average_link_thresh))
    logger.info("Generating scorer file-" + _scorer_file)
    _print_method(all_mentions, _scorer_file)


def print_results(all_mentions, scorer_out_file):
    all_clusters = Clusters.from_mentions_to_predicted_clusters(all_mentions)
    for cluster_id, cluster in all_clusters.items():
        if 'Singleton' in cluster[0].coref_chain and len(cluster) == 1:
            continue

        print('\n\tCluster=' + str(cluster_id))
        for mention in cluster:
            mentions_dict = dict()
            mentions_dict['id'] = mention.mention_id
            mentions_dict['text'] = mention.tokens_str
            mentions_dict['gold'] = mention.coref_chain

            if mention.tokens_number[0] >= 10 and (mention.tokens_number[-1] + 10) < len(mention.mention_context):
                id_start = mention.tokens_number[0] - 10
                id_end = mention.tokens_number[-1] + 10
            elif mention.tokens_number[0] < 10 and (mention.tokens_number[-1] + 10) < len(mention.mention_context):
                id_start = 0
                id_end = mention.tokens_number[-1] + 10
            elif mention.tokens_number[0] >= 10 and (mention.tokens_number[-1] + 10) >= len(mention.mention_context):
                id_start = mention.tokens_number[0] - 10
                id_end = len(mention.mention_context)
            else:
                id_start = 0
                id_end = len(mention.mention_context)

            before = " ".join(mention.mention_context[id_start:mention.tokens_number[0]])
            after = " ".join(mention.mention_context[mention.tokens_number[-1] + 1:id_end])
            mention_txt = " <" + mention.tokens_str + "> "
            mentions_dict['context'] = before + mention_txt + after

            print('\t\tCluster(' + str(cluster_id) + ') Mentions='
                  + str(mentions_dict))


def print_scorer_results(all_mentions, scorer_out_file):
    write_coref_scorer_results(all_mentions, scorer_out_file)


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
    _mentions_file = _arguments.get("--tmf")
    _embed_file = _arguments.get("--tef")
    _model_file = _arguments.get("--mf")
    _use_cuda = True if _arguments.get("--cuda").lower() == "true" else False
    _topic_arg = _arguments.get("--topic")
    _extract_method_str = _arguments.get("--em")
    _average_link_thresh = float(_arguments.get("--alt"))
    logger.info(_arguments)

    _topic_config = Topics.get_topic_config(_topic_arg)
    _extract_method = RelationExtraction.get_extract_method(_extract_method_str)

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    logger.info("loading model from-" + ntpath.basename(_model_file))

    # _print_method = print_results
    _print_method = print_scorer_results

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
    _scorer_file = _model_file + "_" + str(_average_link_thresh)
    cluster_and_print()
