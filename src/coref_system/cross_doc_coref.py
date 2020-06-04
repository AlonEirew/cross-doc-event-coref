import logging
import random
from typing import List

import numpy as np
import torch

from src.configuration import Configuration, ConfigType
from src.dataobjs.cluster import Clusters
from src.dataobjs.topics import Topics
from src.coref_system.relation_extraction import HeadLemmaRelationExtractor, PairWizeRelationExtraction, RelationTypeEnum
from src.utils.clustering_utils import agglomerative_clustering, naive_clustering, ClusteringType
from src.utils.embed_utils import EmbedFromFile
from src.utils.io_utils import write_coref_scorer_results


def run_cdc_pipeline(cluster_algo, model, print_method, event_topics):
    if configuration.cluster_algo_type == ClusteringType.AgglomerativeClustering:
        for average_link_thresh in configuration.cluster_average_link_thresh:
            scorer_file = configuration.save_model_file + "_" + str(average_link_thresh)
            cluster_and_print(cluster_algo, model, print_method, event_topics, average_link_thresh, scorer_file)
    elif configuration.cluster_algo_type == ClusteringType.NaiveClustering:
        for average_link_thresh in configuration.cluster_average_link_thresh:
            for pair_thresh in configuration.cluster_pairs_thresh:
                model.pairthreshold = pair_thresh
                model.cache = dict()
                scorer_file = configuration.save_model_file + "_" + str(average_link_thresh) + "_" + str(pair_thresh)
                cluster_and_print(cluster_algo, model, print_method, event_topics, average_link_thresh, scorer_file)


def cluster_and_print(cluster_algo, model, print_method, event_topics, average_link_thresh, scorer_file):
    all_mentions = list()
    logger.info('Running event coref resolution for average_link_thresh=' + str(average_link_thresh))
    for topic in event_topics.topics_dict.values():
        logger.info("Evaluating Topic No:" + str(topic.topic_id))
        all_mentions.extend(cluster_algo(model, topic, average_link_thresh))
    logger.info("Generating scorer file-" + scorer_file)
    print_method(all_mentions, scorer_file)


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
    pairwize_model = torch.load(configuration.load_model_file)
    pairwize_model.set_embed_utils(EmbedFromFile(configuration.embed_files,
                                                 configuration.embed_config.model_size))
    pairwize_model.eval()
    return pairwize_model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    configuration = Configuration(ConfigType.Clustering)

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    print("loading configuration file-" + configuration.load_model_file)

    # _print_method = print_results
    _print_method = print_scorer_results

    _event_topics = Topics()
    _event_topics.create_from_file(configuration.mentions_file, True)

    if configuration.to_single_topic and len(_event_topics.topics_dict) > 1:
        _event_topics.to_single_topic()

    _cluster_algo = None
    _model = None
    if configuration.cluster_algo_type == ClusteringType.AgglomerativeClustering:
        _cluster_algo = agglomerative_clustering
        if configuration.cluster_extractor == RelationTypeEnum.PAIRWISE:
            _model = get_pairwise_model()
        elif configuration.cluster_extractor == RelationTypeEnum.SAME_HEAD_LEMMA:
            _model = HeadLemmaRelationExtractor()
    elif configuration.cluster_algo_type == ClusteringType.NaiveClustering:
        _cluster_algo = naive_clustering
        if configuration.cluster_extractor == RelationTypeEnum.PAIRWISE:
            _model = PairWizeRelationExtraction(get_pairwise_model(), pairthreshold=-1)
        elif configuration.cluster_extractor == RelationTypeEnum.SAME_HEAD_LEMMA:
            _model = HeadLemmaRelationExtractor()

    logger.info("Running clustering algo:" + _cluster_algo.__name__ + " with model:" + type(_model).__name__)
    run_cdc_pipeline(_cluster_algo, _model, _print_method, _event_topics)
