import logging
import random
from typing import List

import numpy as np
import torch

from src import configuration
from src.dataobjs.cluster import Clusters
from src.dataobjs.topics import Topics
from src.coref_system.relation_extraction import HeadLemmaRelationExtractor, PairWizeRelationExtraction, RelationTypeEnum
from src.utils.clustering_utils import agglomerative_clustering, naive_clustering, ClusteringType
from src.utils.embed_utils import EmbedFromFile
from src.utils.io_utils import write_coref_scorer_results


def run_cdc_pipeline(cluster_algo, model, print_method, event_topics):
    if configuration.coref_cluster_type == ClusteringType.AgglomerativeClustering:
        for average_link_thresh in configuration.coref_average_link_thresh:
            scorer_file = configuration.coref_scorer_out_file + "_" + str(average_link_thresh)
            cluster_and_print(cluster_algo, model, print_method, event_topics, average_link_thresh, scorer_file)
    elif configuration.coref_cluster_type == ClusteringType.NaiveClustering:
        for average_link_thresh in configuration.coref_average_link_thresh:
            for pair_thresh in configuration.coref_pairs_thresh:
                model.pairthreshold = pair_thresh
                model.cache = dict()
                scorer_file = configuration.coref_scorer_out_file + "_" + str(average_link_thresh) + "_" + str(pair_thresh)
                cluster_and_print(cluster_algo, model, print_method, event_topics, average_link_thresh, scorer_file)


def cluster_and_print(cluster_algo, model, print_method, event_topics, average_link_thresh, scorer_file):
    all_mentions = list()
    logger.info('Running event coref resolution for average_link_thresh=' + str(average_link_thresh))
    for topic in event_topics.topics_dict.values():
        logger.info("Evaluating Topic No:" + str(topic.topic_id))
        all_mentions.extend(cluster_algo(model, topic, average_link_thresh))
    logger.info("Generating scorer file-" + scorer_file)
    print_method(all_mentions, scorer_file)


def print_results(clusters: List[Clusters], type: str, scorer_out_file):
    print('-=' + type + ' Clusters=-')
    for topic_cluster in clusters:
        print('\n\tTopic=' + topic_cluster.topic_id)
        for cluster in topic_cluster.clusters_list:
            cluster_mentions = list()
            for mention in cluster.mentions:
                mentions_dict = dict()
                mentions_dict['id'] = mention.mention_id
                mentions_dict['text'] = mention.tokens_str
                cluster_mentions.append(mentions_dict)

            print('\t\tCluster(' + str(cluster.coref_chain) + ') Mentions='
                  + str(cluster_mentions))


def print_scorer_results(all_clusters, scorer_out_file):
    all_mentions = Clusters.from_clusters_to_mentions_list(all_clusters)
    write_coref_scorer_results(all_mentions, scorer_out_file)


def print_scorer_results_ment(all_mentions, scorer_out_file):
    write_coref_scorer_results(all_mentions, scorer_out_file)


def get_pairwise_model():
    pairwize_model = torch.load(configuration.coref_load_model_file)
    pairwize_model.set_embed_utils(EmbedFromFile(configuration.coref_embed_util,
                                                 configuration.coref_embed_config.model_size))
    pairwize_model.eval()
    return pairwize_model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    print("loading configuration file-" + configuration.coref_load_model_file)

    # print_method = print_results
    # print_method = print_scorer_results
    _print_method = print_scorer_results_ment

    _event_topics = Topics()
    _event_topics.create_from_file(configuration.coref_input_file, True)

    if configuration.coref_cluster_topics and len(_event_topics.topics_dict) > 1:
        _event_topics = _event_topics.to_single_topic()

    _cluster_algo = None
    _model = None
    if configuration.coref_cluster_type == ClusteringType.AgglomerativeClustering:
        _cluster_algo = agglomerative_clustering
        if configuration.coref_extractor == RelationTypeEnum.PAIRWISE:
            _model = get_pairwise_model()
        elif configuration.coref_extractor == RelationTypeEnum.SAME_HEAD_LEMMA:
            _model = HeadLemmaRelationExtractor()
    elif configuration.coref_cluster_type == ClusteringType.NaiveClustering:
        _cluster_algo = naive_clustering
        if configuration.coref_extractor == RelationTypeEnum.PAIRWISE:
            _model = PairWizeRelationExtraction(get_pairwise_model(), pairthreshold=-1)
        elif configuration.coref_extractor == RelationTypeEnum.SAME_HEAD_LEMMA:
            _model = HeadLemmaRelationExtractor()

    logger.info("Running clustering algo:" + _cluster_algo.__name__ + " with model:" + type(_model).__name__)
    run_cdc_pipeline(_cluster_algo, _model, _print_method, _event_topics)
