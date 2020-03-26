import logging
from typing import List

import torch

from src.dataobjs.cluster import Clusters
from src.dataobjs.topics import Topics
from src.coref_system.relation_extraction import HeadLemmaRelationExtractor, PairWizeRelationExtraction, RelationTypeEnum
from src.pairwize_model import configuration
from src.utils.clustering_utils import agglomerative_clustering, naive_clustering, ClusteringType
from src.utils.io_utils import write_coref_scorer_results


def run_cdc_pipeline(cluster_algo, model, print_method, event_topics):
    for average_link_thresh in configuration.dt_average_link_thresh:
        all_mentions = list()
        logger.info('Running event coreference resolution')

        for topic in event_topics.topics_list:
            logger.info("Evaluating Topic No:" + topic.topic_id)
            all_mentions.extend(cluster_algo(model, topic, average_link_thresh))

        scorer_file = configuration.scorer_out_file + "_" + str(average_link_thresh)
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
    pairwize_model = torch.load(configuration.dt_load_model_file)
    pairwize_model.bert_utils = configuration.dt_bert_util
    pairwize_model.eval()
    return pairwize_model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("loading configuration file-" + configuration.dt_load_model_file)

    # print_method = print_results
    # print_method = print_scorer_results
    _print_method = print_scorer_results_ment

    _event_topics = Topics()
    _event_topics.create_from_file(configuration.dt_input_file, True)

    if configuration.cluster_topics and len(_event_topics.topics_list) > 1:
        _event_topics = _event_topics.to_single_topic()

    _cluster_algo = None
    _model = None
    if configuration.dt_cluster_type == ClusteringType.AgglomerativeClustering:
        _cluster_algo = agglomerative_clustering
        if configuration.dt_extractor == RelationTypeEnum.PAIRWISE:
            _model = get_pairwise_model()
        elif configuration.dt_extractor == RelationTypeEnum.SAME_HEAD_LEMMA:
            _model = HeadLemmaRelationExtractor()
    elif configuration.dt_cluster_type == ClusteringType.NaiveClustering:
        _cluster_algo = naive_clustering
        if configuration.dt_extractor == RelationTypeEnum.PAIRWISE:
            _model = PairWizeRelationExtraction(get_pairwise_model(), pairthreshold=configuration.dt_pair_thresh)
        elif configuration.dt_extractor == RelationTypeEnum.SAME_HEAD_LEMMA:
            _model = HeadLemmaRelationExtractor()

    run_cdc_pipeline(_cluster_algo, _model, _print_method, _event_topics)

################################## CREATE GOLD BASE LINE ################################
    # mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Dev_Event_gold_mentions.json')
    # for ment in mentions:
    #     ment.predicted_coref_chain = ment.coref_chain
    #
    # write_coref_scorer_results(sorted(mentions, key=lambda ment: ment.coref_chain, reverse=False),
    #                            str(LIBRARY_ROOT) + "/resources/gold_scorer/wec/CD_dev_event_mention_based.txt")
