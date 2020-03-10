import logging
from typing import List

from src.dataobjs.cluster import Clusters
from src.dataobjs.sieves_config import EventSievesConfiguration
from src.dataobjs.topics import Topics
from src.dt_system.computed_relation_extraction import ComputedRelationExtraction
from src.dt_system.cross_doc_sieves import run_event_coref
from src.dt_system.pairwize_relation_extraction import PairWizeRelationExtraction
from src.dt_system.relation_type_enum import RelationTypeEnum
from src.dt_system.sieves_container_init import SievesContainerInitialization
from src.pairwize_model import configuration
from src.utils.io_utils import write_coref_scorer_results


def run_example(cdc_settings, event_mentions_topics):
    event_clusters = None
    if cdc_settings.event_config.run_evaluation:
        logger.info('Running event coreference resolution')
        event_clusters = run_event_coref(event_mentions_topics, cdc_settings)

    return event_clusters


def create_example_settings():
    model_file = configuration.dt_load_model_file
    bert_file = configuration.bert_dt_file
    event_config = EventSievesConfiguration()
    event_config.sieves_order = [
        (RelationTypeEnum.PAIRWISE, 1.0)
        # (RelationTypeEnum.SAME_HEAD_LEMMA, 1.0)
    ]
    sieves_container = SievesContainerInitialization(event_coref_config=event_config, sieves_model_list=[
        PairWizeRelationExtraction(model_file, bert_file)
        # ComputedRelationExtraction()
    ])

    event_config.run_evaluation = True

    # CDCResources hold default attribute values that might need to be change,
    # (using the defaults values in this example), use to configure attributes
    # such as resources files location, output directory, resources init methods and other.
    # check in class and see if any attributes require change in your set-up
    return sieves_container


def run_cdc_pipeline(print_method, event_mentions_topics):
    cdc_settings = create_example_settings()
    event_clusters = run_example(cdc_settings, event_mentions_topics)

    print('-=Cross Document Coref Results=-')
    if cdc_settings.event_config.run_evaluation:
        print_method(event_clusters, 'Event')


def print_results(clusters: List[Clusters], type: str):
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


def print_scorer_results(all_clusters, eval_type):
    all_mentions = Clusters.from_clusters_to_mentions_list(all_clusters)
    write_coref_scorer_results(all_mentions, configuration.scorer_out_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # print_method = print_results
    print_method = print_scorer_results

    _event_mentions_topics = Topics()
    _event_mentions_topics.create_from_file(configuration.dt_input_file, True)

    if configuration.cluster_topics and len(_event_mentions_topics.topics_list) > 1:
        _event_mentions_topics = _event_mentions_topics.to_single_topic()

    run_cdc_pipeline(print_method, _event_mentions_topics)

################################## CREATE GOLD BASE LINE ################################
    # mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Dev_Event_gold_mentions.json')
    # for ment in mentions:
    #     ment.predicted_coref_chain = ment.coref_chain
    #
    # write_coref_scorer_results(sorted(mentions, key=lambda ment: ment.coref_chain, reverse=False),
    #                            str(LIBRARY_ROOT) + "/resources/gold_scorer/wec/CD_dev_event_mention_based.txt")
