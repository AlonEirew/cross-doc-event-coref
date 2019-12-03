import logging
from typing import List

from src import LIBRARY_ROOT
from src.dataobjs.cluster import Clusters
from src.dataobjs.sieves_config import EventSievesConfiguration, EntitySievesConfiguration
from src.dataobjs.topics import Topics
from src.dt_system.computed_relation_extraction import ComputedRelationExtraction
from src.dt_system.cross_doc_sieves import run_event_coref, run_entity_coref
from src.dt_system.relation_type_enum import RelationTypeEnum
from src.dt_system.sieves_container_init import SievesContainerInitialization
from src.utils.io_utils import write_coref_scorer_results


def run_example(cdc_settings):
    event_clusters = None
    if cdc_settings.event_config.run_evaluation:
        event_mentions_topics = Topics()
        event_mentions_topics.create_from_file(
            str(LIBRARY_ROOT) + "/resources/final_dataset/WEC_Dev_Event_gold_mentions.json", True)
        logger.info('Running event coreference resolution')
        event_clusters = run_event_coref(event_mentions_topics, cdc_settings)

    entity_clusters = None
    if cdc_settings.entity_config.run_evaluation:
        entity_mentions_topics = Topics()
        entity_mentions_topics.create_from_file(
            str(LIBRARY_ROOT / 'resources' / 'corpora' / 'ecb' / 'gold_json' / 'ECB_Test_Entity_gold_mentions.json'), True)
        logger.info('Running entity coreference resolution')
        entity_clusters = run_entity_coref(entity_mentions_topics, cdc_settings)

    return event_clusters, entity_clusters


def create_example_settings():
    event_config = EventSievesConfiguration()
    event_config.sieves_order = [
        (RelationTypeEnum.SAME_HEAD_LEMMA, 1.0)
    ]

    event_config.run_evaluation = True

    entity_config = EntitySievesConfiguration()
    entity_config.sieves_order = [
        (RelationTypeEnum.SAME_HEAD_LEMMA, 1.0)
    ]

    entity_config.run_evaluation = False

    # CDCResources hold default attribute values that might need to be change,
    # (using the defaults values in this example), use to configure attributes
    # such as resources files location, output directory, resources init methods and other.
    # check in class and see if any attributes require change in your set-up
    return SievesContainerInitialization(event_config, entity_config, [ComputedRelationExtraction()])


def run_cdc_pipeline(print_method):
    cdc_settings = create_example_settings()
    event_clusters, entity_clusters = run_example(cdc_settings)

    print('-=Cross Document Coref Results=-')
    if cdc_settings.event_config.run_evaluation:
        print_method(event_clusters, 'Event')

    print('################################')
    if cdc_settings.entity_config.run_evaluation:
        print_method(entity_clusters, 'Entity')


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
    if eval_type == 'Event':
        out_file = str(LIBRARY_ROOT) + "/output/event_scorer_results.txt"
    else:
        out_file = str(LIBRARY_ROOT / 'output' / 'entity_scorer_results.txt')

    all_mentions = Clusters.from_clusters_to_mentions_list(all_clusters)
    write_coref_scorer_results(all_mentions, out_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # print_method = print_results
    print_method = print_scorer_results

    run_cdc_pipeline(print_method)

################################## CREATE GOLD BASE LINE ################################
    # mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Dev_Event_gold_mentions.json')
    # for ment in mentions:
    #     ment.predicted_coref_chain = ment.coref_chain
    #
    # write_coref_scorer_results(sorted(mentions, key=lambda ment: ment.coref_chain, reverse=False),
    #                            str(LIBRARY_ROOT) + "/resources/gold_scorer/wec/CD_dev_event_mention_based.txt")
